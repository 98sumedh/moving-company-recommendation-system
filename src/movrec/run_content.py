"""Step 4: move-profile vectors + cosine kNN, compared with earlier models on the same cases.

Tunes k and the similarity power on val, runs a drop-one feature-group ablation on val,
refits on train+val and scores test once. Writes reports/content.md and results/content.json.
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from movrec import DATA_DIR, PROJECT_ROOT
from movrec.als import ALSRecommender, ImplicitALS
from movrec.baselines import SegmentPopularity
from movrec.content import ContentKNN
from movrec.data import load_experiment_config, load_modeling_data
from movrec.evaluate import make_cases, paired_diff_ci, summarize
from movrec.report_utils import markdown_table, test_tables
from movrec.run_baselines import BANNER, evaluate_model, target_columns
from movrec.split import time_split


def run(data_dir: Path = DATA_DIR, verify: bool = True) -> dict:
    cfg = load_experiment_config()
    ev, bl, cc, ac = cfg["evaluation"], cfg["baselines"], cfg["content"], cfg["als"]
    prior_results = {name: json.loads((PROJECT_ROOT / "results" / f"{name}.json").read_text())["target_sets"]
                     for name in ("baselines", "als")}
    data = load_modeling_data(data_dir, verify=verify)
    split = time_split(data.bookings, **cfg["split"])
    keys, map_col = bl["segment_keys"], f"map@{ev['map_k']}"
    train_val = split.train | split.val
    groups = cc["groups"]
    results = {"target_sets": {}}

    for target_set in ev["target_sets"]:
        matrix = data.matrix[:, target_columns(data, target_set)]
        val_cases = make_cases(matrix[split.val], np.random.default_rng(ev["seed"]))
        test_cases = make_cases(matrix[split.test], np.random.default_rng(ev["seed"] + 1))
        args = (data.bookings, matrix)
        seg_alpha = prior_results["baselines"][target_set]["chosen"]["segment_alpha"]

        def val_score(model):
            pc = evaluate_model(model, split.train, split.val, *args, val_cases, cfg)
            return pc[map_col].mean(), pc[pc["cart_size"] == 0][map_col].mean()

        # --- 1. equal group weights: tune k and power
        tuning = []
        for k, power in itertools.product(cc["k_grid"], cc["power_grid"]):
            overall, empty = val_score(ContentKNN(groups, k, power))
            tuning.append({"weights": "equal", "k": k, "power": power, map_col: overall, f"{map_col} (empty cart)": empty})
        best = max(tuning, key=lambda t: t[map_col])
        k_eq, power = best["k"], best["power"]

        # --- 2. drop-one feature-group ablation (equal weights)
        ablation = [{"features": "all groups", map_col: best[map_col], f"{map_col} (empty cart)": best[f"{map_col} (empty cart)"]}]
        for g in groups:
            overall, empty = val_score(ContentKNN([x for x in groups if x != g], k_eq, power))
            ablation.append({"features": f"without {g}", map_col: overall, f"{map_col} (empty cart)": empty})
        overall, empty = val_score(ContentKNN(["core"], k_eq, power))
        ablation.append({"features": "core only", map_col: overall, f"{map_col} (empty cart)": empty})
        overall, empty = val_score(SegmentPopularity(keys, seg_alpha))
        ablation.append({"features": "(segment popularity, for reference)", map_col: overall, f"{map_col} (empty cart)": empty})

        # --- 3. tune group weights by coordinate search, then retune k
        weights = {g: (1.0 if g == "core" else 0.0) for g in groups}
        current = val_score(ContentKNN(groups, k_eq, power, group_weights=weights))[0]
        weight_search = [{"step": "core only", **weights, map_col: current}]
        for g in groups:
            if g == "core":
                continue
            for w in cc["group_weight_grid"]:
                trial = {**weights, g: w}
                score = val_score(ContentKNN(groups, k_eq, power, group_weights=trial))[0]
                if score > current:
                    current, weights = score, trial
            weight_search.append({"step": f"set {g}", **weights, map_col: current})
        for kk in cc["k_grid"]:
            overall, empty = val_score(ContentKNN(groups, kk, power, group_weights=weights))
            tuning.append({"weights": "tuned", "k": kk, "power": power, map_col: overall, f"{map_col} (empty cart)": empty})
        k_tuned = max((t for t in tuning if t["weights"] == "tuned"), key=lambda t: t[map_col])["k"]

        # --- test once
        a = prior_results["als"][target_set]["chosen"]["als_plus_segment_prior"]
        models = {
            "segment_popularity": SegmentPopularity(keys, seg_alpha),
            "als_plus_segment_prior": ALSRecommender(
                ImplicitALS(a["factors"], a["reg"], a["alpha"], ac["iterations"], seed=0),
                prior=SegmentPopularity(keys, seg_alpha), prior_weight=a["prior_weight"]),
            "content_knn_equal_weights": ContentKNN(groups, k_eq, power),
            "content_knn": ContentKNN(groups, k_tuned, power, group_weights=weights),
        }
        per_case = {name: evaluate_model(m, train_val, split.test, *args, test_cases, cfg) for name, m in models.items()}
        boot = np.random.default_rng(ev["seed"] + 4)
        ref = per_case["segment_popularity"][map_col].to_numpy()
        test = {name: {"summary": summarize(pc, ev["cart_size_slices"]).reset_index(names="cart_size").to_dict("records"),
                       "diff_vs_segment_popularity": paired_diff_ci(pc[map_col].to_numpy(), ref, boot)}
                for name, pc in per_case.items()}
        empty_rows = per_case["segment_popularity"]["cart_size"].to_numpy() == 0
        empty_diff = paired_diff_ci(per_case["content_knn"][map_col].to_numpy()[empty_rows], ref[empty_rows], boot)
        results["target_sets"][target_set] = {
            "tuning": tuning, "ablation": ablation, "weight_search": weight_search,
            "chosen": {"k_equal_weights": k_eq, "k": k_tuned, "power": power, "group_weights": weights},
            "test": test, "empty_cart_diff": empty_diff}
    return results


def render(results: dict, cfg: dict) -> str:
    map_col = f"map@{cfg['evaluation']['map_k']}"
    out = ["# Step 4: move-profile vectors and cosine nearest neighbours", "", BANNER, "",
           "Each booking becomes a vector of what is known at booking time: move type and size "
           "(**core**), floors, stairs and elevators (**access**), regions and distance (**route**), "
           "month, weekday, lead time and arrival window (**calendar**), and estimate method, "
           "payment method and crew (**quote**). Categoricals are one-hot, numerics standardized, each "
           "group scaled to equal weight, rows L2-normalized. The truck is never a feature.", "",
           "A new booking is matched to its k most similar past bookings by cosine similarity; "
           "services are scored by the similarity-weighted share of neighbours who bought them. "
           "The partial cart is not used, so this is the model for **empty carts**. Same split, cases "
           "and tuning rules as earlier steps.", "",
           "The first run used equal group weights and lost to segment popularity. The ablation showed "
           "weak groups diluting `core`, so group weights are now tuned on val (both versions are "
           "reported on test).", ""]
    for ts, r in results["target_sets"].items():
        label = "all 45 services" if ts == "all" else "add-ons only, truck excluded"
        tun = pd.DataFrame(r["tuning"]).sort_values(map_col, ascending=False)
        d, lo, hi = r["empty_cart_diff"]
        out += [f"## Targets: {label}", "", "### Validation tuning", "", markdown_table(tun, "{:.4f}"), "",
                "### Feature-group ablation (validation, equal weights)", "", markdown_table(pd.DataFrame(r["ablation"]), "{:.4f}"), "",
                "### Group-weight search (validation)", "", markdown_table(pd.DataFrame(r["weight_search"]), "{:.4f}"), "",
                f"Chosen: equal weights k = {r['chosen']['k_equal_weights']}; tuned weights "
                f"{r['chosen']['group_weights']}, k = {r['chosen']['k']}; power = {r['chosen']['power']}.", "",
                "### Test", "", *test_tables(r["test"], map_col),
                f"Empty carts only: content kNN vs segment popularity {map_col} {d:+.3f} (95% CI {lo:+.3f} to {hi:+.3f}).", ""]
    return "\n".join(out)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Run step 4 (content kNN).")
    parser.add_argument("--data", type=Path, default=DATA_DIR)
    args = parser.parse_args(argv)
    results = run(args.data)
    (PROJECT_ROOT / "results" / "content.json").write_text(json.dumps(results, indent=2, default=float) + "\n")
    report = PROJECT_ROOT / "reports" / "content.md"
    report.write_text(render(results, load_experiment_config()))
    print(f"wrote {report}")


if __name__ == "__main__":
    main()
