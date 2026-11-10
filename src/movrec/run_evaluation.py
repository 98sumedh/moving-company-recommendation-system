"""Step 6: one evaluation of every model on the same test cases.

Every model is rebuilt from its chosen settings (results/*.json from steps 2-5), fit on
train+val, and scored on the test partial carts. Adds category hit rate, slices by cart
size and move type, robustness to the reveal seed, two generator-aware ceilings, and a
recap of the ablations. Writes reports/evaluation.md and results/evaluation.json.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from movrec import DATA_DIR, PROJECT_ROOT
from movrec.als import ImplicitALS
from movrec.assumptions import load_assumptions
from movrec.baselines import CoOccurrence, GlobalPopularity, SegmentPopularity
from movrec.catalog import load_catalog
from movrec.content import ContentKNN
from movrec.data import load_experiment_config, load_modeling_data
from movrec.evaluate import make_cases, paired_diff_ci, per_case_metrics, summarize
from movrec.hybrid import blend
from movrec.oracle import oracle_scores
from movrec.report_utils import markdown_table
from movrec.run_baselines import BANNER, target_columns
from movrec.run_hybrid import ALSScores
from movrec.split import time_split

MAIN_MODELS = ["global_popularity", "segment_popularity", "cooccurrence", "als", "als_plus_segment_prior",
               "content_knn_equal_weights", "content_knn", "hybrid_global", "hybrid_by_cart_size"]
ORACLES = ["ceiling_profile_only", "ceiling_true_segment"]
KEY_MODELS = ["segment_popularity", "als_plus_segment_prior", "content_knn", "hybrid_global", *ORACLES]
ROBUSTNESS_MODELS = ["segment_popularity", "als_plus_segment_prior", "hybrid_global"]
ROBUSTNESS_SEEDS = [11, 12, 13]   # offsets added to the evaluation seed


def fit_components(chosen: dict, cfg: dict, bookings: pd.DataFrame, matrix: np.ndarray) -> dict:
    keys, it = cfg["baselines"]["segment_keys"], cfg["als"]["iterations"]
    a_mix, a_pure, c = chosen["als_plus_segment_prior"], chosen["als"], chosen["content"]
    comps = {
        "glob": GlobalPopularity(),
        "seg": SegmentPopularity(keys, chosen["segment_alpha"]),
        "cooc": CoOccurrence(keys, chosen["segment_alpha"], prior_weight=0.0),
        "als": ALSScores(ImplicitALS(a_mix["factors"], a_mix["reg"], a_mix["alpha"], it, seed=0)),
        "als_pure": ALSScores(ImplicitALS(a_pure["factors"], a_pure["reg"], a_pure["alpha"], it, seed=0)),
        "knn": ContentKNN(cfg["content"]["groups"], c["k"], c["power"], group_weights=c["group_weights"]),
        "knn_eq": ContentKNN(cfg["content"]["groups"], c["k_equal_weights"], c["power"]),
    }
    return {name: m.fit(bookings, matrix) for name, m in comps.items()}


def model_scores(comps: dict, chosen: dict, bookings: pd.DataFrame, cases, slices) -> dict:
    parts = {name: m.score(bookings, cases.revealed) for name, m in comps.items()}
    empty = cases.cart_size == 0
    als_pure = parts["als_pure"].copy()
    als_pure[empty] = parts["glob"][empty]
    w_mix = chosen["als_plus_segment_prior"]["prior_weight"]
    scores = {
        "global_popularity": parts["glob"],
        "segment_popularity": parts["seg"],
        "cooccurrence": parts["cooc"] + chosen["cooc_prior_weight"] * parts["seg"],
        "als": als_pure,
        "als_plus_segment_prior": parts["als"] + w_mix * parts["seg"],
        "content_knn_equal_weights": parts["knn_eq"],
        "content_knn": parts["knn"],
    }
    for name, w in chosen["hybrid"].items():
        scores[name] = blend(parts, w, cases.cart_size, slices)
    return scores


def run(data_dir: Path = DATA_DIR, verify: bool = True) -> dict:
    cfg = load_experiment_config()
    ev = cfg["evaluation"]
    slices, map_k, ks = ev["cart_size_slices"], ev["map_k"], ev["ks"]
    map_col = f"map@{map_k}"
    prior = {n: json.loads((PROJECT_ROOT / "results" / f"{n}.json").read_text())["target_sets"]
             for n in ("baselines", "als", "content", "hybrid")}
    A, catalog = load_assumptions(), load_catalog()
    data = load_modeling_data(data_dir, verify=verify)
    segments = pd.read_csv(Path(data_dir) / "_hidden" / "segments.csv").set_index("booking_id")["segment"]
    split = time_split(data.bookings, **cfg["split"])
    train_val = split.train | split.val
    fit_b = data.bookings[train_val].reset_index(drop=True)
    test_b = data.bookings[split.test].reset_index(drop=True)
    results = {"target_sets": {}}

    for target_set in ev["target_sets"]:
        cols = target_columns(data, target_set)
        items = [s for s, keep in zip(data.items, cols) if keep]
        categories = data.services.set_index("id").loc[items, "category"].to_numpy()
        matrix = data.matrix[:, cols]
        chosen = {
            "segment_alpha": prior["baselines"][target_set]["chosen"]["segment_alpha"],
            "cooc_prior_weight": prior["baselines"][target_set]["chosen"]["cooc_prior_weight"],
            **prior["als"][target_set]["chosen"],
            "content": prior["content"][target_set]["chosen"],
            "hybrid": prior["hybrid"][target_set]["weights"],
        }
        comps = fit_components(chosen, cfg, fit_b, matrix[train_val])

        def evaluate(case_seed):
            cases = make_cases(matrix[split.test], np.random.default_rng(case_seed))
            b = test_b.iloc[cases.rows].reset_index(drop=True)
            return cases, b, model_scores(comps, chosen, b, cases, slices)

        # --- main evaluation (same cases as steps 2-5)
        cases, case_b, scores = evaluate(ev["seed"] + 1)
        scores["ceiling_profile_only"] = oracle_scores(A, catalog, case_b, cases.revealed, items)
        scores["ceiling_true_segment"] = oracle_scores(A, catalog, case_b, cases.revealed, items,
                                                       segment=case_b["booking_id"].map(segments).to_numpy())
        per_case = {name: per_case_metrics(s, cases, ks=ks, map_k=map_k, categories=categories)
                    .assign(move_type=case_b["move_type"].to_numpy()) for name, s in scores.items()}

        boot = np.random.default_rng(ev["seed"] + 6)
        ref = per_case["segment_popularity"][map_col].to_numpy()
        seg_map, ceiling_map = ref.mean(), per_case["ceiling_profile_only"][map_col].mean()
        main = []
        for name in MAIN_MODELS + ORACLES:
            pc = per_case[name]
            d, lo, hi = paired_diff_ci(pc[map_col].to_numpy(), ref, boot)
            row = {"model": name, **{c: pc[c].mean() for c in pc.columns if c not in ("cart_size", "move_type")}}
            row["vs segment pop"] = d
            row["95% CI"] = f"{lo:+.3f} to {hi:+.3f}"
            row["headroom captured"] = (row[map_col] - seg_map) / (ceiling_map - seg_map)
            main.append(row)

        by_cart = {name: summarize(per_case[name].drop(columns="move_type"), slices)[map_col].to_dict() for name in KEY_MODELS}
        by_move = {name: per_case[name].groupby("move_type")[map_col].mean().to_dict() for name in KEY_MODELS}
        move_counts = per_case["segment_popularity"]["move_type"].value_counts().to_dict()

        # --- robustness: different reveal seeds
        robust = []
        for offset in ROBUSTNESS_SEEDS:
            r_cases, _, r_scores = evaluate(ev["seed"] + offset)
            m = {n: per_case_metrics(r_scores[n], r_cases, ks=ks, map_k=map_k)[map_col].to_numpy() for n in ROBUSTNESS_MODELS}
            d, lo, hi = paired_diff_ci(m["hybrid_global"], m["segment_popularity"], boot)
            robust.append({"reveal seed": ev["seed"] + offset, **{n: v.mean() for n, v in m.items()},
                           "hybrid_global vs segment pop": f"{d:+.3f} ({lo:+.3f} to {hi:+.3f})"})

        results["target_sets"][target_set] = {
            "n_items": len(items), "main": main, "by_cart_size": by_cart, "by_move_type": by_move,
            "move_type_counts": move_counts, "robustness": robust,
            "ablation_features_val": prior["content"][target_set]["ablation"],
            "ablation_components_test": prior["hybrid"][target_set]["ablation"],
        }
    return results


def render(results: dict, cfg: dict) -> str:
    map_col = f"map@{cfg['evaluation']['map_k']}"
    out = ["# Step 6: evaluation", "", BANNER, "",
           "Every model from steps 2-5 is rebuilt with its chosen settings, fit on train+val, and scored "
           "on the same test partial carts (bookings made Apr-Sep 2026, the summer peak). Items already "
           "in the cart are never recommended. **cat_hr@k** counts a hit when a top-k item is in the "
           "same category as a hidden one (for example, the 25 in wardrobe box when the customer "
           "bought the 20 in).", "",
           "**Ceilings (diagnostics, not models).** `ceiling_true_segment` ranks by an approximation of "
           "the generator's own probabilities, including the hidden segment that no model sees. "
           "`ceiling_profile_only` averages the hidden segment out with the generator's segment "
           "probabilities, so it uses only what a model could know. Both are approximate (see "
           "`movrec/oracle.py`), so they are reference points, not exact bounds. **Headroom captured** = "
           "(model - segment popularity) / (profile-only ceiling - segment popularity), shown only for "
           "models above segment popularity.", "",
           "**The profile-only ceiling is not a strict upper bound.** It ignores what a revealed cart "
           "says about the customer beyond its co-occurrence lifts (a fuller cart hints at a bigger "
           "spender, through the generator's shared per-booking noise), and it ignores the random "
           "add/drop noise. Models that read the cart (ALS) can pick some of that up, so a model can "
           "match or beat it in places. Read it as 'roughly what the observable profile is worth', "
           "and the gap to the true-segment ceiling as 'roughly what the hidden segment is worth'.", ""]
    for ts, r in results["target_sets"].items():
        label = "all 45 services" if ts == "all" else f"add-ons only, truck excluded ({r['n_items']} services)"
        main = pd.DataFrame(r["main"])
        main["headroom captured"] = [f"{h:.0%}" if v > 0 else "-" for h, v in zip(main["headroom captured"], main["vs segment pop"])]
        out += [f"## Targets: {label}", "", "### All models (test)", "", markdown_table(main, "{:.3f}"), ""]

        cart = pd.DataFrame(r["by_cart_size"]).T
        cart.columns = [f"cart={c}" for c in cart.columns]
        out += [f"### {map_col} by partial-cart size", "", markdown_table(cart.reset_index(names="model"), "{:.3f}"), ""]

        move = pd.DataFrame(r["by_move_type"]).T
        move.columns = [f"{c} (n={r['move_type_counts'].get(c, 0)})" for c in move.columns]
        out += [f"### {map_col} by move type", "", markdown_table(move.reset_index(names="model"), "{:.3f}"), ""]

        out += ["### Robustness to the reveal seed", "",
                "Same fitted models, different random partial carts.", "",
                markdown_table(pd.DataFrame(r["robustness"]), "{:.3f}"), ""]

        out += ["### Ablations (recap)", "",
                "Feature groups for content kNN (validation, equal weights; from step 4):", "",
                markdown_table(pd.DataFrame(r["ablation_features_val"]), "{:.4f}"), "",
                "Hybrid components, dropped one at a time and re-tuned on val (test; from step 5):", "",
                markdown_table(pd.DataFrame(r["ablation_components_test"]), "{:+.4f}"), ""]
    return "\n".join(out)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Run step 6 (evaluation).")
    parser.add_argument("--data", type=Path, default=DATA_DIR)
    args = parser.parse_args(argv)
    results = run(args.data)
    (PROJECT_ROOT / "results" / "evaluation.json").write_text(json.dumps(results, indent=2, default=float) + "\n")
    report = PROJECT_ROOT / "reports" / "evaluation.md"
    report.write_text(render(results, load_experiment_config()))
    print(f"wrote {report}")


if __name__ == "__main__":
    main()
