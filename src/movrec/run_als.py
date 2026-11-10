"""Step 3: implicit ALS with fold-in, compared with the step 2 baselines on the same cases.

Tunes on val (fit on train), refits on train+val, scores test once per init seed.
Writes reports/als.md and results/als.json.
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from movrec import DATA_DIR, PROJECT_ROOT
from movrec.als import ALSRecommender, ImplicitALS
from movrec.baselines import CoOccurrence, GlobalPopularity, SegmentPopularity
from movrec.data import load_experiment_config, load_modeling_data
from movrec.evaluate import make_cases, paired_diff_ci, per_case_metrics, summarize
from movrec.report_utils import markdown_table, test_tables
from movrec.run_baselines import BANNER, evaluate_model, target_columns
from movrec.split import time_split


def run(data_dir: Path = DATA_DIR, verify: bool = True) -> dict:
    cfg = load_experiment_config()
    ev, bl, ac = cfg["evaluation"], cfg["baselines"], cfg["als"]
    baseline_choice = json.loads((PROJECT_ROOT / "results" / "baselines.json").read_text())["target_sets"]
    data = load_modeling_data(data_dir, verify=verify)
    split = time_split(data.bookings, **cfg["split"])
    keys, map_col = bl["segment_keys"], f"map@{ev['map_k']}"
    train_val = split.train | split.val
    results = {"target_sets": {}}

    for target_set in ev["target_sets"]:
        matrix = data.matrix[:, target_columns(data, target_set)]
        val_cases = make_cases(matrix[split.val], np.random.default_rng(ev["seed"]))
        test_cases = make_cases(matrix[split.test], np.random.default_rng(ev["seed"] + 1))
        args = (data.bookings, matrix)
        chosen_bl = baseline_choice[target_set]["chosen"]

        def als(f, reg, alpha, seed=0):
            return ImplicitALS(factors=f, reg=reg, alpha=alpha, iterations=ac["iterations"], seed=seed)

        # --- tune on val, jointly: fit each ALS config once on train, then score every
        #     prior weight from that one fit (w = 0 is pure ALS with a popularity fallback
        #     for empty carts).
        val_b = data.bookings[split.val].reset_index(drop=True).iloc[val_cases.rows].reset_index(drop=True)
        train_b, train_m = data.bookings[split.train].reset_index(drop=True), matrix[split.train]
        prior_s = SegmentPopularity(keys, chosen_bl["segment_alpha"]).fit(train_b, train_m).score(val_b, val_cases.revealed)
        global_p = train_m.mean(axis=0)
        empty = val_cases.revealed.sum(axis=1) == 0
        tuning = []
        for f, reg, alpha in itertools.product(ac["factors_grid"], ac["reg_grid"], ac["alpha_grid"]):
            als_s = als(f, reg, alpha).fit(train_m).score(val_cases.revealed)
            for w in [0.0] + ac["prior_weight_grid"]:
                s = als_s + w * prior_s
                if w == 0:
                    s = s.copy()
                    s[empty] = global_p
                score = per_case_metrics(s, val_cases, ks=ev["ks"], map_k=ev["map_k"])[map_col].mean()
                tuning.append({"model": "als" if w == 0 else "als_plus_segment_prior",
                               "factors": f, "reg": reg, "alpha": alpha, "prior_weight": w, map_col: score})
        chosen = {name: max((t for t in tuning if t["model"] == name), key=lambda t: t[map_col])
                  for name in ("als", "als_plus_segment_prior")}

        def build(name, seed):
            c = chosen[name]
            model = als(c["factors"], c["reg"], c["alpha"], seed)
            if name == "als":
                return ALSRecommender(model, fallback=GlobalPopularity())
            return ALSRecommender(model, prior=SegmentPopularity(keys, chosen_bl["segment_alpha"]), prior_weight=c["prior_weight"])

        # --- test: baselines once, ALS variants once per seed
        per_case = {
            "segment_popularity": evaluate_model(SegmentPopularity(keys, chosen_bl["segment_alpha"]), train_val, split.test, *args, test_cases, cfg),
            "cooccurrence": evaluate_model(CoOccurrence(keys, chosen_bl["segment_alpha"], chosen_bl["cooc_prior_weight"]), train_val, split.test, *args, test_cases, cfg),
        }
        seed_scores = {"als": [], "als_plus_segment_prior": []}
        for seed in ac["test_seeds"]:
            for name in seed_scores:
                pc = evaluate_model(build(name, seed), train_val, split.test, *args, test_cases, cfg)
                seed_scores[name].append(pc[map_col].mean())
                if seed == ac["test_seeds"][0]:
                    per_case[name] = pc

        boot = np.random.default_rng(ev["seed"] + 3)
        ref = per_case["segment_popularity"][map_col].to_numpy()
        test = {}
        for name, pc in per_case.items():
            test[name] = {
                "summary": summarize(pc, ev["cart_size_slices"]).reset_index(names="cart_size").to_dict("records"),
                "diff_vs_segment_popularity": paired_diff_ci(pc[map_col].to_numpy(), ref, boot),
            }
            if name in seed_scores:
                test[name]["seed_map"] = seed_scores[name]
        results["target_sets"][target_set] = {
            "tuning": tuning,
            "chosen": {name: {k: c[k] for k in ("factors", "reg", "alpha", "prior_weight")} for name, c in chosen.items()},
            "test": test,
        }
    return results


def render(results: dict, cfg: dict) -> str:
    map_col = f"map@{cfg['evaluation']['map_k']}"
    out = ["# Step 3: implicit ALS with fold-in", "", BANNER, "",
           "ALS is trained on the booking x service matrix (train split). Test bookings are new "
           "carts, so each one is **folded in**: its partial cart is solved against the trained "
           "service factors with a weighted least-squares step (`movrec.als.fold_in`). Same split, "
           "same test cases and same tuning rules as step 2.", "",
           "- **als**: ALS scores only. An empty cart folds in to the zero vector, so it falls back "
           "to global popularity there.",
           "- **als_plus_segment_prior**: ALS scores + w x segment popularity (move_type x home_size), "
           "w tuned on val.", "",
           "Tuning is joint over factors, regularization, alpha and prior weight (each ALS config is "
           "fit once; every prior weight is scored from that fit). The grids were widened once, and "
           "the search made joint, after a first run picked grid edges with a two-stage search; "
           "choices are still made on val only.", ""]
    for ts, r in results["target_sets"].items():
        label = "all 45 services" if ts == "all" else "add-ons only, truck excluded"
        tun = pd.DataFrame(r["tuning"]).sort_values(map_col, ascending=False)
        top = pd.concat([tun[tun["model"] == m].head(5) for m in ("als_plus_segment_prior", "als")])
        out += [f"## Targets: {label}", "", "### Validation tuning (top 5 per model)", "",
                markdown_table(top, "{:.4f}"), ""]
        for name, c in r["chosen"].items():
            out.append(f"- Chosen for **{name}**: factors={c['factors']}, reg={c['reg']}, alpha={c['alpha']}, "
                       f"prior weight={c['prior_weight']}.")
        out += ["", "### Test", "", *test_tables(r["test"], map_col)]
    return "\n".join(out)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Run step 3 (implicit ALS).")
    parser.add_argument("--data", type=Path, default=DATA_DIR)
    args = parser.parse_args(argv)
    results = run(args.data)
    (PROJECT_ROOT / "results" / "als.json").write_text(json.dumps(results, indent=2, default=float) + "\n")
    report = PROJECT_ROOT / "reports" / "als.md"
    report.write_text(render(results, load_experiment_config()))
    print(f"wrote {report}")


if __name__ == "__main__":
    main()
