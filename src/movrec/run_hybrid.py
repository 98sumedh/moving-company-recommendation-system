"""Step 5: weighted hybrid, one weight set for all carts vs. weights by partial-cart size.

Component settings come from steps 2-4 (results/*.json). Component scores are computed
once per split (fit on train for val, on train+val for test); blend weights are tuned on
val by grid search. Includes a drop-one-component ablation. Writes reports/hybrid.md and
results/hybrid.json.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from movrec import DATA_DIR, PROJECT_ROOT
from movrec.als import ImplicitALS
from movrec.baselines import CoOccurrence, SegmentPopularity
from movrec.content import ContentKNN
from movrec.data import load_experiment_config, load_modeling_data
from movrec.evaluate import cart_size_label, make_cases, paired_diff_ci, per_case_metrics, summarize
from movrec.hybrid import blend, tune_weights
from movrec.report_utils import markdown_table, test_tables
from movrec.run_baselines import BANNER, target_columns
from movrec.split import time_split


class ALSScores:
    """Raw ALS fold-in scores as a component (no prior, no fallback)."""

    def __init__(self, als):
        self.als = als

    def fit(self, bookings, matrix):
        self.als.fit(matrix)
        return self

    def score(self, bookings, revealed):
        return self.als.score(revealed)


def components(chosen: dict, keys, cfg) -> dict:
    a, c = chosen["als"], chosen["content"]
    return {
        "seg": SegmentPopularity(keys, chosen["segment_alpha"]),
        "knn": ContentKNN(cfg["content"]["groups"], c["k"], c["power"], group_weights=c["group_weights"]),
        "als": ALSScores(ImplicitALS(a["factors"], a["reg"], a["alpha"], cfg["als"]["iterations"], seed=0)),
        "cooc": CoOccurrence(keys, chosen["segment_alpha"], prior_weight=0.0),
    }


def component_scores(models: dict, fit_rows, eval_rows, bookings, matrix, cases) -> dict:
    fit_b = bookings[fit_rows].reset_index(drop=True)
    eval_b = bookings[eval_rows].reset_index(drop=True).iloc[cases.rows].reset_index(drop=True)
    return {name: m.fit(fit_b, matrix[fit_rows]).score(eval_b, cases.revealed) for name, m in models.items()}


def tune(parts, cases, grid, map_k, slices, by_cart_size: bool) -> dict:
    if not by_cart_size:
        return {"all": tune_weights(parts, cases, grid, map_k)[0]}
    labels = cart_size_label(cases.cart_size, slices)
    return {label: tune_weights(parts, cases, grid, map_k, rows=np.flatnonzero(labels == label))[0]
            for label in np.unique(labels)}


def edges(weights: dict, grid: dict) -> list[str]:
    """Components (other than the seg anchor) whose chosen weight is the top of the grid."""
    return sorted({f"{label}:{n}" for label, w in weights.items() for n, v in w.items()
                   if n != "seg" and v == max(grid[n]) and v > 0})


def run(data_dir: Path = DATA_DIR, verify: bool = True) -> dict:
    cfg = load_experiment_config()
    ev, hc = cfg["evaluation"], cfg["hybrid"]
    grid, slices, map_k = hc["weight_grid"], ev["cart_size_slices"], ev["map_k"]
    map_col = f"map@{map_k}"
    prior = {n: json.loads((PROJECT_ROOT / "results" / f"{n}.json").read_text())["target_sets"]
             for n in ("baselines", "als", "content")}
    data = load_modeling_data(data_dir, verify=verify)
    split = time_split(data.bookings, **cfg["split"])
    train_val = split.train | split.val
    results = {"target_sets": {}}

    for target_set in ev["target_sets"]:
        matrix = data.matrix[:, target_columns(data, target_set)]
        val_cases = make_cases(matrix[split.val], np.random.default_rng(ev["seed"]))
        test_cases = make_cases(matrix[split.test], np.random.default_rng(ev["seed"] + 1))
        chosen = {"segment_alpha": prior["baselines"][target_set]["chosen"]["segment_alpha"],
                  "als": prior["als"][target_set]["chosen"]["als_plus_segment_prior"],
                  "content": prior["content"][target_set]["chosen"]}
        keys = cfg["baselines"]["segment_keys"]

        val_parts = component_scores(components(chosen, keys, cfg), split.train, split.val, data.bookings, matrix, val_cases)
        test_parts = component_scores(components(chosen, keys, cfg), train_val, split.test, data.bookings, matrix, test_cases)

        def test_metrics(weights):
            s = blend(test_parts, weights, test_cases.cart_size, slices)
            return per_case_metrics(s, test_cases, ks=ev["ks"], map_k=map_k)

        # --- hybrids, tuned on val
        hybrids = {name: tune(val_parts, val_cases, grid, map_k, slices, by_size)
                   for name, by_size in (("hybrid_global", False), ("hybrid_by_cart_size", True))}

        # --- reference models, rebuilt from the same component scores
        als_w = chosen["als"]["prior_weight"]
        per_case = {
            "segment_popularity": test_metrics({"all": {"seg": 1}}),
            "als_plus_segment_prior": test_metrics({"all": {"als": 1, "seg": als_w}}),
            "content_knn": test_metrics({"all": {"knn": 1}}),
            **{name: test_metrics(w) for name, w in hybrids.items()},
        }

        # --- drop-one-component ablation of the by-cart-size hybrid (re-tuned on val without it)
        full = per_case["hybrid_by_cart_size"][map_col].to_numpy()
        boot = np.random.default_rng(ev["seed"] + 5)
        ablation = []
        for comp in grid:
            reduced = {**grid, comp: [0]}
            w = tune(val_parts, val_cases, reduced, map_k, slices, True)
            m = test_metrics(w)[map_col].to_numpy()
            d, lo, hi = paired_diff_ci(m, full, boot)
            ablation.append({"without": comp, map_col: m.mean(), "change vs full hybrid": d, "95% CI low": lo, "95% CI high": hi})

        ref = per_case["segment_popularity"][map_col].to_numpy()
        best_single = per_case["als_plus_segment_prior"][map_col].to_numpy()
        test = {name: {"summary": summarize(pc, slices).reset_index(names="cart_size").to_dict("records"),
                       "diff_vs_segment_popularity": paired_diff_ci(pc[map_col].to_numpy(), ref, boot)}
                for name, pc in per_case.items()}
        vs_best = {name: paired_diff_ci(per_case[name][map_col].to_numpy(), best_single, boot) for name in hybrids}
        results["target_sets"][target_set] = {
            "components": chosen, "weights": hybrids,
            "grid_edges": {name: edges(w, grid) for name, w in hybrids.items()},
            "test": test, "hybrid_vs_als_plus_prior": vs_best, "ablation": ablation,
        }
    return results


def render(results: dict, cfg: dict) -> str:
    map_col = f"map@{cfg['evaluation']['map_k']}"
    out = ["# Step 5: hybrid", "", BANNER, "",
           "score(j) = w_seg x segment popularity + w_knn x content kNN + w_als x ALS fold-in "
           "+ w_cooc x co-occurrence. Component settings come from steps 2-4; only the blend "
           "weights are tuned here, on val, by grid search "
           f"({', '.join(f'{k}: {v}' for k, v in cfg['hybrid']['weight_grid'].items())}).", "",
           "- **hybrid_global**: one weight set for every cart.",
           "- **hybrid_by_cart_size**: a separate weight set for each partial-cart size (0, 1, 2, 3+).", "",
           "Rankings depend only on weight ratios, so `seg` = 1 at its top value is not a search limit. "
           "The knn and als grids were widened once after the first run hit their top values. Remaining "
           "edge flags ignore that only ratios matter: knn = 8, als = 3, cooc = 0.3 with seg = 0 ranks "
           "the same as knn = 1, als = 0.375, cooc = 0.04, i.e. it asks for less co-occurrence, not "
           "more knn. The grids were not widened again, to avoid tuning to noise.", ""]
    for ts, r in results["target_sets"].items():
        label = "all 45 services" if ts == "all" else "add-ons only, truck excluded"
        wrows = [{"model": name, "cart size": size, **w} for name, ws in r["weights"].items() for size, w in ws.items()]
        out += [f"## Targets: {label}", "", "### Chosen weights (validation)", "", markdown_table(pd.DataFrame(wrows)), ""]
        for name, e in r["grid_edges"].items():
            if e:
                out.append(f"- {name}: weight at the top of its grid for {', '.join(e)} (search may be too narrow there).")
        out += ["", "### Test", "", *test_tables(r["test"], map_col)]
        for name, (d, lo, hi) in r["hybrid_vs_als_plus_prior"].items():
            out.append(f"- {name} vs als_plus_segment_prior (best single model so far): {map_col} {d:+.3f} (95% CI {lo:+.3f} to {hi:+.3f})")
        out += ["", "### Drop-one-component ablation (test; weights re-tuned on val without the component)", "",
                markdown_table(pd.DataFrame(r["ablation"]), "{:+.4f}"), ""]
    return "\n".join(out)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Run step 5 (hybrid).")
    parser.add_argument("--data", type=Path, default=DATA_DIR)
    args = parser.parse_args(argv)
    results = run(args.data)
    (PROJECT_ROOT / "results" / "hybrid.json").write_text(json.dumps(results, indent=2, default=float) + "\n")
    report = PROJECT_ROOT / "reports" / "hybrid.md"
    report.write_text(render(results, load_experiment_config()))
    print(f"wrote {report}")


if __name__ == "__main__":
    main()
