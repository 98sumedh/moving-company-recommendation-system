"""Step 2: time split + baselines. Tunes on val, refits on train+val, scores test once.

Writes reports/baselines.md and results/baselines.json.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from movrec import DATA_DIR, PROJECT_ROOT
from movrec.baselines import CoOccurrence, GlobalPopularity, SegmentPopularity
from movrec.data import load_experiment_config, load_modeling_data
from movrec.evaluate import bootstrap_ci, make_cases, per_case_metrics, summarize
from movrec.report_utils import markdown_table
from movrec.split import describe, time_split

BANNER = ("> **SYNTHETIC DATA.** Booking records are generated (see README). These numbers validate "
          "the pipeline and compare methods on the same data; they are not real-world performance.")


def target_columns(data, target_set: str) -> np.ndarray:
    if target_set == "all":
        return np.ones(len(data.items), dtype=bool)
    if target_set == "no_truck":
        return ~data.truck_mask()
    raise ValueError(target_set)


def evaluate_model(model, fit_rows, eval_rows, bookings, matrix, cases, cfg):
    model.fit(bookings[fit_rows].reset_index(drop=True), matrix[fit_rows])
    eval_bookings = bookings[eval_rows].reset_index(drop=True).iloc[cases.rows].reset_index(drop=True)
    scores = model.score(eval_bookings, cases.revealed)
    ev = cfg["evaluation"]
    return per_case_metrics(scores, cases, ks=ev["ks"], map_k=ev["map_k"])


def run(data_dir: Path = DATA_DIR, verify: bool = True) -> dict:
    cfg = load_experiment_config()
    data = load_modeling_data(data_dir, verify=verify)
    split = time_split(data.bookings, **cfg["split"])
    ev, bl = cfg["evaluation"], cfg["baselines"]
    keys = bl["segment_keys"]
    map_col = f"map@{ev['map_k']}"
    train_val = split.train | split.val
    results = {"split": describe(data.bookings, split).to_dict("records"), "target_sets": {}}

    for target_set in ev["target_sets"]:
        cols = target_columns(data, target_set)
        matrix = data.matrix[:, cols]
        val_cases = make_cases(matrix[split.val], np.random.default_rng(ev["seed"]))
        test_cases = make_cases(matrix[split.test], np.random.default_rng(ev["seed"] + 1))
        args = (data.bookings, matrix)

        # --- tune on val (fit on train only)
        tuning = []
        for alpha in bl["segment_alpha_grid"]:
            m = evaluate_model(SegmentPopularity(keys, alpha), split.train, split.val, *args, val_cases, cfg)
            tuning.append({"model": "segment_popularity", "param": f"alpha={alpha}", map_col: m[map_col].mean()})
        best_alpha = max(bl["segment_alpha_grid"], key=lambda a: next(
            t[map_col] for t in tuning if t["param"] == f"alpha={a}"))
        for w in bl["cooc_prior_weight_grid"]:
            m = evaluate_model(CoOccurrence(keys, best_alpha, w), split.train, split.val, *args, val_cases, cfg)
            tuning.append({"model": "cooccurrence", "param": f"prior_weight={w}", map_col: m[map_col].mean()})
        best_w = max(bl["cooc_prior_weight_grid"], key=lambda w: next(
            t[map_col] for t in tuning if t["param"] == f"prior_weight={w}"))

        # --- final: refit on train+val, score test once
        models = [GlobalPopularity(), SegmentPopularity(keys, best_alpha), CoOccurrence(keys, best_alpha, best_w)]
        boot_rng = np.random.default_rng(ev["seed"] + 2)
        test = {}
        for model in models:
            per_case = evaluate_model(model, train_val, split.test, *args, test_cases, cfg)
            table = summarize(per_case, ev["cart_size_slices"])
            ci = {m: bootstrap_ci(per_case[m].to_numpy(), boot_rng) for m in (map_col, "hr@5")}
            test[model.name] = {"summary": table.reset_index(names="cart_size").to_dict("records"), "ci95": ci}
        results["target_sets"][target_set] = {
            "n_items": int(cols.sum()),
            "tuning": tuning,
            "chosen": {"segment_alpha": best_alpha, "cooc_prior_weight": best_w},
            "test": test,
        }
    return results


def render(results: dict, cfg: dict) -> str:
    ev = cfg["evaluation"]
    map_col = f"map@{ev['map_k']}"
    out = ["# Step 2: time split and baselines", "", BANNER, "",
           "## Split (by booking time)", "", markdown_table(pd.DataFrame(results["split"])), "",
           "Tuning uses val only (models fit on train). Final models are refit on train+val and "
           "scored once on test. Test is the summer 2026 peak; val is winter/spring, so tuning and "
           "testing happen in different seasons on purpose.", "",
           "**Protocol:** one case per booking with at least one target item. A random subset of "
           "size k ~ Uniform{0..n-1} is revealed as the partial cart; the rest is hidden. Items "
           "already in the cart are never recommended. Segment = move_type x home_size (observable, "
           "not the hidden segment). The co-occurrence prior-weight grid was widened once after the "
           "first run chose its top edge; the choice is still made on val only.", ""]
    for ts, r in results["target_sets"].items():
        label = "all 45 services" if ts == "all" else f"add-ons only, truck excluded ({r['n_items']} services)"
        out += [f"## Targets: {label}", "", "### Validation tuning", "",
                markdown_table(pd.DataFrame(r["tuning"]), "{:.4f}"), "",
                f"Chosen: segment alpha = {r['chosen']['segment_alpha']}, "
                f"co-occurrence prior weight = {r['chosen']['cooc_prior_weight']}.", "",
                "### Test (all cases)", ""]
        rows = []
        for name, t in r["test"].items():
            overall = next(s for s in t["summary"] if s["cart_size"] == "all")
            row = {"model": name, "cases": overall["cases"]}
            row.update({k: v for k, v in overall.items() if k not in ("cart_size", "cases")})
            lo, hi = t["ci95"][map_col]
            row[f"{map_col} 95% CI"] = f"{lo:.3f}-{hi:.3f}"
            rows.append(row)
        out += [markdown_table(pd.DataFrame(rows), "{:.3f}"), "", f"### Test by partial-cart size ({map_col} / hr@5)", ""]
        slices = [s["cart_size"] for s in next(iter(r["test"].values()))["summary"] if s["cart_size"] != "all"]
        sl_rows = []
        for name, t in r["test"].items():
            by = {s["cart_size"]: s for s in t["summary"]}
            row = {"model": name}
            for s in slices:
                row[f"cart={s} (n={by[s]['cases']})"] = f"{by[s][map_col]:.3f} / {by[s]['hr@5']:.3f}"
            sl_rows.append(row)
        out += [markdown_table(pd.DataFrame(sl_rows)), ""]
    return "\n".join(out)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Run step 2 baselines.")
    parser.add_argument("--data", type=Path, default=DATA_DIR)
    args = parser.parse_args(argv)
    results = run(args.data)
    (PROJECT_ROOT / "results").mkdir(exist_ok=True)
    (PROJECT_ROOT / "results" / "baselines.json").write_text(json.dumps(results, indent=2, default=float) + "\n")
    report = PROJECT_ROOT / "reports" / "baselines.md"
    report.write_text(render(results, load_experiment_config()))
    print(f"wrote {report}")


if __name__ == "__main__":
    main()
