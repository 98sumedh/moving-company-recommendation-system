"""Small helpers for writing markdown reports."""
import pandas as pd


def markdown_table(df: pd.DataFrame, floatfmt: str = "{:.2f}") -> str:
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].map(floatfmt.format)
    header = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "---|" * len(df.columns)
    rows = ["| " + " | ".join(map(str, r)) + " |" for r in df.itertuples(index=False)]
    return "\n".join([header, sep, *rows])


def test_tables(test: dict, map_col: str) -> list[str]:
    """Overall test table (with paired difference vs segment popularity) and the
    by-cart-size table. `test` maps model name -> {"summary", "diff_vs_segment_popularity", ["seed_map"]}."""
    rows = []
    for name, t in test.items():
        o = next(s for s in t["summary"] if s["cart_size"] == "all")
        d, lo, hi = t["diff_vs_segment_popularity"]
        rows.append({
            "model": name, "hr@3": o["hr@3"], "hr@5": o["hr@5"], "recall@5": o["recall@5"], map_col: o[map_col],
            f"{map_col} vs segment pop (95% CI)": "-" if name == "segment_popularity" else f"{d:+.3f} ({lo:+.3f} to {hi:+.3f})",
            "seed spread": "-" if "seed_map" not in t else f"{min(t['seed_map']):.3f}-{max(t['seed_map']):.3f}",
        })
    slices = [s["cart_size"] for s in next(iter(test.values()))["summary"] if s["cart_size"] != "all"]
    sliced = []
    for name, t in test.items():
        by = {s["cart_size"]: s for s in t["summary"]}
        sliced.append({"model": name, **{f"cart={s} (n={by[s]['cases']})": f"{by[s][map_col]:.3f} / {by[s]['hr@5']:.3f}"
                                         for s in slices}})
    return [markdown_table(pd.DataFrame(rows), "{:.3f}"), "",
            f"### Test by partial-cart size ({map_col} / hr@5)", "", markdown_table(pd.DataFrame(sliced)), ""]
