"""Offline evaluation on simulated partial carts.

For each booking with at least one target item, a random subset of its cart is revealed
(the partial cart) and the rest is hidden. A model scores every item; items already in
the partial cart are excluded; the hidden items are the relevant ones.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Cases:
    rows: np.ndarray       # row index into the bookings being evaluated
    revealed: np.ndarray   # cases x items, bool (the partial cart)
    hidden: np.ndarray     # cases x items, bool (what we want to recommend)

    @property
    def cart_size(self) -> np.ndarray:
        return self.revealed.sum(axis=1)


def make_cases(matrix: np.ndarray, rng: np.random.Generator) -> Cases:
    """Reveal k ~ Uniform{0..n-1} items per cart, so at least one item is always hidden."""
    rows = np.flatnonzero(matrix.sum(axis=1) > 0)
    revealed = np.zeros((len(rows), matrix.shape[1]), dtype=bool)
    for c, r in enumerate(rows):
        items = np.flatnonzero(matrix[r])
        k = rng.integers(0, len(items))
        revealed[c, rng.permutation(items)[:k]] = True
    hidden = matrix[rows] & ~revealed
    return Cases(rows, revealed, hidden)


def top_k(scores: np.ndarray, revealed: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k best items per row, never recommending what is already in the cart.
    Ties break by catalog order (stable sort), so results are deterministic."""
    s = np.where(revealed, -np.inf, scores.astype(float))
    return np.argsort(-s, axis=1, kind="stable")[:, :k]


def per_case_metrics(scores: np.ndarray, cases: Cases, ks=(3, 5), map_k: int = 5,
                     categories: np.ndarray | None = None) -> pd.DataFrame:
    """Per-case HR@k, Recall@k, MAP@map_k. With `categories` (one label per item), also
    category HR@k: a hit if a top-k item shares a category with any hidden item
    (e.g. the 25 in wardrobe box when the customer bought the 20 in)."""
    max_k = max(max(ks), map_k)
    top = top_k(scores, cases.revealed, max_k)
    rel = np.take_along_axis(cases.hidden, top, axis=1)          # cases x max_k, bool
    n_hidden = cases.hidden.sum(axis=1)
    out = {}
    for k in ks:
        hits = rel[:, :k].sum(axis=1)
        out[f"hr@{k}"] = (hits > 0).astype(float)
        out[f"recall@{k}"] = hits / n_hidden
    # AP@k = (1 / min(|hidden|, k)) * sum_i precision@i * rel_i
    r = rel[:, :map_k]
    precision_at_i = r.cumsum(axis=1) / np.arange(1, map_k + 1)
    out[f"map@{map_k}"] = (precision_at_i * r).sum(axis=1) / np.minimum(n_hidden, map_k)
    if categories is not None:
        codes, cat = np.unique(np.asarray(categories), return_inverse=True)
        hidden_cat = np.zeros((len(top), len(codes)), dtype=bool)
        rows, items = np.nonzero(cases.hidden)
        hidden_cat[rows, cat[items]] = True
        for k in ks:
            top_cat = cat[top[:, :k]]
            out[f"cat_hr@{k}"] = np.take_along_axis(hidden_cat, top_cat, axis=1).any(axis=1).astype(float)
    df = pd.DataFrame(out)
    df["cart_size"] = cases.cart_size
    return df


def cart_size_label(size: np.ndarray, slices) -> np.ndarray:
    top = max(slices)
    return np.array([f"{s}" if s < top else f"{top}+" for s in np.minimum(size, top)])


def summarize(per_case: pd.DataFrame, slices=(0, 1, 2, 3)) -> pd.DataFrame:
    metrics = [c for c in per_case.columns if c != "cart_size"]
    overall = per_case[metrics].mean().to_frame("all").T
    by = per_case.assign(slice=cart_size_label(per_case["cart_size"].to_numpy(), slices))
    sliced = by.groupby("slice")[metrics].mean()
    counts = pd.concat([pd.Series({"all": len(per_case)}), by.groupby("slice").size()])
    table = pd.concat([overall, sliced])
    table.insert(0, "cases", counts.reindex(table.index).astype(int))
    return table


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator, n_boot: int = 500, level: float = 0.95):
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    lo, hi = np.quantile(means, [(1 - level) / 2, (1 + level) / 2])
    return float(lo), float(hi)


def paired_diff_ci(a: np.ndarray, b: np.ndarray, rng, n_boot: int = 500) -> tuple[float, float, float]:
    """Mean of a - b over the same cases, with a 95% paired-bootstrap interval."""
    d = a - b
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    lo, hi = np.quantile(d[idx].mean(axis=1), [0.025, 0.975])
    return float(d.mean()), float(lo), float(hi)
