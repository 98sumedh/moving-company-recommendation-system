"""Weighted hybrid of the earlier models.

    score(j) = sum_m w_m(cart size) * score_m(j)

Components score on similar scales (popularity-like probabilities, ALS preferences in
roughly [0, 1]), so a linear blend works without normalising. Weights can differ by
partial-cart size: with an empty cart only the profile models know anything; as the cart
grows, cart-based models (ALS, co-occurrence) may deserve more weight. Weights are chosen
on validation by grid search over precomputed component scores.
"""
import itertools

import numpy as np

from movrec.evaluate import Cases, cart_size_label, per_case_metrics


class Hybrid:
    name = "hybrid"

    def __init__(self, components: dict, weights: dict, slices=(0, 1, 2, 3)):
        """components: name -> model with fit/score. weights: cart-size label -> {name: weight}.
        A single label "all" applies the same weights to every cart size."""
        self.components, self.weights, self.slices = components, weights, slices

    def fit(self, bookings, matrix):
        for model in self.components.values():
            model.fit(bookings, matrix)
        return self

    def score(self, bookings, revealed):
        parts = {name: m.score(bookings, revealed) for name, m in self.components.items()}
        return blend(parts, self.weights, revealed.sum(axis=1), self.slices)


def blend(parts: dict, weights: dict, cart_size: np.ndarray, slices) -> np.ndarray:
    labels = cart_size_label(cart_size, slices) if "all" not in weights else np.full(len(cart_size), "all")
    out = np.zeros_like(next(iter(parts.values())), dtype=float)
    for label, w in weights.items():
        rows = labels == label
        for name, weight in w.items():
            if weight:
                out[rows] += weight * parts[name][rows]
    return out


def weight_grid(grid: dict) -> list[dict]:
    """grid: component -> list of candidate weights. Returns every combination."""
    names = list(grid)
    return [dict(zip(names, combo)) for combo in itertools.product(*(grid[n] for n in names))]


def tune_weights(parts: dict, cases, grid: dict, map_k: int, rows: np.ndarray | None = None) -> tuple[dict, float]:
    """Best weight combination (by MAP@k) over the given case rows."""
    rows = np.arange(len(cases.rows)) if rows is None else rows
    sub = Cases(cases.rows[rows], cases.revealed[rows], cases.hidden[rows])
    sub_parts = {n: p[rows] for n, p in parts.items()}
    best, best_score = None, -np.inf
    for w in weight_grid(grid):
        s = sum(weight * sub_parts[n] for n, weight in w.items() if weight)
        if np.isscalar(s):  # all-zero weights
            continue
        score = per_case_metrics(s, sub, ks=(1,), map_k=map_k)[f"map@{map_k}"].mean()
        if score > best_score:
            best, best_score = w, score
    return best, float(best_score)
