"""Baseline recommenders. Each has fit(bookings, matrix) and score(bookings, revealed) -> scores.

`bookings` is the profile table (observable columns only); `matrix` / `revealed` are
bookings x items boolean carts. Higher score = recommend first.
"""
import numpy as np
import pandas as pd


class GlobalPopularity:
    name = "global_popularity"

    def fit(self, bookings: pd.DataFrame, matrix: np.ndarray):
        self.p = matrix.mean(axis=0)
        return self

    def score(self, bookings: pd.DataFrame, revealed: np.ndarray) -> np.ndarray:
        return np.tile(self.p, (len(revealed), 1))


class SegmentPopularity:
    """Popularity within an observable segment (e.g. move_type x home_size), shrunk toward
    global popularity: p(j | s) = (count_sj + alpha * p_j) / (n_s + alpha)."""

    name = "segment_popularity"

    def __init__(self, keys: list[str], alpha: float = 20.0):
        self.keys = keys
        self.alpha = alpha

    def _key(self, bookings: pd.DataFrame) -> pd.Series:
        return bookings[self.keys].astype(str).agg("|".join, axis=1)

    def fit(self, bookings: pd.DataFrame, matrix: np.ndarray):
        self.p_global = matrix.mean(axis=0)
        key = self._key(bookings).to_numpy()
        self.table = {}
        for k in np.unique(key):
            rows = matrix[key == k]
            self.table[k] = (rows.sum(axis=0) + self.alpha * self.p_global) / (len(rows) + self.alpha)
        return self

    def score(self, bookings: pd.DataFrame, revealed: np.ndarray) -> np.ndarray:
        key = self._key(bookings)
        return np.vstack([self.table.get(k, self.p_global) for k in key])


class CoOccurrence:
    """score(j) = mean over cart items i of P(j | i), plus prior_weight x segment popularity.
    The prior breaks ties and is the whole score for an empty cart."""

    name = "cooccurrence"

    def __init__(self, keys: list[str], alpha: float = 20.0, prior_weight: float = 0.1):
        self.prior = SegmentPopularity(keys, alpha)
        self.prior_weight = prior_weight

    def fit(self, bookings: pd.DataFrame, matrix: np.ndarray):
        m = matrix.astype(float)
        counts = m.T @ m                                   # items x items co-counts
        item_counts = np.diag(counts).copy()
        with np.errstate(divide="ignore", invalid="ignore"):
            self.cond = np.where(item_counts[:, None] > 0, counts / item_counts[:, None], 0.0)  # P(j | i)
        np.fill_diagonal(self.cond, 0.0)
        self.prior.fit(bookings, matrix)
        return self

    def score(self, bookings: pd.DataFrame, revealed: np.ndarray) -> np.ndarray:
        r = revealed.astype(float)
        cooc = (r @ self.cond) / np.maximum(r.sum(axis=1, keepdims=True), 1.0)
        return cooc + self.prior_weight * self.prior.score(bookings, revealed)
