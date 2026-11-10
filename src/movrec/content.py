"""Content side: move-profile vectors and cosine nearest neighbours.

Every booking is described only by what is known when it is made (move type, size,
access, route, calendar, quote details). The truck is NOT a feature, because it is a
recommendation target. A new booking is matched to the k most similar past bookings,
and services are scored by the similarity-weighted share of those neighbours who bought them.
This works with an empty cart, which is where cart-based models have nothing to go on.
"""
import numpy as np
import pandas as pd

# feature group -> (categorical columns, numeric columns). Numeric columns are transformed below.
FEATURE_GROUPS = {
    "core": (["move_type", "home_size"], ["log_sq_ft", "log_weight"]),
    "access": ([], ["origin_floor", "dest_floor", "origin_stairs_flights", "dest_stairs_flights",
                    "origin_elevator", "dest_elevator"]),
    "route": (["origin_region", "dest_region"], ["log_distance"]),
    "calendar": (["arrival_window"], ["month_sin", "month_cos", "is_weekend", "log_lead_time"]),
    "quote": (["estimate_method", "payment_method", "crew"], []),
}


def derived_columns(bookings: pd.DataFrame) -> pd.DataFrame:
    month = bookings["move_date"].dt.month
    return bookings.assign(
        log_sq_ft=np.log1p(bookings["sq_ft"]),
        log_weight=np.log1p(bookings["est_weight_lbs"]),
        log_distance=np.log1p(bookings["distance_miles"]),
        log_lead_time=np.log1p(bookings["lead_time_days"]),
        month_sin=np.sin(2 * np.pi * month / 12),
        month_cos=np.cos(2 * np.pi * month / 12),
    )


class ProfileEncoder:
    """One-hot categoricals + standardized numerics, per feature group. Each group block is
    scaled to the same total weight (so a group with many columns doesn't dominate), then
    rows are L2-normalized so a dot product is a cosine similarity."""

    def __init__(self, groups: list[str], group_weights: dict | None = None):
        self.groups = groups
        self.group_weights = group_weights or {}

    def fit(self, bookings: pd.DataFrame):
        df = derived_columns(bookings)
        self.categories, self.stats = {}, {}
        for g in self.groups:
            cats, nums = FEATURE_GROUPS[g]
            for c in cats:
                self.categories[c] = sorted(df[c].astype(str).unique())
            for c in nums:
                x = df[c].astype(float)
                self.stats[c] = (x.mean(), x.std() or 1.0)
        return self

    def transform(self, bookings: pd.DataFrame) -> np.ndarray:
        df = derived_columns(bookings)
        blocks = []
        for g in self.groups:
            cats, nums = FEATURE_GROUPS[g]
            cols = []
            for c in cats:
                values = df[c].astype(str).to_numpy()
                cols.append(np.stack([values == v for v in self.categories[c]], axis=1).astype(np.float32))
            for c in nums:
                mean, std = self.stats[c]
                cols.append(((df[c].astype(float).to_numpy() - mean) / std)[:, None].astype(np.float32))
            block = np.hstack(cols)
            blocks.append(block * self.group_weights.get(g, 1.0) / np.sqrt(block.shape[1]))
        x = np.hstack(blocks)
        return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-9)

    def fit_transform(self, bookings: pd.DataFrame) -> np.ndarray:
        return self.fit(bookings).transform(bookings)


class ContentKNN:
    """score(j) = sum_n w_n * bought_nj / sum_n w_n over the k nearest past bookings,
    with w_n = max(cosine, 0) ** power."""

    name = "content_knn"

    def __init__(self, groups: list[str], k: int = 100, power: float = 1.0, chunk: int = 1024,
                 group_weights: dict | None = None):
        weights = group_weights or {}
        self.encoder = ProfileEncoder([g for g in groups if weights.get(g, 1.0) > 0], weights)
        self.k, self.power, self.chunk = k, power, chunk

    def fit(self, bookings: pd.DataFrame, matrix: np.ndarray):
        self.x = self.encoder.fit_transform(bookings)
        self.carts = matrix.astype(np.float32)
        self.p_global = matrix.mean(axis=0)
        return self

    def score(self, bookings: pd.DataFrame, revealed: np.ndarray) -> np.ndarray:
        q = self.encoder.transform(bookings)
        k = min(self.k, len(self.x))
        out = np.empty((len(q), self.carts.shape[1]))
        for start in range(0, len(q), self.chunk):
            sims = q[start:start + self.chunk] @ self.x.T                     # cosine similarities
            idx = np.argpartition(-sims, k - 1, axis=1)[:, :k]                # k nearest (unordered)
            w = np.maximum(np.take_along_axis(sims, idx, axis=1), 0) ** self.power
            num = np.einsum("qk,qkj->qj", w, self.carts[idx])
            den = w.sum(axis=1, keepdims=True)
            out[start:start + self.chunk] = np.where(den > 0, num / np.maximum(den, 1e-12), self.p_global)
        return out
