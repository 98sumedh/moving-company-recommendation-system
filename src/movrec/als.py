"""Implicit-feedback ALS (Hu, Koren & Volinsky, 2008) in plain NumPy, plus fold-in.

Rows are bookings, columns are services. Preference p_ui = 1 if the service is in the
cart, else 0; confidence c_ui = 1 + alpha * p_ui. ALS alternates two ridge regressions:

    booking step:  x_u = (Y'Y + alpha * sum_{i in u} y_i y_i' + reg*I)^-1 (1 + alpha) sum_{i in u} y_i
    service step:  y_i = (X'X + alpha * sum_{u has i} x_u x_u' + reg*I)^-1 (1 + alpha) sum_{u has i} x_u

Most bookings are one-off customers, so at prediction time there is no trained booking
vector. `fold_in` solves the booking step for a new partial cart against the trained
service factors: a small weighted least-squares problem per booking.
"""
import numpy as np


def solve_rows(fixed: np.ndarray, matrix: np.ndarray, reg: float, alpha: float, chunk: int = 4096) -> np.ndarray:
    """For every row of `matrix` (rows x cols, bool), solve the weighted ridge problem
    against the fixed column factors `fixed` (cols x f). Batched over rows with np.linalg.solve."""
    f = fixed.shape[1]
    gram = fixed.T @ fixed + reg * np.eye(f)                     # Y'Y + reg*I, shared by all rows
    m = matrix.astype(float)
    out = np.empty((len(m), f))
    for start in range(0, len(m), chunk):
        block = m[start:start + chunk]
        # alpha * sum_{i in row} y_i y_i'   -> (rows, f, f)
        a = gram + alpha * np.einsum("ri,if,ig->rfg", block, fixed, fixed, optimize=True)
        b = (1.0 + alpha) * block @ fixed                        # (rows, f)
        out[start:start + chunk] = np.linalg.solve(a, b[..., None])[..., 0]
    return out


def fold_in(item_factors: np.ndarray, carts: np.ndarray, reg: float, alpha: float) -> np.ndarray:
    """Booking vectors for new (partial) carts. An empty cart gives the zero vector."""
    return solve_rows(item_factors, carts, reg, alpha)


class ImplicitALS:
    def __init__(self, factors: int = 16, reg: float = 1.0, alpha: float = 20.0,
                 iterations: int = 15, seed: int = 0):
        self.factors, self.reg, self.alpha = factors, reg, alpha
        self.iterations, self.seed = iterations, seed

    def fit(self, matrix: np.ndarray):
        rng = np.random.default_rng(self.seed)
        self.item_factors = rng.normal(0, 0.1, size=(matrix.shape[1], self.factors))
        for _ in range(self.iterations):
            user = solve_rows(self.item_factors, matrix, self.reg, self.alpha)
            self.item_factors = solve_rows(user, matrix.T, self.reg, self.alpha)
        return self

    def score(self, carts: np.ndarray) -> np.ndarray:
        """Predicted preference for every service, given each booking's partial cart."""
        return fold_in(self.item_factors, carts, self.reg, self.alpha) @ self.item_factors.T


class ALSRecommender:
    """ALS scores from the partial cart, plus prior_weight x segment popularity.

    prior_weight = 0 gives "pure" ALS. An empty cart folds in to the zero vector, so pure
    ALS falls back to global popularity there (it has nothing else to go on).
    """

    def __init__(self, als: ImplicitALS, prior=None, prior_weight: float = 0.0, fallback=None):
        self.als, self.prior, self.prior_weight, self.fallback = als, prior, prior_weight, fallback
        self.name = "als" if prior_weight == 0 else "als_plus_segment_prior"

    def fit(self, bookings, matrix):
        self.als.fit(matrix)
        for part in (self.prior, self.fallback):
            if part is not None:
                part.fit(bookings, matrix)
        return self

    def score(self, bookings, revealed):
        s = self.als.score(revealed)
        if self.prior is not None and self.prior_weight:
            s = s + self.prior_weight * self.prior.score(bookings, revealed)
        if self.fallback is not None:
            empty = revealed.sum(axis=1) == 0
            s[empty] = self.fallback.score(bookings[empty], revealed[empty])
        return s
