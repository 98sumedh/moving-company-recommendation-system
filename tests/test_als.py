import numpy as np
import pandas as pd

from movrec.als import ALSRecommender, ImplicitALS, fold_in, solve_rows
from movrec.baselines import GlobalPopularity


def test_fold_in_matches_weighted_least_squares():
    """fold_in must equal the textbook weighted ridge solution, solved with np.linalg.lstsq."""
    rng = np.random.default_rng(0)
    Y = rng.normal(size=(10, 4))
    cart = np.zeros((1, 10), bool); cart[0, [1, 4, 7]] = True
    reg, alpha = 0.5, 10.0
    x = fold_in(Y, cart, reg, alpha)[0]

    c = 1 + alpha * cart[0]
    p = cart[0].astype(float)
    A = np.vstack([np.sqrt(c)[:, None] * Y, np.sqrt(reg) * np.eye(4)])
    b = np.concatenate([np.sqrt(c) * p, np.zeros(4)])
    expected = np.linalg.lstsq(A, b, rcond=None)[0]
    assert np.allclose(x, expected)


def test_empty_cart_folds_to_zero():
    Y = np.random.default_rng(1).normal(size=(6, 3))
    assert np.allclose(fold_in(Y, np.zeros((2, 6), bool), 1.0, 5.0), 0)


def test_batching_does_not_change_results():
    rng = np.random.default_rng(2)
    Y = rng.normal(size=(8, 3))
    m = rng.random((50, 8)) < 0.3
    assert np.allclose(solve_rows(Y, m, 1.0, 5.0, chunk=7), solve_rows(Y, m, 1.0, 5.0, chunk=4096))


def two_cluster_data(n=400, seed=3):
    """Items 0-2 are bought together, items 3-5 are bought together."""
    rng = np.random.default_rng(seed)
    m = np.zeros((n, 6), bool)
    group = rng.random(n) < 0.5
    m[group, :3] = rng.random((group.sum(), 3)) < 0.8
    m[~group, 3:] = rng.random(((~group).sum(), 3)) < 0.8
    return m


def test_als_recommends_within_cluster():
    m = two_cluster_data()
    # few factors + real regularization; with factors close to n_items ALS just memorises each item
    als = ImplicitALS(factors=2, reg=10, alpha=10, iterations=10).fit(m)
    cart = np.zeros((2, 6), bool); cart[0, 0] = True; cart[1, 4] = True
    s = als.score(cart)
    assert s[0, [1, 2]].min() > s[0, 3:].max()
    assert s[1, [3, 5]].min() > s[1, :3].max()


def test_fit_is_deterministic_given_seed():
    m = two_cluster_data()
    a = ImplicitALS(factors=3, iterations=3, seed=5).fit(m).item_factors
    b = ImplicitALS(factors=3, iterations=3, seed=5).fit(m).item_factors
    assert np.allclose(a, b)


def test_pure_als_falls_back_to_popularity_for_empty_carts():
    m = two_cluster_data()
    bookings = pd.DataFrame({"move_type": ["x"] * len(m), "home_size": ["y"] * len(m)})
    rec = ALSRecommender(ImplicitALS(factors=3, iterations=3), fallback=GlobalPopularity()).fit(bookings, m)
    s = rec.score(bookings.head(1), np.zeros((1, 6), bool))
    assert np.allclose(s[0], m.mean(axis=0))
