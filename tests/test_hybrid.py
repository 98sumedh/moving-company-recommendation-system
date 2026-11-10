import numpy as np

from movrec.evaluate import Cases
from movrec.hybrid import Hybrid, blend, tune_weights, weight_grid


def test_weight_grid_is_full_product():
    g = weight_grid({"a": [0, 1], "b": [0, 0.5, 1]})
    assert len(g) == 6 and {"a": 1, "b": 0.5} in g


def test_blend_uses_weights_per_cart_size():
    parts = {"x": np.ones((3, 2)), "y": np.full((3, 2), 2.0)}
    weights = {"0": {"x": 1, "y": 0}, "1": {"x": 0, "y": 1}, "2+": {"x": 1, "y": 1}}
    out = blend(parts, weights, np.array([0, 1, 5]), slices=(0, 1, 2))
    assert np.allclose(out[:, 0], [1, 2, 3])


def test_blend_all_label_applies_everywhere():
    parts = {"x": np.ones((2, 2))}
    assert np.allclose(blend(parts, {"all": {"x": 3}}, np.array([0, 4]), slices=(0, 1)), 3)


def test_tune_picks_the_informative_component():
    # hidden item is always item 0; component "good" ranks it first, "bad" ranks it last
    n = 50
    hidden = np.zeros((n, 3), bool); hidden[:, 0] = True
    cases = Cases(np.arange(n), np.zeros((n, 3), bool), hidden)
    parts = {"good": np.tile([1.0, 0.5, 0.0], (n, 1)), "bad": np.tile([0.0, 0.5, 1.0], (n, 1))}
    w, score = tune_weights(parts, cases, {"good": [0, 1], "bad": [0, 1]}, map_k=3)
    assert w == {"good": 1, "bad": 0} and score == 1.0


class Const:
    def __init__(self, v): self.v = np.asarray(v, float)
    def fit(self, b, m): self.fitted = True; return self
    def score(self, b, r): return np.tile(self.v, (len(r), 1))


def test_hybrid_fits_components_and_scores():
    h = Hybrid({"a": Const([1, 0]), "b": Const([0, 1])}, {"all": {"a": 1, "b": 2}}).fit(None, None)
    assert h.components["a"].fitted
    assert np.allclose(h.score(None, np.zeros((2, 2), bool)), [[1, 2], [1, 2]])
