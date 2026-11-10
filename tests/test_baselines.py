import numpy as np
import pandas as pd

from movrec.baselines import CoOccurrence, GlobalPopularity, SegmentPopularity


def toy():
    bookings = pd.DataFrame({"move_type": ["a", "a", "a", "b", "b"], "home_size": ["x"] * 5})
    #                 item0  item1  item2
    m = np.array([[1, 1, 0],
                  [1, 1, 0],
                  [1, 0, 0],
                  [0, 0, 1],
                  [0, 1, 1]], dtype=bool)
    return bookings, m


def test_global_popularity():
    b, m = toy()
    s = GlobalPopularity().fit(b, m).score(b.head(1), np.zeros((1, 3), bool))
    assert np.allclose(s[0], [3 / 5, 3 / 5, 2 / 5])


def test_segment_popularity_alpha_zero_and_unseen():
    b, m = toy()
    model = SegmentPopularity(["move_type", "home_size"], alpha=0).fit(b, m)
    s = model.score(pd.DataFrame({"move_type": ["b", "zzz"], "home_size": ["x", "x"]}), np.zeros((2, 3), bool))
    assert np.allclose(s[0], [0, 0.5, 1.0])
    assert np.allclose(s[1], [3 / 5, 3 / 5, 2 / 5])   # unseen segment -> global


def test_segment_popularity_shrinks_to_global():
    b, m = toy()
    big = SegmentPopularity(["move_type", "home_size"], alpha=1e9).fit(b, m)
    s = big.score(b.tail(1), np.zeros((1, 3), bool))
    assert np.allclose(s[0], [3 / 5, 3 / 5, 2 / 5])


def test_cooccurrence_follows_the_cart():
    b, m = toy()
    model = CoOccurrence(["move_type", "home_size"], alpha=0, prior_weight=0.0).fit(b, m)
    cart = np.array([[False, False, True]])            # item2 in cart
    s = model.score(b.tail(1), cart)[0]
    assert s[1] > s[0]                                 # item1 co-occurs with item2, item0 never does
    assert np.isclose(s[1], 0.5)                       # P(item1 | item2) = 1/2
