import numpy as np
import pandas as pd
import pytest

from movrec.content import FEATURE_GROUPS, ContentKNN, ProfileEncoder


def toy_bookings(n=40, seed=0):
    rng = np.random.default_rng(seed)
    big = np.arange(n) % 2 == 0
    return pd.DataFrame({
        "move_type": np.where(big, "local_residential", "single_item"),
        "home_size": np.where(big, "4br_plus", "single_item"),
        "sq_ft": np.where(big, 2500, 0) + rng.integers(0, 50, n),
        "est_weight_lbs": np.where(big, 10000, 250) + rng.integers(0, 50, n),
        "origin_floor": 1, "dest_floor": 1, "origin_stairs_flights": 0, "dest_stairs_flights": 0,
        "origin_elevator": False, "dest_elevator": False,
        "origin_region": "sd_central", "dest_region": "sd_east", "distance_miles": 10.0,
        "arrival_window": "morning", "move_date": pd.Timestamp("2025-06-01"), "is_weekend": False,
        "lead_time_days": 10, "estimate_method": "photos_video", "payment_method": "card",
        "crew": np.where(big, "crew_4", "crew_2"),
    }), big


def test_rows_are_unit_length_and_truck_is_not_a_feature():
    b, _ = toy_bookings()
    x = ProfileEncoder(list(FEATURE_GROUPS)).fit_transform(b)
    assert np.allclose(np.linalg.norm(x, axis=1), 1)
    assert not any("truck" in c for cats, nums in FEATURE_GROUPS.values() for c in cats + nums)


def test_unseen_category_encodes_without_error():
    b, _ = toy_bookings()
    enc = ProfileEncoder(["core"]).fit(b)
    new = b.head(1).assign(home_size="office_l")
    assert np.isfinite(enc.transform(new)).all()


def test_knn_recommends_what_similar_bookings_bought():
    b, big = toy_bookings()
    m = np.zeros((len(b), 3), bool)
    m[big, 0] = True        # big homes buy item 0
    m[~big, 2] = True       # single items buy item 2
    model = ContentKNN(list(FEATURE_GROUPS), k=5).fit(b, m)
    s = model.score(b.iloc[[0, 1]], np.zeros((2, 3), bool))
    assert s[0].argmax() == 0 and s[1].argmax() == 2
    assert np.isclose(s[0, 0], 1.0)


def test_chunking_does_not_change_scores():
    b, big = toy_bookings()
    m = np.random.default_rng(1).random((len(b), 4)) < 0.4
    a = ContentKNN(["core", "calendar"], k=7, chunk=3).fit(b, m).score(b, np.zeros((len(b), 4), bool))
    c = ContentKNN(["core", "calendar"], k=7, chunk=4096).fit(b, m).score(b, np.zeros((len(b), 4), bool))
    assert np.allclose(a, c)


def test_unknown_group_is_rejected():
    b, _ = toy_bookings()
    with pytest.raises(KeyError):
        ProfileEncoder(["not_a_group"]).fit(b)


def test_zero_weight_drops_a_group():
    b, _ = toy_bookings()
    m = np.zeros((len(b), 2), bool)
    zero = ContentKNN(["core", "calendar"], k=5, group_weights={"core": 1.0, "calendar": 0.0}).fit(b, m)
    only = ContentKNN(["core"], k=5).fit(b, m)
    assert np.allclose(zero.x, only.x)
