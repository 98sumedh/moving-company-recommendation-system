"""The ceiling is a diagnostic; these tests check it is a sensible one."""
import numpy as np
import pytest

from movrec.baselines import SegmentPopularity
from movrec.data import cart_matrix
from movrec.evaluate import make_cases, per_case_metrics
from movrec.generator.profiles import booking_features
from movrec.oracle import oracle_scores, segment_probabilities


@pytest.fixture(scope="module")
def setup(small_data, catalog):
    b = small_data.bookings.reset_index(drop=True)
    items = catalog.ids
    m = cart_matrix(b, small_data.booking_services, items)
    seg = small_data.segments.set_index("booking_id").loc[b["booking_id"], "segment"].to_numpy()
    return b, m, items, seg


def test_segment_probabilities_are_distributions(setup, A):
    b, _, _, _ = setup
    names, p = segment_probabilities(A, b, booking_features(b, A))
    assert np.allclose(p.sum(axis=1), 1) and (p >= 0).all()
    commercial = (b["move_type"] == "commercial_office").to_numpy()
    assert np.allclose(p[commercial, names.index("commercial_office")], 0.85)


def test_ceilings_beat_segment_popularity_and_true_segment_is_highest(setup, A, catalog):
    b, m, items, seg = setup
    half = len(b) // 2
    cases = make_cases(m[half:], np.random.default_rng(0))
    test_b = b.iloc[half:].reset_index(drop=True).iloc[cases.rows].reset_index(drop=True)
    pop = SegmentPopularity(["move_type", "home_size"], 5).fit(b.iloc[:half], m[:half]).score(test_b, cases.revealed)
    prof = oracle_scores(A, catalog, test_b, cases.revealed, items)
    true = oracle_scores(A, catalog, test_b, cases.revealed, items, segment=seg[half:][cases.rows])
    score = {name: per_case_metrics(s, cases)["map@5"].mean() for name, s in
             (("pop", pop), ("prof", prof), ("true", true))}
    assert score["prof"] > score["pop"]
    assert score["true"] > score["prof"]


def test_revealed_truck_rules_out_other_trucks(setup, A, catalog):
    b, _, items, _ = setup
    row = b[b["move_type"] == "local_residential"].head(1)
    revealed = np.zeros((1, len(items)), bool)
    revealed[0, items.index("truck_16ft")] = True
    s = oracle_scores(A, catalog, row, revealed, items)[0]
    assert s[items.index("truck_10ft")] == 0 and s[items.index("truck_26ft")] == 0


def test_works_on_a_subset_of_items(setup, A, catalog):
    b, _, items, _ = setup
    subset = [s for s in items if not s.startswith("truck_")]
    s = oracle_scores(A, catalog, b.head(5), np.zeros((5, len(subset)), bool), subset)
    assert s.shape == (5, len(subset)) and np.isfinite(s).all()
