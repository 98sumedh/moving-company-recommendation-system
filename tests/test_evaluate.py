import numpy as np

from movrec.evaluate import Cases, make_cases, per_case_metrics, summarize, top_k


def toy_case(hidden_items, revealed_items, n=6):
    h = np.zeros((1, n), bool); h[0, hidden_items] = True
    r = np.zeros((1, n), bool); r[0, revealed_items] = True
    return Cases(rows=np.array([0]), revealed=r, hidden=h)


def test_metrics_by_hand():
    cases = toy_case(hidden_items=[0, 2], revealed_items=[])
    scores = np.array([[0.3, 0.5, 0.9, 0.1, 0.0, 0.0]])   # ranking: 2, 1, 0, 3, ...
    m = per_case_metrics(scores, cases, ks=(1, 3), map_k=5).iloc[0]
    assert m["hr@1"] == 1 and m["recall@1"] == 0.5
    assert m["hr@3"] == 1 and m["recall@3"] == 1
    assert np.isclose(m["map@5"], (1 / 1 + 2 / 3) / 2)


def test_cart_items_are_never_recommended():
    cases = toy_case(hidden_items=[3], revealed_items=[0, 1])
    scores = np.array([[9.0, 8.0, 0.1, 0.2, 0.0, 0.0]])
    top = top_k(scores, cases.revealed, 3)[0]
    assert 0 not in top and 1 not in top
    assert top[0] == 3


def test_make_cases_partitions_each_cart():
    rng = np.random.default_rng(0)
    m = rng.random((200, 10)) < 0.3
    cases = make_cases(m, np.random.default_rng(1))
    assert (m[cases.rows].sum(axis=1) > 0).all()
    assert not (cases.revealed & cases.hidden).any()
    assert ((cases.revealed | cases.hidden) == m[cases.rows]).all()
    assert (cases.hidden.sum(axis=1) >= 1).all()
    again = make_cases(m, np.random.default_rng(1))
    assert (again.revealed == cases.revealed).all()


def test_summary_slices():
    cases = toy_case(hidden_items=[1], revealed_items=[0, 2, 3, 4])
    per_case = per_case_metrics(np.ones((1, 6)), cases)
    table = summarize(per_case, slices=(0, 1, 2, 3))
    assert list(table.index) == ["all", "3+"]
    assert table.loc["all", "cases"] == 1


def test_category_hit_counts_a_sibling_variant():
    cases = toy_case(hidden_items=[1], revealed_items=[])
    cats = np.array(["wardrobe", "wardrobe", "tape", "tape", "x", "y"])
    scores = np.array([[0.9, 0.0, 0.8, 0.0, 0.0, 0.0]])        # top-1 is item 0, sibling of hidden item 1
    m = per_case_metrics(scores, cases, ks=(1,), map_k=1, categories=cats).iloc[0]
    assert m["hr@1"] == 0 and m["cat_hr@1"] == 1
