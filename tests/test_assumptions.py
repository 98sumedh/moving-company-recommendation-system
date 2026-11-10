import pytest

from movrec.assumptions import AssumptionError, find_problems, load_assumptions, render_markdown
from movrec.data_quality import build_report
from movrec.generator.generate import generate, write_dataset


def test_file_is_valid(A):
    assert len(A.keys()) > 40


def test_validation_catches_missing_source():
    entries = {"x": {"value": 1, "description": "d", "source_type": "public", "confidence": "high", "used_in": ["a"]}}
    problems = find_problems(entries)
    assert any("missing 'source'" in p for p in problems)
    assert any("needs a 'url'" in p for p in problems)


def test_bad_source_type_rejected(tmp_path):
    p = tmp_path / "a.yaml"
    p.write_text("x: {value: 1, description: d, source: s, source_type: vibes, confidence: high, used_in: [a]}\n")
    with pytest.raises(AssumptionError):
        load_assumptions(p)


def test_undefined_key_raises(A):
    with pytest.raises(KeyError):
        A["not_a_real_assumption"]


def test_every_assumption_is_used(tmp_path):
    """No dead documentation: generating data + the report reads every entry."""
    A = load_assumptions()
    data = generate(seed=1, n_bookings=1500, A=A)
    write_dataset(data, tmp_path, seed=1)
    import movrec.data_quality as dq
    original = dq.load_assumptions
    dq.load_assumptions = lambda: A
    try:
        build_report(tmp_path)
    finally:
        dq.load_assumptions = original
    unused = set(A.keys()) - A.used
    assert not unused, f"assumptions never read by the code: {sorted(unused)}"


def test_markdown_render(A):
    md = render_markdown(A)
    assert "synthetic" in md.lower()
    assert all(f"`{k}`" in md for k in A.keys())
