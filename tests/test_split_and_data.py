import pytest

from movrec.data import ManifestMismatch, load_modeling_data, verify_manifest
from movrec.generator.generate import write_dataset
from movrec.split import time_split


@pytest.fixture(scope="module")
def modeling(small_data, tmp_path_factory):
    out = tmp_path_factory.mktemp("data")
    write_dataset(small_data, out, seed=7)
    return out, load_modeling_data(out)


def test_never_reads_hidden(modeling):
    _, data = modeling
    assert not any("segment" in c for c in data.bookings.columns)


def test_matrix_matches_lines(modeling, small_data):
    _, data = modeling
    assert data.matrix.sum() == len(small_data.booking_services)
    assert data.matrix.shape == (len(small_data.bookings), 45)
    assert data.truck_mask().sum() == 3


def test_manifest_detects_tampering(modeling):
    out, _ = modeling
    verify_manifest(out)
    path = out / "bookings.csv"
    original = path.read_bytes()
    path.write_bytes(original + b"\n")
    try:
        with pytest.raises(ManifestMismatch):
            load_modeling_data(out)
    finally:
        path.write_bytes(original)


def test_time_split_is_chronological_and_complete(modeling):
    _, data = modeling
    s = time_split(data.bookings, "2025-09-01", "2026-01-01")
    assert (s.train.astype(int) + s.val + s.test == 1).all()
    ts = data.bookings["booking_ts"]
    assert ts[s.train].max() < ts[s.val].min() <= ts[s.val].max() < ts[s.test].min()


def test_split_rejects_bad_order(modeling):
    _, data = modeling
    with pytest.raises(ValueError):
        time_split(data.bookings, "2026-01-01", "2025-01-01")
