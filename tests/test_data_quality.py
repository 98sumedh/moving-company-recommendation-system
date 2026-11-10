from movrec.data_quality import build_report
from movrec.generator.generate import write_dataset


def test_report_builds_with_banner(small_data, tmp_path):
    manifest = write_dataset(small_data, tmp_path, seed=7)
    assert manifest["synthetic"] is True
    assert (tmp_path / "_hidden" / "segments.csv").exists()
    report = build_report(tmp_path)
    assert "SYNTHETIC DATA" in report
    for section in ("Cold start", "Add-ons per booking", "Seasonality", "Combinations", "DIAGNOSTICS ONLY"):
        assert section in report
