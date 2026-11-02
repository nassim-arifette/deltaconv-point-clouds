"""A failed audit must never silently replace a complete paired comparison."""
import copy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import build_site_data


def complete_report():
    return {"status": "ok", "rows": [
        {"case_id": case, "method": method, "status": "ok", "shape": "sphere",
         "density": "uniform", "seed": seed, "sigma": 0., "normal_source": "pca"}
        for seed, case in enumerate(("case-0", "case-1")) for method in build_site_data.METHODS
    ]}


def test_complete_paired_report_is_exportable():
    build_site_data.validate_paired_results(complete_report())


@pytest.mark.parametrize("problem", [
    "partial_status", "failed_row", "missing_method", "missing_case", "duplicate",
    "unknown_method", "different_case", "different_metadata", "summary_failure", "empty",
])
def test_incomplete_audit_preserves_previous_site(tmp_path, monkeypatch, problem):
    report = complete_report()
    if problem == "partial_status":
        report["status"] = "partial_failure"
    elif problem == "failed_row":
        report["rows"][0]["status"] = "failed"
    elif problem == "missing_method":
        report["rows"] = [r for r in report["rows"] if r["method"] != build_site_data.METHODS[-1]]
    elif problem == "missing_case":
        report["rows"].pop()
    elif problem == "duplicate":
        report["rows"].append(copy.deepcopy(report["rows"][0]))
    elif problem == "unknown_method":
        report["rows"][0]["method"] = "future-method"
    elif problem == "different_case":
        report["rows"][0]["case_id"] = "unpaired-case"
    elif problem == "different_metadata":
        report["rows"][0]["sigma"] = .02
    elif problem == "summary_failure":
        report["summary"] = [{"method": build_site_data.METHODS[0], "failed": 1}]
    else:
        report["rows"] = []
    source = tmp_path / "results/cpu_reference"
    source.mkdir(parents=True)
    (source / "results.json").write_text(json.dumps(report), encoding="utf-8")
    site = tmp_path / "site"
    (site / "assets").mkdir(parents=True)
    previous = site / "assets/results.json"
    previous.write_bytes(b"previous verified export")
    monkeypatch.setattr(build_site_data, "ROOT", tmp_path)
    monkeypatch.setattr(build_site_data, "SITE", site)
    with pytest.raises(ValueError):
        build_site_data.main()
    assert previous.read_bytes() == b"previous verified export"
    assert not (site / "downloads").exists()
