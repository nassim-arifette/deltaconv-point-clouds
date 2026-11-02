"""Copy measured artifacts into the website sources; never invent display metrics."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "web/public"
METHODS = ("A_upstream", "B_tensor_penalty", "C_invariant", "D_local_bandwidth")
CONDITION_FIELDS = ("shape", "density", "seed", "sigma", "normal_source")


def validate_paired_results(report):
    """Reject incomplete comparisons before changing any published artifact."""
    if report.get("status") != "ok":
        raise ValueError("site export requires an audit with status='ok'")
    rows = report.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("site export requires measured rows")
    cases = {method: set() for method in METHODS}
    conditions = {}
    for row in rows:
        if not isinstance(row, dict) or row.get("status") != "ok":
            raise ValueError("site export refuses failed or invalid measurement rows")
        method, case_id = row.get("method"), row.get("case_id")
        if method not in cases:
            raise ValueError(f"unexpected method: {method!r}")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError("each measurement needs a nonempty case_id")
        if case_id in cases[method]:
            raise ValueError(f"duplicate measurement: {method}/{case_id}")
        if any(field not in row for field in CONDITION_FIELDS):
            raise ValueError(f"missing condition metadata: {case_id}")
        condition = tuple(row[field] for field in CONDITION_FIELDS)
        if case_id in conditions and conditions[case_id] != condition:
            raise ValueError(f"inconsistent condition metadata: {case_id}")
        conditions[case_id] = condition
        cases[method].add(case_id)
    if any(not ids for ids in cases.values()):
        raise ValueError("all four methods must have measured cases")
    reference = cases[METHODS[0]]
    if any(ids != reference for ids in cases.values()):
        raise ValueError("the four methods must have identical case_id sets")
    if any(row.get("failed", 0) != 0 for row in report.get("summary", [])):
        raise ValueError("site export refuses reported failures")


def main():
    """Copy the measured artifacts the website links to into web/public/files."""
    source = ROOT / "results/cpu_reference"
    result = json.loads((source / "results.json").read_text(encoding="utf-8"))
    validate_paired_results(result)
    files = SITE / "files"
    files.mkdir(parents=True, exist_ok=True)
    for name in ("results.json", "metrics.csv"):
        shutil.copyfile(source / name, files / name)
    shutil.copyfile(ROOT / "THIRD_PARTY_NOTICES.md", files / "THIRD_PARTY_NOTICES.md")
    shutil.copyfile(ROOT / "DeltaConv_script_presentation.pdf", files / "presentation_notes_fr.pdf")
    note = (ROOT / "docs/research.md").read_text(encoding="utf-8")
    note = note.replace("../results/cpu_reference/", "").replace("../THIRD_PARTY_NOTICES.md", "THIRD_PARTY_NOTICES.md")
    (files / "research_note_fr.md").write_text(note, encoding="utf-8")
    for notebook in (ROOT / "notebooks").glob("*.ipynb"):
        shutil.copyfile(notebook, files / notebook.name)
    digests = {p.relative_to(SITE).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
               for p in sorted(files.iterdir()) if p.name != "export_manifest.json"}
    (files / "export_manifest.json").write_text(json.dumps({"schema_version": 2, "source": "results/cpu_reference", "sha256": digests}, indent=2) + "\n", encoding="utf-8")
    print("Copied measured artifacts into", files)


if __name__ == "__main__":
    main()
