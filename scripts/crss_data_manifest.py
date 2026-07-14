"""Record SHA-256 of the CRSS input files used by the real-data analysis.
Writes a NEW manifest; no existing result file is modified."""
import hashlib, json, pathlib, platform, subprocess, sys

RAW = pathlib.Path("<redacted-path>")
OUT = pathlib.Path("<redacted-path>")

def sha256(p: pathlib.Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

files = {}
for year in range(2016, 2022):
    for d in RAW.glob(f"CRSS{year}*"):
        for p in d.rglob("*"):
            if p.is_file() and p.name.lower() in ("vehicle.csv", "person.csv"):
                rel = str(p.relative_to(RAW))
                files[rel] = {"sha256": sha256(p), "bytes": p.stat().st_size, "year": year}

manifest = {
    "purpose": "Input-data fingerprint for the exploratory CRSS real-data analysis (Sec. 5.6).",
    "source": "NHTSA Crash Report Sampling System (CRSS), public release, downloaded from nhtsa.gov.",
    "note": "Files are not redistributed. These hashes let a third party verify that the files "
            "they download from NHTSA are the ones the reported estimates were computed on.",
    "consumed_by": [
        "scripts/realdata_tier2_fastresult_v2.py",
        "scripts/realdata_falsification_test.py",
        "scripts/realdata_changeover_design.py",
    ],
    "esc_treatment_source": "data/esc_equipment_list_nhtsa_api_v2.csv (NHTSA 5-Star Safety Ratings API)",
    "n_files": len(files),
    "files": dict(sorted(files.items())),
    "python_version": sys.version.split()[0],
    "platform": platform.platform(),
    "hostname": "redacted-host",
    "GPU": "none (CPU-only)",
}
OUT.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(f"wrote {OUT.name}: {len(files)} files")
for k, v in sorted(files.items()):
    print(f"  {k:38s} {v['sha256'][:16]}…  {v['bytes']/1e6:7.1f} MB")
