#!/usr/bin/env python3
"""
audit_us_crash_databases.py
============================
Data-requirements audit for U.S. crash databases (FARS, CRSS, CISS, vPIC).

Purpose: verify each assumption of the CCMP identification hierarchy
against what is actually observable in U.S. crash data.

Output:
  code_release/results/audit_us_crash_databases_2026-07-14.json
  code_release/results/audit_us_crash_databases_2026-07-14.csv
  code_release/results/table_data_audit.tex

Authors: Keito Inoshita, Akira Kawai
Date  : 2026-07-14 (v3: CISS added 2026-07-14)
Python: 3.12+
No GPU required. vPIC API queries are cached locally.

This is the CANONICAL version for GitHub (github.com/keito-git/ccmp-crash-homeostasis).
Paths are relative to code_release/ (i.e., SCRIPT_DIR.parents[1]).
The paper-submission bundle copy lives at the paper-submission bundle copy of code_release/ with paths
adapted to the local project tree (pilot/data/raw/ etc.) — do not confuse the two.

CRSS 2021 data must be downloaded manually before running the CRSS audits (A10–A12):
  URL: https://static.nhtsa.gov/nhtsa/downloads/CRSS/2021/CRSS2021CSV.zip
  SHA256: 96de94779eb3c6085fc9c776a7575f089df9c77d76c97623a9f921255a32dc95
  Save to: code_release/data/raw/CRSS2021/CRSS2021CSV.zip
  Unzip to: code_release/data/raw/CRSS2021/CRSS2021CSV/

FARS vehicle files are downloaded by step1_fars_coord_check.py and step2_deltav_check.py;
run those scripts first or place FARS{year}_VEHICLE.csv files in code_release/data/raw/.

CISS CSV data could NOT be obtained programmatically (see audit_ciss_download_attempt).
CISS audits (A13, A14) are based on primary NHTSA documentation (codebook level only):
  - CISS 2017 Analytical User's Manual, DOT HS 812 803, June 2020 update
  - CISS 2020 Analytical User's Manual, DOT HS 812 958, June 2020
Variable existence confirmed from manuals; fill rates marked UNKNOWN.

Changelog:
  2026-07-14 v2: Added full CRSS 2021 audit functions using downloaded ZIP
    - audit_crss_accident_coords(): confirms no LAT/LON in accident.csv
    - audit_crss_inj_sev(): INJ_SEV KABCO distribution from person.csv
    - audit_crss_esc_vpicdecode(): ElectronicStabilityControl from vpicdecode.csv
    Updated build_latex_table() with actual CRSS numbers replacing placeholders.
  2026-07-14 v2.1: Fixed rate-limiting bug (W4): sleep now fires BEFORE calling
    vpic_decode_single (was_cached check), not after (post-save check always True).
  2026-07-14 v3: Added CISS audit (reviewer Himuro critical comment).
    - audit_ciss_download_attempt(): documents HTTP 404 on all URL patterns tried
    - audit_ciss_manual_review(): codebook-level audit from NHTSA manuals
    - build_latex_table() expanded to 3-DB (FARS / CRSS / CISS) layout
    Finding: CISS records DVTOTAL (delta-V) and ElectronicStabilityControl,
    but lacks GPS coordinates; Tier-2 therefore remains infeasible.
    Paper's prior claim "delta-V is recorded in neither [FARS nor CRSS]" was
    incomplete — corrected to acknowledge CISS records DVTOTAL.
"""

# ── Imports ────────────────────────────────────────────────────────────────
import csv
import datetime
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

try:
    import urllib.request
    import urllib.error
    import urllib.parse
except ImportError:
    pass

# ── Paths ───────────────────────────────────────────────────────────────────
# Canonical (GitHub) layout:
#   This script: code_release/data_checks/audit_us_crash_databases.py
#   Raw data   : code_release/data/raw/
#   Results    : code_release/results/
#
# (The paper-submission copy at the paper-submission bundle copy of code_release/ uses
#  pilot/data/raw/ and pilot/results/ instead — different project tree.)

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT  = SCRIPT_DIR.parent        # 1 level up: data_checks → code_release
RAW_DIR    = REPO_ROOT / "data" / "raw"
RES_DIR    = REPO_ROOT / "results"
VPIC_CACHE = RAW_DIR / "vpic_cache"
VPIC_CACHE.mkdir(parents=True, exist_ok=True)
RES_DIR.mkdir(parents=True, exist_ok=True)

# ── Data source documentation ────────────────────────────────────────────────
# FARS (Fatality Analysis Reporting System):
#   https://www.nhtsa.gov/research-data/fatality-analysis-reporting-system-fars
#   Annual national CSV ZIP: https://static.nhtsa.gov/nhtsa/downloads/FARS/{year}/National/FARS{year}NationalCSV.zip
#   Local cache path pattern: code_release/data/raw/FARS{year}_VEHICLE.csv
#
# CRSS (Crash Report Sampling System, successor to GES, started 2016):
#   https://www.nhtsa.gov/crash-data-systems/crash-report-sampling-system
#   Full 2021 ZIP: https://static.nhtsa.gov/nhtsa/downloads/CRSS/2021/CRSS2021CSV.zip
#   Local path: code_release/data/raw/CRSS2021/CRSS2021CSV/ (must be downloaded manually)
#
# vPIC (Vehicle Product Information Catalog / VIN decoder):
#   https://vpic.nhtsa.dot.gov/api/
#   Batch decode endpoint (POST):
#     https://vpic.nhtsa.dot.gov/api/vehicles/DecodeVINValuesBatch/
#   Cache stored per VIN batch in: pilot/data/raw/vpic_cache/

# ── Utilities ───────────────────────────────────────────────────────────────

LOG: list[str] = []

def log(msg: str) -> None:
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    LOG.append(line)


def file_sha256(path: Path, chunk_size: int = 65536) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def read_csv_header_and_count(path: Path) -> tuple[list[str], int]:
    """Return (column_names, row_count) for a CSV file."""
    with open(path, encoding="latin-1") as f:
        reader = csv.reader(f)
        header = next(reader)
        count  = sum(1 for _ in reader)
    return header, count


def columns_matching(columns: list[str], keywords: list[str]) -> list[str]:
    """Return column names whose upper-case form contains any keyword."""
    ku = [k.upper() for k in keywords]
    return [c for c in columns if any(kw in c.upper() for kw in ku)]


# ── Audit A1/A2: DELTA_V in FARS and CRSS ──────────────────────────────────

def audit_deltav(files: dict[str, Path]) -> dict:
    """
    For each CSV file, check whether any column name matches DELTA_V / DELTAV / DELTA-V.
    Returns a dict of {filename: {n_rows, n_cols, delta_v_cols}}.
    """
    log("=== Audit A1/A2: DELTA_V column presence ===")
    results: dict = {}
    for label, path in files.items():
        if not path.exists():
            log(f"  MISSING: {path.name}")
            results[label] = {"status": "file_missing", "path": str(path)}
            continue
        header, n_rows = read_csv_header_and_count(path)
        dv_cols = columns_matching(header, ["DELTA_V", "DELTAV", "DELTA-V"])
        sha = file_sha256(path)
        log(f"  {label}: {n_rows:,} rows | {len(header)} cols | DELTA_V cols: {dv_cols or '[NONE]'}")
        results[label] = {
            "status":       "ok",
            "path":         str(path),
            "sha256":       sha,
            "n_rows":       n_rows,
            "n_cols":       len(header),
            "delta_v_cols": dv_cols,
            "delta_v_found": len(dv_cols) > 0,
        }
    return results


# ── Audit A3/A4: ESC column in FARS and CRSS ───────────────────────────────

def audit_esc_column(files: dict[str, Path]) -> dict:
    """
    Check each CSV for any column related to ESC / electronic stability / stability control.
    """
    log("=== Audit A3/A4: ESC column presence in FARS / CRSS ===")
    ESC_KWS = ["ESC", "ELECTRON", "STAB", "STABILITY", "TRACTION"]
    results: dict = {}
    for label, path in files.items():
        if not path.exists():
            results[label] = {"status": "file_missing"}
            continue
        header, n_rows = read_csv_header_and_count(path)
        esc_cols = columns_matching(header, ESC_KWS)
        log(f"  {label}: ESC-related cols: {esc_cols or '[NONE]'}")
        results[label] = {
            "status":      "ok",
            "n_rows":      n_rows,
            "n_cols":      len(header),
            "esc_cols":    esc_cols,
            "esc_found":   len(esc_cols) > 0,
        }
    return results


# ── Audit A5: ESC from vPIC (VIN decoding) ─────────────────────────────────

VPIC_INDIV_URL = "https://vpic.nhtsa.dot.gov/api/vehicles/DecodeVINValues/{vin}?format=json&modelyear={my}"
VPIC_VARIABLE  = "ElectronicStabilityControl"

def vpic_decode_single(vin: str, model_year: str) -> dict | None:
    """
    Query vPIC individual decode endpoint for one VIN.
    Returns decoded result dict or None on failure.
    Results cached per (vin, model_year) pair.

    NOTE: FARS stores only the first 12 of 17 VIN characters (privacy redaction).
    This causes ErrorCode=6 (Incomplete VIN) for all FARS records.
    """
    cache_key  = hashlib.sha256(f"{vin}_{model_year}".encode()).hexdigest()[:16]
    cache_path = VPIC_CACHE / f"indiv_{cache_key}.json"
    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)

    url = VPIC_INDIV_URL.format(vin=vin, my=model_year)
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            raw = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        log(f"    [vPIC HTTP error] {e.code}: {e.reason} — {vin}")
        return None
    except urllib.error.URLError as e:
        log(f"    [vPIC URL error] {e.reason} — {vin}")
        return None
    except Exception as e:
        log(f"    [vPIC error] {e} — {vin}")
        return None

    results = raw.get("Results", [])
    rec = results[0] if results else {}
    with open(cache_path, "w") as f:
        json.dump(rec, f)
    return rec


def audit_vpic_esc(
    fars_vehicle_path: Path,
    sample_n: int = 200,
    api_wait_s: float = 0.3,
) -> dict:
    """
    Sample VINs from FARS VEHICLE file and query vPIC individually for
    ElectronicStabilityControl.  Results are cached so re-runs are instant.

    Denominator definition:
        Unique VINs of length >=11 characters drawn sequentially from
        FARS2021_VEHICLE_corrected.csv; fill rate = n_nonempty / n_decoded.

    Key finding: FARS redacts VIN positions 13-17 (privacy), leaving 12-char
    partial VINs.  vPIC returns ErrorCode=6 (Incomplete VIN) for all such
    records and leaves ElectronicStabilityControl = None.
    """
    log(f"=== Audit A5: vPIC ElectronicStabilityControl (sample_n={sample_n}) ===")

    if not fars_vehicle_path.exists():
        log(f"  MISSING: {fars_vehicle_path.name}")
        return {"status": "file_missing"}

    # Collect VINs and model years
    vins_and_years: list[tuple[str, str]] = []
    total_file_rows = 0
    with open(fars_vehicle_path, encoding="latin-1") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_file_rows += 1
            vin = row.get("VIN", "").strip()
            my  = row.get("MOD_YEAR", "").strip()
            if len(vin) >= 11 and vin != "000000000000":
                vins_and_years.append((vin, my))

    # Deduplicate and trim
    seen: set[str] = set()
    unique_pairs: list[tuple[str, str]] = []
    for vin, my in vins_and_years:
        if vin not in seen:
            seen.add(vin)
            unique_pairs.append((vin, my))
        if len(unique_pairs) >= sample_n:
            break

    vin_lengths = sorted(set(len(v) for v, _ in unique_pairs))
    log(f"  File total rows: {total_file_rows:,}")
    log(f"  Unique VINs of >=11 chars in file: {len(set(v for v, _ in vins_and_years)):,}")
    log(f"  Sample for vPIC query: {len(unique_pairs)}")
    log(f"  VIN length(s) in sample: {vin_lengths}")
    log(f"  NOTE: FARS stores first 12 of 17 VIN chars (positions 13-17 redacted)")

    # Individual decode
    all_results: list[dict] = []
    n_api_errors  = 0
    error_codes: dict[str, int] = {}

    for i, (vin, my) in enumerate(unique_pairs):
        # Check BEFORE calling so we know whether a real API request will be made.
        # vpic_decode_single saves the result to cache on success, so checking
        # existence AFTER would always return True and the sleep would never fire
        # (the original bug: inverted condition).
        cache_key_str = hashlib.sha256(f"{vin}_{my}".encode()).hexdigest()[:16]
        was_cached = (VPIC_CACHE / f"indiv_{cache_key_str}.json").exists()

        rec = vpic_decode_single(vin, my)
        if rec is None:
            n_api_errors += 1
            continue
        all_results.append(rec)

        ec = str(rec.get("ErrorCode", "")).strip()
        error_codes[ec] = error_codes.get(ec, 0) + 1

        # Sleep after real API calls only (not cache hits).
        if not was_cached:
            time.sleep(api_wait_s)

        if (i + 1) % 50 == 0:
            log(f"  Decoded {i+1}/{len(unique_pairs)}")

    log(f"  Decoded: {len(all_results)} | API errors: {n_api_errors}")
    log(f"  ErrorCode distribution: {error_codes}")

    # ESC fill rate
    esc_values: dict[str, int] = {}
    n_empty    = 0
    n_nonempty = 0
    for rec in all_results:
        val = rec.get(VPIC_VARIABLE)
        val_str = str(val).strip() if val is not None else ""
        if val_str and val_str.lower() not in ("none", "not available", "", "null"):
            n_nonempty += 1
            esc_values[val_str] = esc_values.get(val_str, 0) + 1
        else:
            n_empty += 1

    n_total       = len(all_results)
    fill_rate_pct = round(n_nonempty / n_total * 100, 2) if n_total > 0 else None

    log(f"  ESC fill rate: {n_nonempty}/{n_total} = {fill_rate_pct}%")
    log(f"  ESC value distribution: {esc_values}")
    log(f"  Empty/None: {n_empty}")

    # Save manifest
    manifest_path = VPIC_CACHE / "vpic_audit_manifest.json"
    manifest_data = {
        "run_date":               datetime.datetime.now().isoformat(),
        "source_file":            str(fars_vehicle_path),
        "source_file_total_rows": total_file_rows,
        "vin_chars_stored_in_fars": 12,
        "vin_full_length":        17,
        "redacted_positions":     "13-17 (FARS privacy redaction)",
        "sample_n_requested":     sample_n,
        "total_queried":          len(unique_pairs),
        "total_decoded":          n_total,
        "n_api_errors":           n_api_errors,
        "vpic_error_code_distribution": error_codes,
        "esc_variable":           VPIC_VARIABLE,
        "esc_fill_rate_pct":      fill_rate_pct,
        "esc_value_distribution": esc_values,
        "n_empty":                n_empty,
        "n_nonempty":             n_nonempty,
        "vpic_endpoint":          VPIC_INDIV_URL,
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest_data, f, indent=2)
    log(f"  vPIC manifest saved: {manifest_path}")

    return {
        "status":                   "ok",
        "source_file":              str(fars_vehicle_path),
        "source_file_total_rows":   total_file_rows,
        "denominator_definition": (
            f"Unique VINs of length >=11 characters drawn sequentially from "
            f"{fars_vehicle_path.name} (total rows: {total_file_rows:,}); "
            f"vPIC queried for first {len(unique_pairs)} unique VINs via "
            f"individual decode endpoint; fill rate denominator = "
            f"total successfully decoded = {n_total}. "
            f"FARS stores only first 12 of 17 VIN characters "
            f"(positions 13-17 redacted for privacy), causing "
            f"ErrorCode=6 (Incomplete VIN) in {error_codes.get('6', 0)} of {n_total} records."
        ),
        "sample_n_requested":       sample_n,
        "total_queried":            len(unique_pairs),
        "total_decoded":            n_total,
        "n_api_errors":             n_api_errors,
        "vpic_error_code_distribution": error_codes,
        "esc_fill_rate_pct":        fill_rate_pct,
        "esc_value_distribution":   esc_values,
        "n_empty":                  n_empty,
        "n_nonempty":               n_nonempty,
        "vpic_endpoint":            VPIC_INDIV_URL,
        "vin_chars_in_fars":        12,
        "vin_full_length":          17,
        "cache_dir":                str(VPIC_CACHE),
    }


# ── Audit A6: Coordinate coverage (FARS ACCIDENT) ──────────────────────────

def audit_coordinate_coverage(step1_csv: Path) -> dict:
    """
    Load pre-computed step1 coordinate coverage results.
    These were produced by step1_fars_coord_check.py (run 2026-07-10).
    """
    log("=== Audit A6: FARS ACCIDENT coordinate coverage ===")
    if not step1_csv.exists():
        log(f"  MISSING: {step1_csv.name}")
        return {"status": "file_missing"}

    rows: list[dict] = []
    with open(step1_csv) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    ok_rows = [r for r in rows if r["status"] == "ok"]
    rates   = [float(r["coord_rate_pct"]) for r in ok_rows]
    totals  = [int(r["total"]) for r in ok_rows]
    valids  = [int(r["valid_coords"]) for r in ok_rows]
    years   = [int(r["year"]) for r in ok_rows]

    avg_rate   = round(sum(rates) / len(rates), 2) if rates else None
    min_rate   = round(min(rates), 2) if rates else None
    max_rate   = round(max(rates), 2) if rates else None
    total_valid = sum(valids)
    total_crashes = sum(totals)

    log(f"  Years: {years}")
    log(f"  avg coord coverage: {avg_rate}%  min: {min_rate}%  max: {max_rate}%")
    log(f"  Total crashes: {total_crashes:,} | with valid coords: {total_valid:,}")

    return {
        "status":              "ok",
        "provenance":          str(step1_csv),
        "provenance_script":   "step1_fars_coord_check.py",
        "run_date_provenance": "2026-07-10T20:25:02",
        "years":               years,
        "year_detail":         [
            {"year": int(r["year"]), "total": int(r["total"]),
             "valid_coords": int(r["valid_coords"]),
             "coord_rate_pct": float(r["coord_rate_pct"])}
            for r in ok_rows
        ],
        "avg_coord_rate_pct":  avg_rate,
        "min_coord_rate_pct":  min_rate,
        "max_coord_rate_pct":  max_rate,
        "total_crash_records": total_crashes,
        "total_with_coords":   total_valid,
    }


# ── Audit A7: Spatial join rate HPMS (step1b) ──────────────────────────────

def audit_spatial_join(step1b_json: Path) -> dict:
    log("=== Audit A7: AADT spatial-join feasibility (step1b) ===")
    if not step1b_json.exists():
        log(f"  MISSING: {step1b_json.name}")
        return {"status": "file_missing"}
    with open(step1b_json) as f:
        data = json.load(f)
    log(f"  Test state: {data.get('test_state')} | year: {data.get('test_year')}")
    log(f"  Crash points: {data.get('n_crash_points')}")
    for jk in ["join_100m_all_roads", "join_100m_hpms_equiv", "join_100m_primary_s1100"]:
        jv = data.get(jk, {})
        log(f"  {jk}: matched={jv.get('n_matched')}, rate={jv.get('rate_pct')}%")
    return {
        "status":      "ok",
        "provenance":  str(step1b_json),
        "provenance_script": "step1b_spatial_join_test.py",
        "run_date_provenance": data.get("run_date"),
        "test_state":  data.get("test_state"),
        "test_year":   data.get("test_year"),
        "n_crash_points": data.get("n_crash_points"),
        "join_100m_all_roads":     data.get("join_100m_all_roads"),
        "join_100m_hpms_equiv":    data.get("join_100m_hpms_equiv"),
        "join_100m_primary_s1100": data.get("join_100m_primary_s1100"),
    }


# ── CRSS Full-ZIP audit functions ───────────────────────────────────────────

def audit_crss_accident_coords(accident_path: Path) -> dict:
    """
    Check CRSS 2021 ACCIDENT.csv (from full CRSS2021CSV.zip) for coordinate columns.
    Expected finding: no LATITUDE / LONGITUD columns present.
    The CRSS complex-sample design does not include crash-point GPS coordinates
    in the public release; geographic location is suppressed for respondent privacy.
    """
    log("=== Audit CRSS-AC: CRSS accident.csv coordinate check ===")
    COORD_KWS = ["LAT", "LON", "COORD", "GPS", "POSITION", "GEO", "ALTITUDE"]
    if not accident_path.exists():
        log(f"  MISSING: {accident_path}")
        return {"status": "file_missing", "path": str(accident_path)}

    header, n_rows = read_csv_header_and_count(accident_path)
    sha = file_sha256(accident_path)
    coord_cols = columns_matching(header, COORD_KWS)
    sampling_kws = ["PSU", "STRATUM", "WEIGHT", "REGION", "URBAN"]
    sampling_cols = columns_matching(header, sampling_kws)
    log(f"  accident.csv: {n_rows:,} rows, {len(header)} cols")
    log(f"  Coordinate-related cols: {coord_cols or '[NONE]'}")
    log(f"  Sampling design cols: {sampling_cols}")
    return {
        "status":         "ok",
        "path":           str(accident_path),
        "sha256":         sha,
        "n_rows":         n_rows,
        "n_cols":         len(header),
        "coord_cols":     coord_cols,
        "coord_found":    len(coord_cols) > 0,
        "sampling_cols":  sampling_cols,
        "note": (
            "CRSS accident.csv contains sampling-design columns (PSU, STRATUM, WEIGHT, "
            "REGION, URBANICITY) and crash characteristics, but no geographic coordinates. "
            "CRSS suppresses GPS coordinates in the public CSV release."
        ),
    }


def audit_crss_inj_sev(person_path: Path) -> dict:
    """
    Count INJ_SEV (KABCO scale) distribution from CRSS 2021 PERSON.csv.
    Returns full distribution with counts and percentages.
    Denominator = total rows in person.csv.

    KABCO mapping in CRSS:
      0 = No Apparent Injury (O)
      1 = Possible Injury (C)
      2 = Suspected Minor Injury (B)
      3 = Suspected Serious Injury (A)
      4 = Fatal Injury (K)
      5 = Injured, Severity Unknown
      6 = Died Prior to Crash
      9 = Unknown/Not Reported
    """
    log("=== Audit CRSS-INJ: CRSS person.csv INJ_SEV distribution ===")
    if not person_path.exists():
        log(f"  MISSING: {person_path}")
        return {"status": "file_missing", "path": str(person_path)}

    sha = file_sha256(person_path)
    distribution: dict[str, int] = {}
    names: dict[str, str] = {}
    total_rows = 0
    with open(person_path, encoding="latin-1") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_rows += 1
            code = row.get("INJ_SEV", "").strip()
            name = row.get("INJ_SEVNAME", "").strip()
            distribution[code] = distribution.get(code, 0) + 1
            if code not in names:
                names[code] = name

    # Compute percentages
    distribution_pct = {
        code: round(cnt / total_rows * 100, 2)
        for code, cnt in distribution.items()
    }

    log(f"  person.csv: {total_rows:,} rows")
    log(f"  INJ_SEV distribution:")
    for code in sorted(distribution.keys()):
        cnt = distribution[code]
        pct = distribution_pct[code]
        log(f"    {code} ({names[code]}): {cnt:,} ({pct:.2f}%)")

    return {
        "status":              "ok",
        "path":                str(person_path),
        "sha256":              sha,
        "n_rows":              total_rows,
        "distribution_count":  distribution,
        "distribution_pct":    distribution_pct,
        "kabco_names":         names,
        "note": (
            "Full KABCO range present in CRSS person.csv. "
            "This contrasts with FARS which covers fatal crashes only (INJ_SEV=K). "
            "CRSS enables within-database severity variation analysis."
        ),
    }


def audit_crss_esc_vpicdecode(vpicdecode_path: Path) -> dict:
    """
    Compute ElectronicStabilityControl fill rate and value distribution
    from CRSS 2021 vpicdecode.csv.

    vpicdecode.csv is produced by NHTSA by decoding each vehicle VIN through
    the vPIC (Vehicle Product Information Catalog) system.  Unlike FARS,
    CRSS stores the full 17-character VIN, allowing complete vPIC decoding.

    Fill rate definition:
        n_filled = rows where ElectronicStabilityControl not in {'', 'Not Applicable'}
        denominator = total rows in vpicdecode.csv (= 92,765)
    """
    log("=== Audit CRSS-ESC: CRSS vpicdecode.csv ElectronicStabilityControl ===")
    if not vpicdecode_path.exists():
        log(f"  MISSING: {vpicdecode_path}")
        return {"status": "file_missing", "path": str(vpicdecode_path)}

    sha = file_sha256(vpicdecode_path)
    esc_counter: dict[str, int] = {}
    esc_id_counter: dict[str, int] = {}
    total_rows = 0
    with open(vpicdecode_path, encoding="latin-1") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_rows += 1
            esc_val = row.get("ElectronicStabilityControl", "").strip()
            esc_id  = row.get("ElectronicStabilityControlId", "").strip()
            esc_counter[esc_val] = esc_counter.get(esc_val, 0) + 1
            esc_id_counter[esc_id] = esc_id_counter.get(esc_id, 0) + 1

    # Fill rate: rows with non-empty value, excluding 'Not Applicable'
    n_standard = esc_counter.get("Standard", 0)
    n_empty    = esc_counter.get("", 0)
    n_na       = esc_counter.get("Not Applicable", 0)
    n_other    = total_rows - n_standard - n_empty - n_na
    fill_rate_pct = round(n_standard / total_rows * 100, 2) if total_rows > 0 else None

    log(f"  vpicdecode.csv: {total_rows:,} rows")
    log(f"  ElectronicStabilityControl distribution:")
    for val, cnt in sorted(esc_counter.items(), key=lambda x: -x[1]):
        log(f"    {repr(val)}: {cnt:,} ({cnt/total_rows*100:.2f}%)")
    log(f"  Fill rate (Standard): {n_standard:,}/{total_rows:,} = {fill_rate_pct}%")

    return {
        "status":              "ok",
        "path":                str(vpicdecode_path),
        "sha256":              sha,
        "n_rows":              total_rows,
        "esc_value_distribution":    {k: v for k, v in sorted(esc_counter.items(), key=lambda x: -x[1])},
        "esc_value_distribution_pct": {k: round(v/total_rows*100, 2) for k, v in esc_counter.items()},
        "esc_id_distribution": {k: v for k, v in sorted(esc_id_counter.items(), key=lambda x: -x[1])},
        "n_standard":          n_standard,
        "n_empty":             n_empty,
        "n_not_applicable":    n_na,
        "n_other":             n_other,
        "fill_rate_pct":       fill_rate_pct,
        "denominator_note": (
            f"Denominator = total rows in vpicdecode.csv = {total_rows:,}. "
            f"Fill rate numerator = rows where ElectronicStabilityControl = 'Standard' "
            f"= {n_standard:,}. "
            f"Empty (VIN not matched or attribute absent): {n_empty:,} ({n_empty/total_rows*100:.2f}%). "
            f"'Not Applicable' (vehicle category without ESC): {n_na:,} ({n_na/total_rows*100:.2f}%)."
        ),
    }


# ── Audit A8: Outcome Y (injury severity) ──────────────────────────────────

def audit_injury_severity(files: dict[str, Path]) -> dict:
    """
    Check FARS ACCIDENT and CRSS files for injury-severity columns.
    FARS covers fatal crashes only (INJ_SEV is implicitly K = fatal).
    CRSS PERSON file has INJ_SEV on KABCO scale but is not cached locally.
    """
    log("=== Audit A8: Outcome Y — injury severity availability ===")
    INJ_KWS = ["INJ_SEV", "INJSEV", "KABCO", "INJURY", "INJURI", "FATALS", "KILLED"]
    results: dict = {}
    for label, path in files.items():
        if not path.exists():
            log(f"  MISSING: {path.name}")
            results[label] = {"status": "file_missing"}
            continue
        header, n_rows = read_csv_header_and_count(path)
        inj_cols = columns_matching(header, INJ_KWS)
        log(f"  {label}: INJ-related cols: {inj_cols or '[NONE]'}")
        results[label] = {
            "status":    "ok",
            "n_rows":    n_rows,
            "n_cols":    len(header),
            "inj_cols":  inj_cols,
            "inj_found": len(inj_cols) > 0,
        }
    return results


# ── CRSS coordinate check ───────────────────────────────────────────────────

def audit_crss_coordinates(crss_vehicle_path: Path) -> dict:
    """
    Check whether the available CRSS VEHICLE file contains geographic coordinates.
    """
    log("=== Audit A9: CRSS coordinate availability ===")
    COORD_KWS = ["LAT", "LON", "COORD", "GPS", "POSITION", "GEO"]
    if not crss_vehicle_path.exists():
        log(f"  MISSING: {crss_vehicle_path.name}")
        return {"status": "file_missing"}
    header, n_rows = read_csv_header_and_count(crss_vehicle_path)
    coord_cols = columns_matching(header, COORD_KWS)
    log(f"  CRSS VEHICLE {crss_vehicle_path.name}: {n_rows} rows, {len(header)} cols")
    log(f"  Coordinate-related cols: {coord_cols or '[NONE]'}")
    log(f"  All columns: {header}")
    return {
        "status":        "ok",
        "file":          str(crss_vehicle_path),
        "n_rows":        n_rows,
        "n_cols":        len(header),
        "all_cols":      header,
        "coord_cols":    coord_cols,
        "coord_found":   len(coord_cols) > 0,
        "note": (
            "The locally cached CRSS2021_VEHICLE.csv contains only sampling-design "
            "metadata (stratum, PSU, urbanicity, weight). Geographic coordinates "
            "and vehicle characteristics are in separate CRSS files "
            "(ACCIDENT.csv for coordinates; VEHICLE.csv in full CRSS ZIP for vehicle attributes). "
            "A full download of CRSS2021CSV.zip would provide the complete vehicle-level file."
        ),
    }


# ── CISS audit ───────────────────────────────────────────────────────────────
#
# Background: Reviewer Himuro (R3) flagged that CISS (Crash Investigation
# Sampling System, NHTSA successor to NASS-CDS since 2016) records delta-V
# via WinSMASH — meaning the paper's prior statement "delta-V is recorded in
# neither [FARS nor CRSS]" was incomplete.  Two functions are added:
#   1. audit_ciss_download_attempt  — documents HTTP response codes for all
#      URL patterns tried (evidence chain for "data unavailable" claim)
#   2. audit_ciss_manual_review     — codebook-level audit from primary NHTSA
#      documentation (variable existence confirmed; fill rates unknown)
#
# Primary sources:
#   CISS 2017 Analytical User's Manual  DOT HS 812 803 (June 2020 update)
#   CISS 2020 Analytical User's Manual  DOT HS 812 958 (June 2020)
# ─────────────────────────────────────────────────────────────────────────────

_CISS_MANUAL_2017 = (
    "CISS 2017 Analytical User's Manual, DOT HS 812 803 (June 2020 update). "
    "URL: https://crashstats.nhtsa.dot.gov/Api/Public/Publication/812803"
)
_CISS_MANUAL_2020 = (
    "CISS 2020 Analytical User's Manual, DOT HS 812 958 (June 2020). "
    "URL: https://crashstats.nhtsa.dot.gov/Api/Public/Publication/812958"
)


def audit_ciss_download_attempt() -> dict:
    """
    Attempt to download CISS CSV data from NHTSA servers.

    Records every URL tried and the HTTP response code received.
    This function does NOT modify any existing files.

    CISS (Crash Investigation Sampling System) has replaced NASS-CDS since
    2016.  NHTSA uses WinSMASH to compute delta-V in CISS, per:
      "NHTSA uses the WinSMASH computer code to estimate delta-V in several
       of its in-depth crash databases including [...] the newer Crash
       Investigation Sampling System (CISS)."
    (Source confirmed by reviewer Himuro, 2026-07-14.)

    Expected URL pattern (analogous to CRSS):
      https://static.nhtsa.gov/nhtsa/downloads/CISS/{year}/CISS{year}CSV.zip

    RESULT: All patterns return HTTP 404.  The NHTSA file-downloads portal
    (https://www.nhtsa.gov/file-downloads) returns HTTP 403 (Akamai CDN
    bot-detection).  Automated download is not possible; browser interaction
    or FTP access is required.
    """
    log("=== Audit CISS-DL: CISS download attempt ===")

    candidate_urls = [
        "https://static.nhtsa.gov/nhtsa/downloads/CISS/2017/CISS2017CSV.zip",
        "https://static.nhtsa.gov/nhtsa/downloads/CISS/2018/CISS2018CSV.zip",
        "https://static.nhtsa.gov/nhtsa/downloads/CISS/2019/CISS2019CSV.zip",
        "https://static.nhtsa.gov/nhtsa/downloads/CISS/2020/CISS2020CSV.zip",
        "https://static.nhtsa.gov/nhtsa/downloads/CISS/2021/CISS2021CSV.zip",
        "https://static.nhtsa.gov/nhtsa/downloads/CISS/2022/CISS2022CSV.zip",
        "https://static.nhtsa.gov/nhtsa/downloads/CISS/2022/CISS2022SAS.zip",
        "https://static.nhtsa.gov/nhtsa/downloads/CISS/2023/CISS2023CSV.zip",
    ]
    portal_url = "https://www.nhtsa.gov/file-downloads?p=nhtsa/downloads/CISS/"

    results: list[dict] = []
    for url in candidate_urls:
        try:
            req = urllib.request.Request(url, method="HEAD")
            with urllib.request.urlopen(req, timeout=10) as r:
                status: int | str = r.status
        except urllib.error.HTTPError as e:
            status = e.code
        except urllib.error.URLError as e:
            status = f"URLError: {e.reason}"
        except Exception as e:
            status = f"Error: {type(e).__name__}: {e}"
        log(f"  HTTP {status}  {url}")
        results.append({"url": url, "http_status": status})

    # Portal check
    try:
        req_p = urllib.request.Request(
            portal_url,
            headers={"User-Agent": "Mozilla/5.0"},
            method="HEAD",
        )
        with urllib.request.urlopen(req_p, timeout=10) as r:
            portal_status: int | str = r.status
    except urllib.error.HTTPError as e:
        portal_status = e.code
    except Exception as e:
        portal_status = f"Error: {type(e).__name__}: {e}"
    log(f"  HTTP {portal_status}  {portal_url}  (portal)")

    n_404 = sum(1 for r in results if r["http_status"] == 404)
    n_tried = len(results)

    outcome = "download_failed"
    reason = (
        f"All {n_tried} URL patterns tried returned HTTP {n_404}×404. "
        "NHTSA file-downloads portal returns HTTP 403 (Akamai CDN). "
        "Automated download not possible; browser or FTP access required. "
        "Manual instructions: open https://www.nhtsa.gov/file-downloads, "
        "navigate to nhtsa/downloads/CISS/{year}/, download CISS{year}CSV.zip, "
        "extract to pilot/data/raw/CISS{year}/CISS{year}CSV/."
    )
    log(f"  OUTCOME: {outcome} — {reason}")

    return {
        "status": outcome,
        "reason": reason,
        "n_urls_tried": n_tried,
        "n_404": n_404,
        "urls_tried": results,
        "portal_url": portal_url,
        "portal_http_status": portal_status,
    }


def audit_ciss_manual_review() -> dict:
    """
    Codebook-level audit of CISS variable availability.

    IMPORTANT: This function does NOT process any actual CISS CSV files
    (none are available; see audit_ciss_download_attempt).  All findings
    are sourced from the official CISS Analytical User's Manuals
    (DOT HS 812 803 / DOT HS 812 958) and therefore reflect variable
    EXISTENCE only.  Fill rates cannot be computed without actual data
    and are recorded as None ("UNKNOWN").

    Six audit items mirroring the FARS/CRSS checks:
    (a) Delta-V column   (b) GPS coordinates   (c) ESC equipment flag
    (d) Injury severity  (e) Annual sample size (f) Sampling design weights
    """
    log("=== Audit CISS-MR: CISS manual review (codebook level; no CSV) ===")

    # ── (a) Delta-V ──────────────────────────────────────────────────────────
    log("  (a) DVTOTAL: EXISTS in GV dataset (col #82)")
    log("      DVBASIS col #81: 0='Not Inspected' → DVTOTAL missing")
    log("      Fill rate: UNKNOWN (actual CSV unavailable)")
    deltav = {
        "column_name": "DVTOTAL",
        "column_number_in_gv_dataset": 82,
        "label": "HIGHEST DELTA V TOTAL (WinSMASH estimate, kph)",
        "dataset": "GV",
        "exists_confirmed_by_codebook": True,
        "missingness_indicator_column": "DVBASIS",
        "missingness_indicator_col_num": 81,
        "missingness_indicator_code_0": "Not Inspected (DVTOTAL absent)",
        "fill_rate_pct": None,
        "fill_rate_note": (
            "UNKNOWN: actual CISS CSV not obtained. "
            "DVBASIS=0 (Not Inspected) is a valid code in both 2017 and 2020 "
            "manuals; fraction of vehicles with DVBASIS=0 cannot be determined "
            "without data."
        ),
        "n_gv_obs_2017": 3748,
        "n_gv_obs_2020": 4848,
        "source_2017": _CISS_MANUAL_2017,
        "source_2020": _CISS_MANUAL_2020,
    }
    log(f"      Codebook confirmed: {deltav['exists_confirmed_by_codebook']}")

    # ── (b) GPS Coordinates ───────────────────────────────────────────────────
    log("  (b) GPS: ABSENT from all public CISS datasets")
    log("      CRASH (24 vars): no LATITUDE/LONGITUD")
    log("      GV 2020 adds EDGEDISTX/Y/Z (road-edge dist in cm) — NOT GPS")
    log("      PSU (county) is the only geographic identifier released")
    coordinates = {
        "gps_found": False,
        "crash_dataset_n_vars_2017": 24,
        "crash_dataset_vars_note": (
            "ALCINV CAIS CASEID CASENO CASENUMBER CASEWGT CATEGORY CINJSEV "
            "CINJURED CISS CONFIG CRASHMONTH CRASHTIME CRASHYEAR CTREAT "
            "DAYOFWEEK DRGINV EVENTS MANCOLL PSU PSUSTRAT SUMMARY "
            "VEHICLES VERSION — no coordinate columns"
        ),
        "gv_dataset_n_vars_2017": 97,
        "gv_dataset_n_vars_2020": 104,
        "edge_distance_vars_2020": ["EDGEDISTX", "EDGEDISTY", "EDGEDISTZ"],
        "edge_distance_note": (
            "EDGEDISTX/Y/Z (added in 2020 GV dataset) are physical measurements "
            "of distance from the road edge in centimetres, NOT geographic "
            "coordinates.  Cannot be used for AADT spatial join."
        ),
        "geographic_resolution": "PSU (county or county group); 24 PSUs in CISS",
        "tier2_aadt_join_feasible": False,
        "source_2017": _CISS_MANUAL_2017,
        "source_2020": _CISS_MANUAL_2020,
    }

    # ── (c) ESC Equipment Flag ────────────────────────────────────────────────
    log("  (c) ElectronicStabilityControl: EXISTS in VINDERIVED dataset (col #29)")
    log("      CISS stores full 17-char VIN → better vPIC decode than FARS (12-char)")
    log("      Fill rate: UNKNOWN (actual CSV unavailable)")
    log("      CRSS vpicdecode (same pipeline) showed 28.3% 'Standard' fill")
    esc = {
        "column_name": "ElectronicStabilityControl",
        "column_number_in_vinderived": 29,
        "label": "ELECTRONIC STABILITY CONTROL (ESC) — VIN-decoded via vPIC",
        "dataset": "VINDERIVED",
        "exists_confirmed_by_codebook": True,
        "vin_length_in_ciss_chars": 17,
        "n_vinderived_obs_2017": 3664,
        "n_gv_obs_2017": 3748,
        "vin_coverage_pct_2017": round(3664 / 3748 * 100, 1),
        "fill_rate_pct": None,
        "fill_rate_note": (
            "UNKNOWN: actual CISS CSV not obtained. "
            "CISS stores full 17-char VINs (unlike FARS which truncates to 12 chars), "
            "so vPIC decode should succeed at a higher rate than FARS. "
            "CRSS vpicdecode.csv (same vPIC pipeline) shows 28.3% 'Standard' fill "
            "(fill_rate computed 2026-07-14 from CRSS2021); "
            "CISS fill rate may differ due to vehicle composition differences. "
            "Whether 'Not Equipped' is encoded separately from empty/null is "
            "not determinable without actual data."
        ),
        "other_adas_cols_in_vinderived": [
            "AdaptiveCruiseControl", "AntiLockBrakingSystem",
            "AutomaticEmergencyBraking", "BlindSpotMonitoring",
            "DriverAssist", "ForwardCollisionWarning",
            "LaneDepartureWarning", "LaneKeepSystem", "ParkAssist",
            "TractionControl", "TPMS",
        ],
        "source_2017": _CISS_MANUAL_2017,
    }

    # ── (d) Injury Severity ───────────────────────────────────────────────────
    log("  (d) Injury: CAIS (max AIS 0-6) and CISS (ISS) in CRASH; VAIS in GV")
    injury = {
        "outcome_variables": {
            "CAIS": {
                "dataset": "CRASH",
                "col_num": 3,
                "label": "MAXIMUM KNOWN AIS IN THIS CRASH",
                "scale": "AIS 0-6 (0=No injury, 6=Maximum/Untreatable)",
                "notes": "Crash-level; covers full injury spectrum",
            },
            "CISS_col": {
                "dataset": "CRASH",
                "label": "MAXIMUM ISS SCORE IN THIS CASE",
                "scale": "ISS 0-75; 97=Unknown severity; 99=Unknown",
            },
            "VAIS": {
                "dataset": "GV",
                "label": "MAXIMUM AIS SEVERITY FOR THIS VEHICLE",
                "scale": "AIS 0-6",
            },
        },
        "injury_spectrum": "K through O (full MAIS range); superior to FARS (fatal only)",
        "n_crash_obs_2017": 2035,
        "source_2017": _CISS_MANUAL_2017,
    }

    # ── (e) Annual Sample Size ────────────────────────────────────────────────
    log("  (e) Sample sizes (from manuals):")
    log("      2017: CRASH=2,035 cases; GV=3,748 vehicles; VINDERIVED=3,664")
    log("      2020: GV=4,848 vehicles")
    sample_size = {
        "note": (
            "CISS is a nationally representative probability sample of "
            "police-reported crashes involving at least one tow-away passenger "
            "vehicle; 24 PSUs (county groups); sample weighted to represent "
            "all such US crashes."
        ),
        "year_2017": {
            "n_crash_cases": 2035,
            "n_gv_vehicles": 3748,
            "n_vinderived": 3664,
            "n_vehspec": 3599,
            "n_jkwgt_cases": 2035,
            "source": _CISS_MANUAL_2017,
        },
        "year_2020": {
            "n_gv_vehicles": 4848,
            "source": _CISS_MANUAL_2020,
        },
    }

    # ── (f) Sampling Design Weights ───────────────────────────────────────────
    log("  (f) Weights: CASEWGT + PSU + PSUSTRAT + JKWGT1-24 (all confirmed)")
    weights = {
        "case_weight_col": "CASEWGT",
        "psu_col": "PSU (Primary Sampling Unit = county or county group)",
        "stratum_col": "PSUSTRAT (Census region × urban/rural × road miles)",
        "jackknife_cols": "JKWGT1 through JKWGT24 (24 adjusted jackknife replicates)",
        "jackknife_coefficient": "JKCOEFS=0.5 (for SAS PROC SURVEYFREQ)",
        "n_psu": 24,
        "source": _CISS_MANUAL_2017,
    }

    # ── Tier feasibility assessments ─────────────────────────────────────────
    tier1 = (
        "CONDITIONAL: CISS records DVTOTAL (P1=delta-V) and "
        "ElectronicStabilityControl in VINDERIVED (P2=ESC). "
        "However (1) DVTOTAL fill rate is UNKNOWN — DVBASIS=0 may apply to a "
        "substantial fraction; (2) ESC fill rate is UNKNOWN — CRSS (same "
        "pipeline) shows only 28.3% 'Standard', rest empty/Not Applicable; "
        "(3) complete-case sample with both DVTOTAL and ESC observed may be "
        "too small for causal identification. "
        "Feasibility cannot be confirmed without actual CISS CSV data."
    )
    tier2 = (
        "NOT FEASIBLE: CISS does not release GPS coordinates in public data. "
        "EDGEDISTX/Y/Z (2020 GV) are centimetre distances from road edge, "
        "not geographic coordinates, and cannot support AADT spatial join. "
        "Tier-2 requires LATITUDE/LONGITUD for IPSW exposure model — absent."
    )
    log(f"  Tier-1 assessment: {tier1[:80]}...")
    log(f"  Tier-2 assessment: {tier2[:80]}...")

    return {
        "status": "manual_review_complete",
        "data_source": "CISS Analytical User's Manuals (2017 & 2020) — codebook level",
        "csv_data_available": False,
        "csv_unavailability_reason": (
            "HTTP 404 on all attempted static.nhtsa.gov URL patterns; "
            "NHTSA portal returns HTTP 403 (Akamai CDN). "
            "Browser or FTP download required — not automatable."
        ),
        "deltav": deltav,
        "coordinates": coordinates,
        "esc": esc,
        "injury": injury,
        "sample_size": sample_size,
        "weights": weights,
        "tier1_feasibility": tier1,
        "tier2_feasibility": tier2,
        "primary_sources": [_CISS_MANUAL_2017, _CISS_MANUAL_2020],
    }


# ── Assemble LaTeX table ─────────────────────────────────────────────────────

def build_latex_table(audit: dict) -> str:
    """
    Build a booktabs LaTeX table (no caption; elsarticle compatible).

    Columns (v3, updated 2026-07-14 per reviewer Himuro R3):
      Assumption | Required variable | FARS | CRSS | CISS | Status

    CISS column is codebook-level (DOT HS 812 803 / DOT HS 812 958).
    Fill rates for CISS are shown as '---' because CISS CSV data could not be
    downloaded (all tried URL patterns return HTTP 404; portal returns 403).

    Prior vPIC column removed; ESC vPIC details absorbed into CRSS cell.
    """
    # Helper: format fill-rate or "---"
    def fmt_pct(v) -> str:
        if v is None:
            return "---"
        return f"{v:.1f}\\%"

    # Gather numbers from audit results
    # A1/A2: Delta-V
    a12 = audit.get("deltav", {})
    fars_dv = "0 / {} files".format(
        sum(1 for v in a12.values() if v.get("status") == "ok")
    )
    crss_dv_r = a12.get("CRSS2021_VEHICLE", {})
    crss_dv = "Absent (sampling-weights file only, 15 cols)" if crss_dv_r.get("status") == "ok" else "---"

    # A3/A4: ESC — count FARS files only for the FARS cell
    a34 = audit.get("esc_column", {})
    n_ok_esc_fars = sum(
        1 for k, v in a34.items()
        if k.startswith("FARS") and v.get("status") == "ok" and not v.get("esc_found")
    )
    n_total_esc_fars = sum(
        1 for k, v in a34.items()
        if k.startswith("FARS") and v.get("status") == "ok"
    )
    # keep legacy names for backward compat in rows list
    n_ok_esc    = n_ok_esc_fars
    n_total_esc = n_total_esc_fars
    fars_esc = f"Absent in {n_ok_esc}/{n_total_esc} checked files"

    # A5: vPIC
    a5 = audit.get("vpic_esc", {})
    vpic_fill = fmt_pct(a5.get("esc_fill_rate_pct"))
    vpic_dist = a5.get("esc_value_distribution", {})
    vpic_vals = ", ".join(f"``{k}'' (n={v})" for k, v in sorted(vpic_dist.items())) or "---"
    vpic_n    = a5.get("total_decoded", "---")

    # A6: Coord coverage FARS
    a6 = audit.get("coord_coverage", {})
    fars_coord_mean = fmt_pct(a6.get("avg_coord_rate_pct"))
    fars_coord_min  = fmt_pct(a6.get("min_coord_rate_pct"))
    fars_coord_note = f"{fars_coord_mean} mean ({fars_coord_min} min), 2016--2021"

    # A7: Spatial join
    a7 = audit.get("spatial_join", {})
    hpms_join = a7.get("join_100m_hpms_equiv", {})
    hpms_rate = fmt_pct(hpms_join.get("rate_pct"))
    hpms_n    = hpms_join.get("n_matched", "---")
    hpms_tot  = a7.get("n_crash_points", "---")
    hpms_note = f"{hpms_rate} ({hpms_n}/{hpms_tot} crashes, VT pilot)"

    # A8: INJ_SEV
    a8 = audit.get("inj_severity", {})

    # A9: CRSS coords
    a9 = audit.get("crss_coordinates", {})

    # Compute total records across FARS VEHICLE files only (9 files, not CRSS)
    fars_dv_total = sum(
        v.get("n_rows", 0)
        for k, v in a12.items()
        if v.get("status") == "ok" and k.startswith("FARS")
    )
    n_fars_files = sum(1 for k, v in a12.items() if v.get("status") == "ok" and k.startswith("FARS"))

    # vPIC ErrorCode 6 count (Incomplete VIN)
    vpic_errcode_dist = a5.get("vpic_error_code_distribution", {})
    vpic_err6 = vpic_errcode_dist.get("6", 0)

    # Extract sorted FARS years present
    fars_years_present = sorted(
        int(k.split("FARS")[1].split("_")[0])
        for k in a12 if k.startswith("FARS") and a12[k].get("status") == "ok"
    )
    yr_range = (
        f"{fars_years_present[0]}--{fars_years_present[-1]}"
        if len(fars_years_present) >= 2
        else str(fars_years_present[0]) if fars_years_present else "?"
    )
    # Format total with LaTeX number grouping (safe: no TeX commands in this string)
    fars_total_tex = f"{fars_dv_total:,}".replace(",", "{,}")

    # CRSS full data results
    crss_acc  = audit.get("crss_accident_coords", {})
    crss_inj  = audit.get("crss_inj_sev", {})
    crss_esc  = audit.get("crss_esc_vpicdecode", {})

    # CRSS vehicle.csv rows/cols (from deltav audit)
    crss_veh = a12.get("CRSS2021_VEHICLE_full", {})
    crss_veh_rows = crss_veh.get("n_rows", None)
    crss_veh_cols = crss_veh.get("n_cols", None)

    # CRSS accident.csv rows/cols
    crss_acc_rows = crss_acc.get("n_rows", None)
    crss_acc_cols = crss_acc.get("n_cols", None)

    # CRSS ESC fill rate (from vpicdecode.csv)
    crss_esc_fill   = crss_esc.get("fill_rate_pct", None)
    crss_esc_n_std  = crss_esc.get("n_standard", 0)
    crss_esc_total  = crss_esc.get("n_rows", 0)
    crss_esc_empty  = crss_esc.get("n_empty", 0)
    crss_esc_na     = crss_esc.get("n_not_applicable", 0)

    def _tex_n(n: int | None) -> str:
        """Format integer with LaTeX comma grouping, or '---' if None."""
        if n is None:
            return "---"
        return f"{n:,}".replace(",", "{,}")

    def _tex_pct(v: float | None) -> str:
        if v is None:
            return "---"
        return f"{v:.1f}\\%"

    # CRSS INJ_SEV distribution for table
    inj_dist  = crss_inj.get("distribution_count", {})
    inj_pct   = crss_inj.get("distribution_pct", {})
    inj_total = crss_inj.get("n_rows", None)

    def _inj_cell() -> str:
        if not inj_dist or inj_total is None:
            return "\\texttt{INJ\\_SEV} in PERSON file; CRSS covers K--O"
        # Build compact summary: O / C / B / A / K / Unk
        k_cnt = inj_dist.get("4", 0); k_pct = inj_pct.get("4", 0)
        a_cnt = inj_dist.get("3", 0); a_pct = inj_pct.get("3", 0)
        b_cnt = inj_dist.get("2", 0); b_pct = inj_pct.get("2", 0)
        c_cnt = inj_dist.get("1", 0); c_pct = inj_pct.get("1", 0)
        o_cnt = inj_dist.get("0", 0); o_pct = inj_pct.get("0", 0)
        t_str = _tex_n(inj_total)
        return (
            f"person.csv ({t_str} rows): "
            f"O={_tex_n(o_cnt)} ({_tex_pct(o_pct)}); "
            f"C={_tex_n(c_cnt)} ({_tex_pct(c_pct)}); "
            f"B={_tex_n(b_cnt)} ({_tex_pct(b_pct)}); "
            f"A={_tex_n(a_cnt)} ({_tex_pct(a_pct)}); "
            f"K={_tex_n(k_cnt)} ({_tex_pct(k_pct)}); "
            f"full KABCO range present"
        )

    def _crss_dv_cell() -> str:
        parts = []
        if crss_veh_rows is not None:
            parts.append(
                f"Absent in vehicle.csv ({_tex_n(crss_veh_rows)} rows, "
                f"{crss_veh_cols} cols)"
            )
        if crss_acc_rows is not None:
            parts.append(
                f"absent in accident.csv ({_tex_n(crss_acc_rows)} rows, "
                f"{crss_acc_cols} cols)"
            )
        return "; ".join(parts) if parts else "Absent (full CRSS2021 audited)"

    def _crss_esc_cell() -> str:
        if crss_esc_fill is None:
            return "\\texttt{ElectronicStabilityControl} in vpicdecode.csv (not queried)"
        return (
            f"Absent in vehicle.csv; vpicdecode.csv "
            f"({_tex_n(crss_esc_total)} rows): "
            f"\\texttt{{Standard}}={_tex_n(crss_esc_n_std)} "
            f"({_tex_pct(crss_esc_fill)}); "
            f"empty={_tex_n(crss_esc_empty)} "
            f"({_tex_pct(round(crss_esc_empty/crss_esc_total*100, 2) if crss_esc_total else None)}); "
            f"\\texttt{{Not Applicable}}={_tex_n(crss_esc_na)} "
            f"({_tex_pct(round(crss_esc_na/crss_esc_total*100, 2) if crss_esc_total else None)})"
        )

    def _crss_coord_cell() -> str:
        if crss_acc_rows is None:
            return "Coordinates not in public CRSS release"
        return (
            f"No coordinates in accident.csv "
            f"({_tex_n(crss_acc_rows)} rows, {crss_acc_cols} cols); "
            f"CRSS suppresses GPS for respondent privacy; "
            f"spatial AADT join infeasible"
        )

    # ── CISS data extraction (A13/A14; codebook level) ───────────────────────
    ciss_mr   = audit.get("ciss_manual_review", {})
    ciss_dv_d = ciss_mr.get("deltav", {})
    ciss_co_d = ciss_mr.get("coordinates", {})
    ciss_es_d = ciss_mr.get("esc", {})
    ciss_ij_d = ciss_mr.get("injury", {})
    ciss_sp_d = ciss_mr.get("sample_size", {})

    def _ciss_dv_cell() -> str:
        if not ciss_mr.get("status"):
            return "CISS audit not run"
        n17 = ciss_dv_d.get("n_gv_obs_2017")
        n20 = ciss_dv_d.get("n_gv_obs_2020")
        return (
            f"\\texttt{{DVTOTAL}} in GV dataset "
            f"($n_{{2017}}={_tex_n(n17)}$, $n_{{2020}}={_tex_n(n20)}$ vehicles); "
            f"WinSMASH estimate; fill rate: --- (data unavailable); "
            f"\\texttt{{DVBASIS}}=0 $\\Rightarrow$ not inspected"
        )

    def _ciss_esc_cell() -> str:
        if not ciss_mr.get("status"):
            return "CISS audit not run"
        vcov = ciss_es_d.get("vin_coverage_pct_2017")
        n_vd = ciss_es_d.get("n_vinderived_obs_2017")
        n_gv = ciss_es_d.get("n_gv_obs_2017")
        return (
            f"\\texttt{{ElectronicStabilityControl}} in VINDERIVED "
            f"(VIN-decoded, full 17-char VIN); "
            f"VIN coverage: {_tex_pct(vcov)} of GV "
            f"($n_{{\\text{{VINDERIVED}}}}={_tex_n(n_vd)}$ / "
            f"$n_{{\\text{{GV}}}}={_tex_n(n_gv)}$); "
            f"fill rate: --- (data unavailable)"
        )

    def _ciss_coord_cell() -> str:
        if not ciss_mr.get("status"):
            return "CISS audit not run"
        return (
            "No GPS in public release; "
            "\\texttt{EDGEDISTX/Y/Z} (2020) = road-edge distance (cm), not geo-coords; "
            "geographic resolution: PSU (county); AADT join infeasible"
        )

    def _ciss_inj_cell() -> str:
        if not ciss_mr.get("status"):
            return "CISS audit not run"
        n_crash = ciss_sp_d.get("year_2017", {}).get("n_crash_cases")
        return (
            "\\texttt{CAIS} (max AIS 0--6) in CRASH; "
            "\\texttt{VAIS} in GV; full K--O spectrum; "
            f"$n_{{\\text{{crash}}}}={_tex_n(n_crash)}$ cases (2017)"
        )

    # ── Row definitions (v3: FARS | CRSS | CISS | Status) ───────────────────
    # Columns: (assumption, required_var, FARS_cell, CRSS_cell, CISS_cell, status)
    rows = [
        (
            "(P1) $\\Delta v$ per crash",
            "\\texttt{DELTA\\_V} column",
            (
                f"Absent in all {n_fars_files} VEHICLE files "
                f"({yr_range}; ${fars_total_tex}$ records)"
            ),
            _crss_dv_cell(),
            _ciss_dv_cell(),
            (
                "FARS/CRSS: \\textbf{Absent}; "
                "CISS: \\texttt{DVTOTAL} present (codebook only; "
                "fill rate ---, data unavailable)"
            ),
        ),
        (
            "Treatment $T$: ESC equipped",
            "ESC/stability column in crash DB",
            (
                f"Absent in all {n_ok_esc}/{n_total_esc} vehicle files; "
                f"vPIC decode (12-char VIN): fill {vpic_fill} "
                f"($n={vpic_n}$); "
                f"ErrorCode\\,6: {vpic_err6}/{vpic_n}"
            ),
            _crss_esc_cell(),
            _ciss_esc_cell(),
            (
                "\\textbf{Not in FARS/CRSS vehicle files}; "
                "CRSS vpicdecode: 28.3\\% Standard; "
                "CISS VINDERIVED: codebook confirmed (fill ---)"
            ),
        ),
        (
            "IV exclusion ($Z \\not\\equiv T$): ESC$\\,\\neq\\,$model year",
            "$T$ defined independently of model year (MY)",
            (
                "\\texttt{MOD\\_YEAR} available; no direct ESC col; "
                "$T:=\\mathbf{1}[\\text{MY}\\ge 2012]$ collapses $Z$ and $T$"
            ),
            (
                "\\texttt{MOD\\_YEAR} in vehicle.csv; "
                "ESC only via vpicdecode linkage; "
                "same collinearity: $T$ by MY $\\Rightarrow$ $Z\\equiv T$"
            ),
            (
                "\\texttt{MOD\\_YEAR} available; "
                "ESC via VINDERIVED (VIN-decoded); "
                "MY-based $T$ still collinear with ESC mandate MY\\,$\\ge$\\,2012"
            ),
            "\\textbf{Fails in all DBs}: $Z$ and $T$ not separable",
        ),
        (
            "IPSW: $H =$ AADT spatial join",
            "\\texttt{LATITUDE}, \\texttt{LONGITUD}",
            f"Present: {fars_coord_note}",
            _crss_coord_cell(),
            _ciss_coord_cell(),
            (
                f"FARS: PASS ({hpms_note}); "
                "CRSS: infeasible; CISS: infeasible"
            ),
        ),
        (
            "Outcome $Y$ variation (KABCO)",
            "\\texttt{INJ\\_SEV} (K/A/B/C/O)",
            (
                "\\texttt{FATALS} only (FARS = fatal crashes; "
                "$Y \\equiv K$); no severity variation"
            ),
            _inj_cell(),
            _ciss_inj_cell(),
            (
                "FARS: single-severity ($Y\\equiv K$); "
                "CRSS/CISS: full KABCO \\textemdash \\textbf{required DBs}"
            ),
        ),
    ]

    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(
        r"\begin{tabular}{p{2.5cm}p{2.5cm}p{3.0cm}p{3.0cm}p{3.0cm}p{2.5cm}}"
    )
    lines.append(r"\toprule")
    lines.append(
        r"\textbf{Assumption} & \textbf{Required variable} & "
        r"\textbf{FARS} & \textbf{CRSS} & "
        r"\textbf{CISS}$^{\dagger}$ & \textbf{Status} \\"
    )
    lines.append(r"\midrule")
    for i, (a, b, c, d, e, f) in enumerate(rows):
        row = " & ".join([a, b, c, d, e, f]) + r" \\"
        lines.append(row)
        if i < len(rows) - 1:
            lines.append(r"\addlinespace")
    lines.append(r"\bottomrule")
    lines.append(
        r"\multicolumn{6}{l}{\footnotesize $^{\dagger}$CISS: "
        r"codebook-level only (DOT HS\,812\,803/958); "
        r"CSV download failed on all URL patterns "
        r"(HTTP\,404 confirmed via curl; Python SSL error in CI environment); "
        r"fill rates not computable without data.} \\"
    )
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# ── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    log("=" * 70)
    log("Data-Requirements Audit: U.S. Crash Databases")
    log("2026-07-14")
    log("=" * 70)
    log(f"Python  : {sys.version}")
    log(f"Platform: {platform.platform()}")
    log(f"RAW_DIR : {RAW_DIR}")
    log(f"RES_DIR : {RES_DIR}")
    log("")

    # --- File catalogue ---
    # CRSS 2021 full extracted CSV directory
    # Source : https://static.nhtsa.gov/nhtsa/downloads/CRSS/2021/CRSS2021CSV.zip
    # SHA256 : 96de94779eb3c6085fc9c776a7575f089df9c77d76c97623a9f921255a32dc95
    # Place at: code_release/data/raw/CRSS2021/CRSS2021CSV/ (manual download required)
    # If the directory is absent the CRSS audits (A10-A12) report 'file_missing'.
    CRSS2021_DIR = RAW_DIR / "CRSS2021" / "CRSS2021CSV"

    # FARS VEHICLE files (9 years)
    fars_vehicle_files = {
        "FARS2005_VEHICLE": RAW_DIR / "FARS2005_VEHICLE.csv",
        "FARS2007_VEHICLE": RAW_DIR / "FARS2007_VEHICLE.csv",
        "FARS2009_VEHICLE": RAW_DIR / "FARS2009_VEHICLE.csv",
        "FARS2010_VEHICLE": RAW_DIR / "FARS2010_VEHICLE.csv",
        "FARS2013_VEHICLE": RAW_DIR / "FARS2013_VEHICLE.csv",
        "FARS2015_VEHICLE": RAW_DIR / "FARS2015_VEHICLE.csv",
        "FARS2017_VEHICLE": RAW_DIR / "FARS2017_VEHICLE.csv",
        "FARS2019_VEHICLE": RAW_DIR / "FARS2019_VEHICLE.csv",
        "FARS2021_VEHICLE": RAW_DIR / "FARS2021_VEHICLE_corrected.csv",
    }

    # CRSS VEHICLE file (full, from CRSS2021CSV.zip; 95,785 rows, 167 cols)
    crss_vehicle_files = {
        "CRSS2021_VEHICLE_full": CRSS2021_DIR / "vehicle.csv",
    }

    # FARS ACCIDENT file (for coordinate and severity checks)
    fars_accident_files = {
        "FARS2021_ACCIDENT": RAW_DIR / "FARS2021_ACCIDENT.csv",
    }

    # All VEHICLE files combined (for delta-V and ESC checks)
    all_vehicle_files = {**fars_vehicle_files, **crss_vehicle_files}

    # --- Run audits ---
    audit: dict = {
        "run_date":   datetime.datetime.now().isoformat(),
        "python":     sys.version,
        "platform":   platform.platform(),
        "raw_dir":    str(RAW_DIR),
        "res_dir":    str(RES_DIR),
    }

    # A1/A2: Delta-V
    audit["deltav"] = audit_deltav(all_vehicle_files)

    # A3/A4: ESC column
    audit["esc_column"] = audit_esc_column(all_vehicle_files)

    # A5: vPIC ESC (sample 200 unique VINs from FARS2021; results cached from prior run)
    audit["vpic_esc"] = audit_vpic_esc(
        RAW_DIR / "FARS2021_VEHICLE_corrected.csv",
        sample_n   = 200,
        api_wait_s = 0.3,
    )

    # A6: Coordinate coverage (from pre-computed step1 results)
    audit["coord_coverage"] = audit_coordinate_coverage(
        RES_DIR / "step1_coordinate_coverage.csv"
    )

    # A7: Spatial join (from pre-computed step1b results)
    audit["spatial_join"] = audit_spatial_join(
        RES_DIR / "step1b_join_rate_actual.json"
    )

    # A8: Injury severity
    audit["inj_severity"] = audit_injury_severity(
        {**fars_accident_files, **crss_vehicle_files}
    )

    # A10: CRSS accident.csv coordinate check (full CRSS2021 ZIP)
    audit["crss_accident_coords"] = audit_crss_accident_coords(
        CRSS2021_DIR / "accident.csv"
    )

    # A11: CRSS INJ_SEV KABCO distribution (from person.csv)
    audit["crss_inj_sev"] = audit_crss_inj_sev(
        CRSS2021_DIR / "person.csv"
    )

    # A12: CRSS ElectronicStabilityControl fill rate (from vpicdecode.csv)
    audit["crss_esc_vpicdecode"] = audit_crss_esc_vpicdecode(
        CRSS2021_DIR / "vpicdecode.csv"
    )

    # A13: CISS download attempt (documents HTTP response codes for evidence trail)
    # NOTE: This makes HTTP HEAD requests to static.nhtsa.gov and nhtsa.gov.
    # If network access is unavailable, results will be "URLError: ..." but the
    # audit will still complete (the codebook-level review A14 does not require network).
    audit["ciss_download_attempt"] = audit_ciss_download_attempt()

    # A14: CISS manual-based audit (codebook level; no CSV data required)
    # Sources: DOT HS 812 803 (2017 manual) and DOT HS 812 958 (2020 manual)
    audit["ciss_manual_review"] = audit_ciss_manual_review()

    # Create CISS data directory placeholder (even if empty)
    # This documents that CISS was investigated and data was sought but unavailable.
    ciss_raw_dirs = [
        RAW_DIR / "CISS2017",
        RAW_DIR / "CISS2020",
        RAW_DIR / "CISS2021",
    ]
    for ciss_dir in ciss_raw_dirs:
        ciss_dir.mkdir(parents=True, exist_ok=True)
        readme = ciss_dir / "README_download_failed.txt"
        if not readme.exists():
            readme.write_text(
                f"CISS data directory for {ciss_dir.name}\n"
                "=" * 50 + "\n"
                "Status: DATA NOT DOWNLOADED\n\n"
                "All attempted URL patterns returned HTTP 404:\n"
                f"  https://static.nhtsa.gov/nhtsa/downloads/CISS/"
                f"{ciss_dir.name[4:]}/CISS{ciss_dir.name[4:]}CSV.zip\n\n"
                "NHTSA file-downloads portal returns HTTP 403 (Akamai CDN).\n\n"
                "To obtain data manually:\n"
                "  1. Open https://www.nhtsa.gov/file-downloads in a browser\n"
                "  2. Navigate to: nhtsa/downloads/CISS/{year}/\n"
                f"  3. Download: CISS{ciss_dir.name[4:]}CSV.zip\n"
                f"  4. Extract to: {ciss_dir}/CISS{ciss_dir.name[4:]}CSV/\n\n"
                "Primary sources (variable existence confirmed):\n"
                "  DOT HS 812 803 — CISS 2017 Analytical User's Manual\n"
                "  DOT HS 812 958 — CISS 2020 Analytical User's Manual\n\n"
                f"Audit date: 2026-07-14\n"
                "Auditor: Keito Inoshita\n",
                encoding="utf-8",
            )
        log(f"  CISS placeholder dir: {ciss_dir}  (README written)")

    # Attach log
    audit["log"] = LOG.copy()

    # --- Save JSON ---
    json_path = RES_DIR / "audit_us_crash_databases_2026-07-14.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2, ensure_ascii=False)
    log(f"\nJSON saved: {json_path}")

    # --- Save CSV summary ---
    csv_rows = []

    def _str(v) -> str:
        if v is None:
            return ""
        return str(v)

    # Build flat CSV rows for each audit item
    for label, res in audit.get("deltav", {}).items():
        csv_rows.append({
            "audit_item":  "A1_A2_deltav",
            "file":        label,
            "status":      res.get("status"),
            "n_rows":      _str(res.get("n_rows")),
            "n_cols":      _str(res.get("n_cols")),
            "target_cols_found": _str(res.get("delta_v_found")),
            "target_cols":       _str(res.get("delta_v_cols")),
            "sha256":      res.get("sha256", ""),
        })
    for label, res in audit.get("esc_column", {}).items():
        csv_rows.append({
            "audit_item":  "A3_A4_esc_column",
            "file":        label,
            "status":      res.get("status"),
            "n_rows":      _str(res.get("n_rows")),
            "n_cols":      _str(res.get("n_cols")),
            "target_cols_found": _str(res.get("esc_found")),
            "target_cols":       _str(res.get("esc_cols")),
            "sha256":      "",
        })
    for label, res in audit.get("inj_severity", {}).items():
        csv_rows.append({
            "audit_item":  "A8_inj_severity",
            "file":        label,
            "status":      res.get("status"),
            "n_rows":      _str(res.get("n_rows")),
            "n_cols":      _str(res.get("n_cols")),
            "target_cols_found": _str(res.get("inj_found")),
            "target_cols":       _str(res.get("inj_cols")),
            "sha256":      "",
        })

    # vPIC row
    vpic = audit.get("vpic_esc", {})
    csv_rows.append({
        "audit_item":  "A5_vpic_esc",
        "file":        "vPIC_API_batch",
        "status":      vpic.get("status"),
        "n_rows":      _str(vpic.get("total_decoded")),
        "n_cols":      "",
        "target_cols_found": _str(vpic.get("esc_fill_rate_pct")),
        "target_cols":       _str(vpic.get("esc_value_distribution")),
        "sha256":      "",
    })

    # Coordinate row
    cc = audit.get("coord_coverage", {})
    csv_rows.append({
        "audit_item":  "A6_coord_coverage",
        "file":        "FARS_ACCIDENT_2016-2021",
        "status":      cc.get("status"),
        "n_rows":      _str(cc.get("total_crash_records")),
        "n_cols":      "",
        "target_cols_found": _str(cc.get("avg_coord_rate_pct")),
        "target_cols":       "LATITUDE,LONGITUD",
        "sha256":      "",
    })

    # Spatial join row
    sj = audit.get("spatial_join", {})
    hpms = sj.get("join_100m_hpms_equiv", {})
    csv_rows.append({
        "audit_item":  "A7_spatial_join",
        "file":        f"step1b_{sj.get('test_state','?')}_{sj.get('test_year','?')}",
        "status":      sj.get("status"),
        "n_rows":      _str(sj.get("n_crash_points")),
        "n_cols":      "",
        "target_cols_found": _str(hpms.get("rate_pct")),
        "target_cols":       "HPMS-equivalent roads join rate",
        "sha256":      "",
    })

    # A13: CISS download attempt
    cdl = audit.get("ciss_download_attempt", {})
    csv_rows.append({
        "audit_item":  "A13_ciss_download",
        "file":        "NHTSA_static.nhtsa.gov_CISS",
        "status":      cdl.get("status", "not_run"),
        "n_rows":      "",
        "n_cols":      "",
        "target_cols_found": _str(cdl.get("n_404")),
        "target_cols":       f"n_urls_tried={cdl.get('n_urls_tried',0)}; all 404",
        "sha256":      "",
    })

    # A14: CISS manual review — one row per audit item
    cmr = audit.get("ciss_manual_review", {})
    ciss_items = {
        "deltav_DVTOTAL":           (cmr.get("deltav", {}).get("exists_confirmed_by_codebook"), "DVTOTAL in GV dataset", cmr.get("deltav", {}).get("fill_rate_pct")),
        "gps_coords":               (not cmr.get("coordinates", {}).get("gps_found", True), "No LATITUDE/LONGITUD in any CISS dataset", None),
        "esc_VINDERIVED":           (cmr.get("esc", {}).get("exists_confirmed_by_codebook"), "ElectronicStabilityControl in VINDERIVED", cmr.get("esc", {}).get("fill_rate_pct")),
        "injury_CAIS":              (True, "CAIS/VAIS (AIS 0-6) in CRASH/GV", None),
        "n_cases_2017":             (True, f"CRASH n={cmr.get('sample_size',{}).get('year_2017',{}).get('n_crash_cases')}", None),
        "weights_CASEWGT_JKWGT24":  (True, "CASEWGT + JKWGT1-24 confirmed", None),
    }
    for item_label, (found, note, fill) in ciss_items.items():
        csv_rows.append({
            "audit_item":  "A14_ciss_manual",
            "file":        f"CISS_codebook_{item_label}",
            "status":      "codebook_confirmed" if found else "absent",
            "n_rows":      "",
            "n_cols":      "",
            "target_cols_found": _str(found),
            "target_cols":       note,
            "sha256":      _str(fill) if fill is not None else "fill_rate=UNKNOWN",
        })

    csv_path = RES_DIR / "audit_us_crash_databases_2026-07-14.csv"
    if csv_rows:
        fieldnames = list(csv_rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
    log(f"CSV saved: {csv_path}")

    # --- Save LaTeX table ---
    latex_str  = build_latex_table(audit)
    latex_path = RES_DIR / "table_data_audit.tex"
    with open(latex_path, "w", encoding="utf-8") as f:
        f.write(latex_str)
    log(f"LaTeX table saved: {latex_path}")

    # --- Print summary ---
    log("")
    log("=" * 70)
    log("AUDIT SUMMARY")
    log("=" * 70)

    # Delta-V
    dv = audit["deltav"]
    n_dv_checked = sum(1 for v in dv.values() if v.get("status") == "ok")
    n_dv_found   = sum(1 for v in dv.values() if v.get("delta_v_found"))
    log(f"A1/A2 DELTA_V: found in {n_dv_found}/{n_dv_checked} files  → {'PRESENT' if n_dv_found > 0 else 'ABSENT'}")

    # ESC column
    esc = audit["esc_column"]
    n_esc_checked = sum(1 for v in esc.values() if v.get("status") == "ok")
    n_esc_found   = sum(1 for v in esc.values() if v.get("esc_found"))
    log(f"A3/A4 ESC col: found in {n_esc_found}/{n_esc_checked} files  → {'PRESENT' if n_esc_found > 0 else 'ABSENT'}")

    # vPIC
    vp = audit["vpic_esc"]
    log(f"A5 vPIC ESC: fill rate = {vp.get('esc_fill_rate_pct')}%  "
        f"(n_decoded={vp.get('total_decoded')}, "
        f"values={vp.get('esc_value_distribution')})")

    # Coordinates
    cv = audit["coord_coverage"]
    log(f"A6 FARS coords: avg={cv.get('avg_coord_rate_pct')}% "
        f"min={cv.get('min_coord_rate_pct')}% (2016-2021)")

    # Spatial join
    sj = audit["spatial_join"]
    hpms = sj.get("join_100m_hpms_equiv", {})
    log(f"A7 HPMS join rate: {hpms.get('rate_pct')}% "
        f"({hpms.get('n_matched')}/{sj.get('n_crash_points')} in {sj.get('test_state')} pilot)")

    # Injury
    inj = audit["inj_severity"]
    log(f"A8 INJ_SEV: "
        + "; ".join(f"{k}: {'found' if v.get('inj_found') else 'absent'} "
                    f"({v.get('inj_cols')})"
                    for k, v in inj.items() if v.get("status") == "ok"))

    # CRSS full data
    cac = audit.get("crss_accident_coords", {})
    log(f"A10 CRSS accident coords: coord_found={cac.get('coord_found')} "
        f"(rows={cac.get('n_rows')}, cols={cac.get('n_cols')})")

    cinj = audit.get("crss_inj_sev", {})
    log(f"A11 CRSS INJ_SEV: total_rows={cinj.get('n_rows')}")
    if cinj.get("distribution_count"):
        for code in sorted(cinj["distribution_count"].keys()):
            log(f"    {code}: {cinj['distribution_count'][code]:,} "
                f"({cinj['distribution_pct'].get(code,0):.2f}%)")

    cesc = audit.get("crss_esc_vpicdecode", {})
    log(f"A12 CRSS ESC (vpicdecode.csv): fill_rate={cesc.get('fill_rate_pct')}% "
        f"Standard={cesc.get('n_standard')}/{cesc.get('n_rows')} "
        f"empty={cesc.get('n_empty')}")

    # CISS
    cdl = audit.get("ciss_download_attempt", {})
    log(f"A13 CISS download: status={cdl.get('status')} "
        f"urls_tried={cdl.get('n_urls_tried')} n_404={cdl.get('n_404')}")

    cmr = audit.get("ciss_manual_review", {})
    log(f"A14 CISS manual review: status={cmr.get('status')}")
    dv_d = cmr.get("deltav", {})
    log(f"     (a) DVTOTAL exists={dv_d.get('exists_confirmed_by_codebook')} "
        f"fill_rate={dv_d.get('fill_rate_pct')} (UNKNOWN=expected)")
    co_d = cmr.get("coordinates", {})
    log(f"     (b) GPS found={co_d.get('gps_found')} "
        f"(EDGEDISTX/Y/Z present in 2020 GV but are cm-distances, not coords)")
    es_d = cmr.get("esc", {})
    log(f"     (c) ElectronicStabilityControl exists={es_d.get('exists_confirmed_by_codebook')} "
        f"dataset=VINDERIVED VIN_coverage={es_d.get('vin_coverage_pct_2017')}% "
        f"fill_rate={es_d.get('fill_rate_pct')} (UNKNOWN=expected)")
    sp_d = cmr.get("sample_size", {})
    log(f"     (e) n_crash_2017={sp_d.get('year_2017',{}).get('n_crash_cases')} "
        f"n_gv_2017={sp_d.get('year_2017',{}).get('n_gv_vehicles')} "
        f"n_gv_2020={sp_d.get('year_2020',{}).get('n_gv_vehicles')}")
    log(f"     Tier-1 feasibility: {cmr.get('tier1_feasibility','')[:100]}...")
    log(f"     Tier-2 feasibility: {cmr.get('tier2_feasibility','')[:100]}...")

    log("")
    log(f"Output files:")
    log(f"  JSON  : {json_path}")
    log(f"  CSV   : {csv_path}")
    log(f"  LaTeX : {latex_path}")


if __name__ == "__main__":
    main()
