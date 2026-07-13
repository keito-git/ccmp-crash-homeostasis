#!/usr/bin/env python3
"""
Step 1: FARS coordinate coverage check (2016-2021).
Assesses LAT/LON field completeness and feasibility of spatial join with HPMS/AADT data.
Go criterion: join rate >= 70%.
"""

import os
import sys
import requests
import zipfile
import io
import json
import hashlib
import platform
import datetime
import time
import traceback
from pathlib import Path

import pandas as pd
import numpy as np

# Paths
PILOT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR   = PILOT_DIR / "data" / "raw"
PROC_DIR  = PILOT_DIR / "data" / "processed"
RES_DIR   = PILOT_DIR / "results"
for d in [RAW_DIR, PROC_DIR, RES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

LOG: list[str] = []

def log(msg: str) -> None:
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    LOG.append(line)

# NHTSA FARS download helper
FARS_BASE = "https://static.nhtsa.gov/nhtsa/downloads/FARS/{year}/National/FARS{year}NationalCSV.zip"

def download_fars_accident(year: int, timeout: int = 120) -> pd.DataFrame | None:
    """Download FARS national accident file for a given year.
    Returns DataFrame or None if download fails.
    """
    url = FARS_BASE.format(year=year)
    cache_path = RAW_DIR / f"FARS{year}_ACCIDENT.csv"

    if cache_path.exists():
        log(f"  [cache] {cache_path.name}")
        try:
            df = pd.read_csv(cache_path, encoding="latin-1", low_memory=False)
            return df
        except Exception as e:
            log(f"  [warn] cache read failed: {e}")

    log(f"  Downloading FARS {year} from {url}")
    try:
        resp = requests.get(url, timeout=timeout, stream=True)
        resp.raise_for_status()
        content = resp.content
        log(f"  Downloaded {len(content)/1e6:.1f} MB")

        # Compute hash for manifest
        sha256 = hashlib.sha256(content).hexdigest()[:16]
        log(f"  sha256[:16]={sha256}")

        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            names = zf.namelist()
            log(f"  ZIP contents: {names[:10]}")
            # Find ACCIDENT file (case-insensitive)
            accident_file = next(
                (n for n in names if "ACCIDENT" in n.upper() and n.upper().endswith(".CSV")),
                None
            )
            if accident_file is None:
                log(f"  [ERROR] No ACCIDENT.CSV found in ZIP for {year}")
                return None

            with zf.open(accident_file) as f:
                df = pd.read_csv(f, encoding="latin-1", low_memory=False)
                log(f"  Loaded {len(df)} records, {len(df.columns)} columns")

                # Cache for reuse
                df.to_csv(cache_path, index=False)
                log(f"  Cached to {cache_path.name}")
                return df

    except requests.exceptions.HTTPError as e:
        log(f"  [ERROR] HTTP {e.response.status_code} for {url}")
        return None
    except requests.exceptions.ConnectionError as e:
        log(f"  [ERROR] ConnectionError: {e}")
        return None
    except requests.exceptions.Timeout:
        log(f"  [ERROR] Timeout after {timeout}s for {url}")
        return None
    except Exception as e:
        log(f"  [ERROR] Unexpected: {e}")
        traceback.print_exc()
        return None

# Coordinate field detection
# FARS field names for coordinates vary by year:
# LATITUDE / LONGITUD  (modern format, since ~2009)
# LAT / LON            (older alternative)
LAT_CANDIDATES = ["LATITUDE", "LAT", "LATITUDEX", "CRASH_LAT"]
LON_CANDIDATES = ["LONGITUD", "LON", "LONGITUDE", "CRASH_LON"]

SPECIAL_MISSING = {0, 77.7777, 77.777, 88.8888, 88.888, 99.9999, 99.999}  # FARS coded unknowns

def detect_coord_fields(df: pd.DataFrame) -> tuple[str | None, str | None]:
    """Return (lat_col, lon_col) detected from DataFrame columns."""
    lat_col = next((c for c in LAT_CANDIDATES if c in df.columns), None)
    lon_col = next((c for c in LON_CANDIDATES if c in df.columns), None)
    return lat_col, lon_col

def coord_coverage(df: pd.DataFrame, lat_col: str, lon_col: str) -> dict:
    """
    Compute valid coordinate rate.
    Valid = not null, not in SPECIAL_MISSING, lat in [-90,90], lon in [-180,0] (CONUS)
    """
    n = len(df)
    lat = pd.to_numeric(df[lat_col], errors="coerce")
    lon = pd.to_numeric(df[lon_col], errors="coerce")

    not_null    = lat.notna() & lon.notna()
    not_special = ~lat.isin(SPECIAL_MISSING) & ~lon.isin(SPECIAL_MISSING)
    in_range    = (lat >= 18) & (lat <= 72) & (lon >= -180) & (lon <= -60)  # US + territories

    valid = not_null & not_special & in_range
    n_valid = int(valid.sum())
    n_null  = int((~not_null).sum())
    n_special = int((not_null & ~not_special).sum())
    n_range_fail = int((not_null & not_special & ~in_range).sum())

    return {
        "total": n,
        "valid_coords": n_valid,
        "coord_rate_pct": round(n_valid / n * 100, 2) if n > 0 else 0.0,
        "null_count": n_null,
        "special_missing_count": n_special,
        "out_of_range_count": n_range_fail,
        "lat_col": lat_col,
        "lon_col": lon_col,
    }

# Main
def main() -> None:
    log("=" * 60)
    log("Step 1: FARS Coordinate Coverage Check")
    log("=" * 60)
    log(f"Python {sys.version}")
    log(f"Platform: {platform.platform()}")
    log(f"pandas {pd.__version__}, numpy {np.__version__}")

    # Years to check
    years = [2016, 2017, 2018, 2019, 2020, 2021]

    results = []

    for year in years:
        log(f"\n--- FARS {year} ---")
        df = download_fars_accident(year)
        if df is None:
            log(f"  SKIPPED: download failed for {year}")
            results.append({
                "year": year,
                "status": "download_failed",
                "total": None,
                "valid_coords": None,
                "coord_rate_pct": None,
                "lat_col": None,
                "lon_col": None,
            })
            continue

        log(f"  Columns: {list(df.columns[:20])}...")
        lat_col, lon_col = detect_coord_fields(df)

        if lat_col is None or lon_col is None:
            log(f"  [WARN] Coordinate columns not found. Available: {list(df.columns)}")
            results.append({
                "year": year,
                "status": "no_coord_columns",
                "total": len(df),
                "valid_coords": 0,
                "coord_rate_pct": 0.0,
                "lat_col": None,
                "lon_col": None,
            })
            continue

        cov = coord_coverage(df, lat_col, lon_col)
        log(f"  lat_col={lat_col}, lon_col={lon_col}")
        log(f"  Total crashes: {cov['total']:,}")
        log(f"  Valid coords:  {cov['valid_coords']:,}")
        log(f"  Coverage:      {cov['coord_rate_pct']:.2f}%")
        log(f"  Null:          {cov['null_count']:,}")
        log(f"  Coded missing: {cov['special_missing_count']:,}")
        log(f"  Out of range:  {cov['out_of_range_count']:,}")

        results.append({
            "year": year,
            "status": "ok",
            **cov,
        })

        # Sample a few rows for sanity check
        valid_mask = (
            pd.to_numeric(df[lat_col], errors="coerce").between(18, 72) &
            pd.to_numeric(df[lon_col], errors="coerce").between(-180, -60)
        )
        sample = df[valid_mask][[lat_col, lon_col]].head(3)
        log(f"  Sample valid coords:\n{sample.to_string()}")

    # Save results
    res_df = pd.DataFrame(results)
    out_path = RES_DIR / "step1_coordinate_coverage.csv"
    res_df.to_csv(out_path, index=False)
    log(f"\nResults saved to {out_path}")

    # Summary
    log("\n" + "=" * 60)
    log("SUMMARY: Step 1 Coordinate Coverage")
    log("=" * 60)
    ok = res_df[res_df["status"] == "ok"]
    if len(ok) > 0:
        avg_rate = ok["coord_rate_pct"].mean()
        min_rate = ok["coord_rate_pct"].min()
        log(f"Years with data: {list(ok['year'])}")
        log(f"Average coord coverage: {avg_rate:.2f}%")
        log(f"Min coord coverage:     {min_rate:.2f}%")
        log(f"Go criterion (>= 70%): {'PASS' if min_rate >= 70 else 'MARGINAL/FAIL'}")

        # Total records with valid coords for spatial join
        total_valid = ok["valid_coords"].sum()
        log(f"Total crash records with valid coords (all years): {total_valid:,}")
    else:
        log("No successful downloads. Cannot assess coordinate coverage.")

    # Save manifest
    manifest = {
        "script": "step1_fars_coord_check.py",
        "run_date": datetime.datetime.now().isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "years_checked": years,
        "log": LOG,
    }
    with open(RES_DIR / "step1_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    log(f"Manifest saved to {RES_DIR / 'step1_manifest.json'}")


if __name__ == "__main__":
    main()
