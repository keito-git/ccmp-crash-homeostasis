#!/usr/bin/env python3
"""
Step 2: Delta-v availability check in FARS/CRSS across pre-ESC (2005-2010) and post-ESC (2013-2019) windows.
Go criterion: each window has >= 1000 valid DELTA_V records.
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

# NHTSA download helpers
FARS_BASE = "https://static.nhtsa.gov/nhtsa/downloads/FARS/{year}/National/FARS{year}NationalCSV.zip"
CRSS_BASE = "https://static.nhtsa.gov/nhtsa/downloads/CRSS/{year}/CRSS{year}CSV.zip"

# Fields potentially related to velocity / crash severity in various NHTSA datasets
# Note: DELTA_V is the target field. Also check TRAV_SP, VSPD_LIM, etc. as proxies.
DV_CANDIDATES = [
    "DELTA_V",     # Primary target: change in velocity
    "PDOF",        # Principal direction of force (related to crash dynamics)
    "TRAV_SP",     # Travel speed at time of crash
    "VSPD_LIM",    # Posted speed limit
    "SPEEDREL",    # Speed-related crash factor
    "DELTA_V_DIR", # Direction of delta-v
    "TOT_SF",      # Total safety failures
    "IMPACT1",     # Initial impact point
    "DEFORMED",    # Vehicle deformation
    "FIRE_EXP",    # Fire/explosion
    "MCARR_ID",    # Motor carrier ID (not relevant but check for unexpected matches)
]


def download_nhtsa_vehicle(source: str, year: int, timeout: int = 180) -> pd.DataFrame | None:
    """Download NHTSA vehicle file (FARS or CRSS) for a given year.
    source: 'FARS' or 'CRSS'
    """
    if source == "FARS":
        url = FARS_BASE.format(year=year)
        vehicle_keyword = "VEHICLE"
        cache_name = f"FARS{year}_VEHICLE.csv"
    elif source == "CRSS":
        url = CRSS_BASE.format(year=year)
        vehicle_keyword = "VEHICLE"
        cache_name = f"CRSS{year}_VEHICLE.csv"
    else:
        raise ValueError(f"Unknown source: {source}")

    cache_path = RAW_DIR / cache_name
    if cache_path.exists():
        log(f"  [cache] {cache_name}")
        try:
            df = pd.read_csv(cache_path, encoding="latin-1", low_memory=False)
            log(f"  Loaded {len(df):,} records from cache")
            return df
        except Exception as e:
            log(f"  [warn] cache read failed: {e}")

    log(f"  Downloading {source} {year} from {url}")
    try:
        resp = requests.get(url, timeout=timeout, stream=True)
        resp.raise_for_status()
        content = resp.content
        log(f"  Downloaded {len(content)/1e6:.1f} MB (sha256[:16]={hashlib.sha256(content).hexdigest()[:16]})")

        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            names = zf.namelist()
            log(f"  ZIP contains {len(names)} files: {names[:8]}")

            # Find vehicle file
            vehicle_file = next(
                (n for n in names
                 if vehicle_keyword in n.upper() and n.upper().endswith(".CSV")),
                None
            )
            if vehicle_file is None:
                log(f"  [ERROR] No VEHICLE.CSV in ZIP. Files: {names}")
                return None

            log(f"  Reading: {vehicle_file}")
            with zf.open(vehicle_file) as f:
                df = pd.read_csv(f, encoding="latin-1", low_memory=False)
            log(f"  Loaded {len(df):,} records, {len(df.columns)} columns")
            df.to_csv(cache_path, index=False)
            log(f"  Cached to {cache_name}")
            return df

    except requests.exceptions.HTTPError as e:
        log(f"  [ERROR] HTTP {e.response.status_code}: {url}")
        return None
    except requests.exceptions.ConnectionError as e:
        log(f"  [ERROR] ConnectionError: {e}")
        return None
    except requests.exceptions.Timeout:
        log(f"  [ERROR] Timeout after {timeout}s")
        return None
    except Exception as e:
        log(f"  [ERROR] Unexpected: {e}")
        traceback.print_exc()
        return None


def check_deltav_in_df(df: pd.DataFrame, source: str, year: int) -> dict:
    """Check for DELTA_V field and related velocity fields in DataFrame."""
    cols = [c.upper() for c in df.columns]
    col_map = {c.upper(): c for c in df.columns}

    found_fields = {}
    for candidate in DV_CANDIDATES:
        if candidate in cols:
            actual_col = col_map[candidate]
            series = pd.to_numeric(df[actual_col], errors="coerce")
            n_total = len(series)
            n_valid = int(series.notna().sum())
            n_nonzero = int((series != 0).sum())
            n_unknown_coded = int(series.isin([99, 999, 9999, 99.9, 999.9]).sum())

            found_fields[candidate] = {
                "col_name": actual_col,
                "n_total": n_total,
                "n_valid_nonnull": n_valid,
                "n_nonzero": n_nonzero,
                "n_coded_unknown": n_unknown_coded,
                "valid_pct": round(n_valid / n_total * 100, 2) if n_total > 0 else 0.0,
                "mean": round(float(series.mean()), 3) if n_valid > 0 else None,
                "median": round(float(series.median()), 3) if n_valid > 0 else None,
                "p25": round(float(series.quantile(0.25)), 3) if n_valid > 0 else None,
                "p75": round(float(series.quantile(0.75)), 3) if n_valid > 0 else None,
            }

    # Check for any velocity-like field we might have missed
    vel_pattern_cols = [c for c in df.columns if any(
        kw in c.upper() for kw in ["VEL", "SPEED", "SPD", "DELTA", "DELT", "DV", "VELO"]
    )]
    if vel_pattern_cols:
        log(f"  Velocity-pattern columns found: {vel_pattern_cols}")

    return {
        "source": source,
        "year": year,
        "n_records": len(df),
        "all_columns": list(df.columns),
        "velocity_pattern_cols": vel_pattern_cols,
        "found_fields": found_fields,
    }


def main() -> None:
    log("=" * 60)
    log("Step 2: Delta-v Availability Check (FARS + CRSS)")
    log("=" * 60)
    log(f"Python {sys.version}")
    log(f"ESC mandate: FMVSS 126 effective 2011-09-01")
    log(f"Pre-ESC window: 2005-2010; Post-ESC window: 2013-2019")
    log(f"Go criterion: pre-ESC >= 1000 AND post-ESC >= 1000 valid DELTA_V records")
    log("")

    # FARS: check select years pre and post ESC
    # Pre-ESC: 2005, 2007, 2009, 2010
    # Post-ESC: 2013, 2015, 2017, 2019
    # Plus 2021 as most recent
    fars_years = [2005, 2007, 2009, 2010, 2013, 2015, 2017, 2019, 2021]
    crss_years = [2016, 2017, 2019, 2021]  # CRSS started 2016

    all_results = []
    delta_v_summary = {"pre_esc": {}, "post_esc": {}}

    log("\n=== FARS Vehicle File Check ===")
    for year in fars_years:
        log(f"\n--- FARS VEHICLE {year} ---")
        df = download_nhtsa_vehicle("FARS", year)
        if df is None:
            log(f"  SKIPPED: download failed")
            all_results.append({"source": "FARS", "year": year, "status": "download_failed"})
            continue

        result = check_deltav_in_df(df, "FARS", year)
        result["status"] = "ok"

        # Summary of DELTA_V specifically
        if "DELTA_V" in result["found_fields"]:
            dv = result["found_fields"]["DELTA_V"]
            log(f"  DELTA_V found: n_valid={dv['n_valid_nonnull']:,} / {dv['n_total']:,} ({dv['valid_pct']:.1f}%)")
            log(f"  DELTA_V stats: mean={dv['mean']}, median={dv['median']}, p25={dv['p25']}, p75={dv['p75']}")
        else:
            log(f"  DELTA_V NOT found in FARS VEHICLE {year}")
            log(f"  Available columns (first 30): {list(df.columns[:30])}")

        # Log TRAV_SP (travel speed) as potential proxy
        if "TRAV_SP" in result["found_fields"]:
            ts = result["found_fields"]["TRAV_SP"]
            log(f"  TRAV_SP (travel speed proxy): n_valid={ts['n_valid_nonnull']:,}, valid_pct={ts['valid_pct']:.1f}%")

        if "VSPD_LIM" in result["found_fields"]:
            vsl = result["found_fields"]["VSPD_LIM"]
            log(f"  VSPD_LIM (speed limit): n_valid={vsl['n_valid_nonnull']:,}, valid_pct={vsl['valid_pct']:.1f}%")

        all_results.append(result)

        # Accumulate for pre/post ESC summary
        window = "pre_esc" if year <= 2010 else "post_esc"
        if year not in (2011, 2012):  # Skip transition years
            dv_count = (
                result["found_fields"]["DELTA_V"]["n_valid_nonnull"]
                if "DELTA_V" in result["found_fields"]
                else 0
            )
            delta_v_summary[window][year] = dv_count

    # CRSS: check for DELTA_V (CRSS likely has it)
    log("\n=== CRSS Vehicle File Check ===")
    for year in crss_years:
        log(f"\n--- CRSS VEHICLE {year} ---")
        df = download_nhtsa_vehicle("CRSS", year)
        if df is None:
            log(f"  SKIPPED: download failed")
            all_results.append({"source": "CRSS", "year": year, "status": "download_failed"})
            continue

        result = check_deltav_in_df(df, "CRSS", year)
        result["status"] = "ok"

        if "DELTA_V" in result["found_fields"]:
            dv = result["found_fields"]["DELTA_V"]
            log(f"  CRSS DELTA_V: n_valid={dv['n_valid_nonnull']:,} / {dv['n_total']:,} ({dv['valid_pct']:.1f}%)")
        else:
            log(f"  DELTA_V NOT found in CRSS {year}")
            log(f"  Columns (first 30): {list(df.columns[:30])}")

        all_results.append(result)

    # Save results
    # Flatten for CSV
    flat_rows = []
    for r in all_results:
        if r.get("status") != "ok":
            flat_rows.append({
                "source": r["source"],
                "year": r["year"],
                "status": r.get("status", "unknown"),
                "n_records": None,
                "delta_v_found": False,
                "delta_v_valid": None,
                "delta_v_valid_pct": None,
                "trav_sp_valid": None,
                "trav_sp_valid_pct": None,
                "vspd_lim_valid": None,
                "vel_pattern_cols": None,
            })
        else:
            dv = r["found_fields"].get("DELTA_V", {})
            ts = r["found_fields"].get("TRAV_SP", {})
            vsl = r["found_fields"].get("VSPD_LIM", {})
            flat_rows.append({
                "source": r["source"],
                "year": r["year"],
                "status": "ok",
                "n_records": r["n_records"],
                "delta_v_found": bool(dv),
                "delta_v_valid": dv.get("n_valid_nonnull"),
                "delta_v_valid_pct": dv.get("valid_pct"),
                "delta_v_mean": dv.get("mean"),
                "delta_v_median": dv.get("median"),
                "trav_sp_valid": ts.get("n_valid_nonnull"),
                "trav_sp_valid_pct": ts.get("valid_pct"),
                "vspd_lim_valid": vsl.get("n_valid_nonnull"),
                "vel_pattern_cols": str(r.get("velocity_pattern_cols", [])),
            })

    df_out = pd.DataFrame(flat_rows)
    out_path = RES_DIR / "step2_deltav_availability.csv"
    df_out.to_csv(out_path, index=False)
    log(f"\nResults saved to {out_path}")

    # Go/No-Go Assessment
    log("\n" + "=" * 60)
    log("Go/No-Go Assessment: Step 2 Delta-v")
    log("=" * 60)

    ok_df = df_out[df_out["status"] == "ok"] if "ok" in df_out["status"].values else pd.DataFrame()

    # Count pre/post ESC DELTA_V records (FARS or CRSS combined)
    # Pre-ESC: <= 2010
    # Post-ESC: >= 2013
    def sum_deltav(source_filter=None, year_filter=None) -> int:
        mask = df_out["status"] == "ok"
        if source_filter:
            mask &= df_out["source"] == source_filter
        if year_filter:
            mask &= df_out["year"].isin(year_filter)
        sub = df_out[mask]
        total = sub["delta_v_valid"].fillna(0).astype(int).sum()
        return int(total)

    pre_years = [y for y in [2005, 2007, 2009, 2010] if y in df_out["year"].values]
    post_years = [y for y in [2013, 2015, 2017, 2019, 2021] if y in df_out["year"].values]

    pre_dv_fars  = sum_deltav("FARS", pre_years)
    post_dv_fars = sum_deltav("FARS", post_years)
    pre_dv_crss  = 0  # CRSS started 2016 (post-ESC only)
    post_dv_crss = sum_deltav("CRSS", post_years)

    log(f"\nFARS DELTA_V records:")
    log(f"  Pre-ESC  (years {pre_years}):  {pre_dv_fars:,}")
    log(f"  Post-ESC (years {post_years}): {post_dv_fars:,}")
    log(f"\nCRSS DELTA_V records (post-ESC only, 2016+):")
    log(f"  Post-ESC: {post_dv_crss:,}")

    log(f"\nCombined (FARS+CRSS):")
    log(f"  Pre-ESC:  {pre_dv_fars:,}")
    log(f"  Post-ESC: {post_dv_fars + post_dv_crss:,}")

    go_criterion_pre  = pre_dv_fars >= 1000
    go_criterion_post = (post_dv_fars + post_dv_crss) >= 1000

    log(f"\nGo criterion (pre-ESC >= 1000):  {'PASS' if go_criterion_pre else 'FAIL'} ({pre_dv_fars:,})")
    log(f"Go criterion (post-ESC >= 1000): {'PASS' if go_criterion_post else 'FAIL'} ({post_dv_fars + post_dv_crss:,})")

    if go_criterion_pre and go_criterion_post:
        log("\nStep 2 verdict: GO")
    else:
        log("\nStep 2 verdict: NO-GO (or CONDITIONAL)")
        log("Fallback: use TRAV_SP (travel speed) as Delta-v proxy for Theorem 2 partial identification only")
        log("Implication: Theorem 1 (physics-injected point identification) cannot be illustrated with FARS data")
        log("Action: Rely on STATS19 speed_limit variable as Delta-v proxy; semi-synthetic SCM as primary validation")

    # Save full details for audit
    # Save column lists for each year (for data dictionary verification)
    col_info = {}
    for r in all_results:
        if r.get("status") == "ok":
            key = f"{r['source']}_{r['year']}"
            col_info[key] = {
                "all_columns": r.get("all_columns", []),
                "velocity_pattern_cols": r.get("velocity_pattern_cols", []),
                "found_fields_summary": {
                    k: {kk: vv for kk, vv in v.items() if kk != "col_name"}
                    for k, v in r.get("found_fields", {}).items()
                }
            }

    with open(RES_DIR / "step2_column_details.json", "w") as f:
        json.dump(col_info, f, indent=2, ensure_ascii=False)

    manifest = {
        "script": "step2_deltav_check.py",
        "run_date": datetime.datetime.now().isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "far_years_checked": fars_years,
        "crss_years_checked": crss_years,
        "go_criterion_pre_esc": go_criterion_pre,
        "go_criterion_post_esc": go_criterion_post,
        "pre_esc_deltav_count_fars": pre_dv_fars,
        "post_esc_deltav_count_fars": post_dv_fars,
        "post_esc_deltav_count_crss": post_dv_crss,
        "log": LOG,
    }
    with open(RES_DIR / "step2_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    log(f"\nManifest saved.")
    log(f"Column details saved to {RES_DIR / 'step2_column_details.json'}")


if __name__ == "__main__":
    main()
