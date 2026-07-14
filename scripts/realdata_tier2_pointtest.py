"""
Real-Data Tier-2 Point Test: CRSS Pooled Analysis (2016-2021)
=============================================================
Risk Homeostasis / CCMP study
Script: realdata_tier2_pointtest.py (NEW file -- existing scripts not modified)
Authors: Keito Inoshita, Akira Kawai
Date  : 2026-07-14
Treatment source: the NHTSA 5-Star Safety Ratings API is used as the primary source of T.

PURPOSE
-------
Execute the Tier-2 point test (H0: tau = 0, i.e., complete risk homeostasis kappa_LATE = 1)
on real crash data (CRSS 2016-2021), applying the IV Wald estimator from Algorithm 1 /
Theorem 3 of the paper.

ANALYSIS PRE-SPECIFICATION (frozen before first execution, 2026-07-14)
----------------------------------------------------------------------
Z definition (primary):
  Z_binary:   Z=0 for MY in [2000, 2010] (pre-mandate)
              Z=1 for MY >= 2012 (fully mandated, 100% after FMVSS 126)
              MY 2011 excluded (transition, 95% phase-in per Sivinski 2011)
              MY < 2000 excluded (before ESC API coverage)
  Z_wide:     Z=0 for MY in [2000, 2009] (more conservative pre-mandate)
              Z=1 for MY >= 2012 (same)
              Sensitivity comparison only

T definition (NHTSA 5-Star Safety Ratings API, primary source):
  For MY in [2000, 2010]:
    - Query api.nhtsa.gov/SafetyRatings for (make, model, year)
    - If all configurations report "Standard" -> T=1 (early adopter, always-taker)
    - If all configurations report "No"       -> T=0 (no ESC, potential complier)
    - If any configuration reports "Optional", OR if configs disagree -> T=unassigned
    - If make/model not in NHTSA API          -> T=unassigned (coverage gap)
  For MY >= 2012:
    - T=1 (100% mandated by FMVSS 126)
  Validation: cross-check against Dang 2007 / Sivinski 2011 list (esc_equipment_list.csv)

Y definition (primary):
  Y_binary = I(INJ_SEV in {1,2,3,4}) = I(any injury)
  INJ_SEV 5 (unknown severity) and 9 (not reported) excluded
  INJ_SEV 0 = O (no apparent injury) -> Y=0

Y_numeric (secondary):
  Y_num = INJ_SEV (0,1,2,3,4) -- treated as continuous for robustness check
  Same exclusions as Y_binary

IPSW construction:
  w = WEIGHT (CRSS person-level design weight)
  No collider IPSW applied (GPS coordinates not available in CRSS for HPMS match)
  Sensitivity: stratum-level IPSW using URBANICITY x INT_HWY x REGION crash-rate proxy
  (see compute_stratum_ipsw() for details)

Estimator (identical to hc_t2_reduction.py _ipsw_wald function):
  tau_hat = (E_w[Y|Z=1] - E_w[Y|Z=0]) / (E_w[T|Z=1] - E_w[T|Z=0])
  SE: Hajek sandwich variance (delta method)
  CI: 95% two-sided normal approximation
  Test: H0: tau = 0, two-sided z-test at alpha=0.05

Bootstrap CI:
  B=500 iterations, stratified cluster bootstrap (cluster by CASENUM within PSUSTRAT)
  Pooled year: composite PSU = CRSS_YEAR * 10000 + PSU

FABRICATION PROHIBITION:
  Results in this file are only from executing the code below.
  No numbers have been inserted manually. All outputs are from actual code execution.

GPU: none (CPU-only)
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import time
import ssl
import urllib.parse
import urllib.request
import urllib.error
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.stats

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR   = Path(__file__).resolve().parents[1]
DATA_DIR   = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
SCRIPTS_DIR = BASE_DIR / "scripts"

CACHE_DIR  = DATA_DIR / "nhtsa_safetyratings_cache"
CACHE_VEHICLES_DIR = CACHE_DIR / "vehicles"
CACHE_INDEX_DIR    = CACHE_DIR / "index"
ESC_LIST_CSV = DATA_DIR / "esc_equipment_list.csv"         # Dang/Sivinski list (validation)
ESC_API_CSV  = DATA_DIR / "esc_equipment_list_nhtsa_api.csv"  # API-derived list (primary)

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_VEHICLES_DIR.mkdir(parents=True, exist_ok=True)
CACHE_INDEX_DIR.mkdir(parents=True, exist_ok=True)

# CRSS data directories (scratchpad for raw data; kept outside code_release per data safety policy)
SCRATCHPAD = Path("/private/tmp/claude-501/-Users-redacted-user/"
                  "373ef59a-33a6-4195-ad55-30c70723863d/scratchpad")

CRSS_YEARS_AVAIL = [2016, 2017, 2018, 2019, 2020, 2021]  # Download-confirmed years

# ---------------------------------------------------------------------------
# Pre-specified analysis parameters (FROZEN 2026-07-14)
# ---------------------------------------------------------------------------
Z_PRE_ESC_MAX_MY     = 2010   # Z=0 upper bound (inclusive) -- preregistration
Z_POST_ESC_MIN_MY    = 2012   # Z=1 lower bound (inclusive) -- FMVSS 126 full mandate
Z_TRANSITION_MY      = 2011   # excluded (95% phase-in, Sivinski 2011 Table p.3)
MY_MIN_COVERAGE      = 2000   # NHTSA API coverage start (ESC data available)
MY_MAX_COVERAGE      = 2010   # Pre-mandate maximum
# Z_WIDE sensitivity: Z=0 for MY in [2000, 2009], Z=1 for MY >= 2012
Z_PRE_ESC_MAX_MY_WIDE = 2009

# FMVSS 126 phase-in rates (Sivinski 2011, Table p.3 -- VERIFIED from source)
FMVSS126_PHASEIN = {
    2009: 0.55,   # "55% with carryover credit"
    2010: 0.75,   # "75% with carryover credit"  (NOT 95% -- correction from prior doc)
    2011: 0.95,   # "95% with carryover credit"
    2012: 1.00,   # "Fully effective"
}

INJ_SEV_VALID    = {0, 1, 2, 3, 4}  # Valid KABCO injury severity codes
INJ_SEV_INJURY   = {1, 2, 3, 4}     # Any injury (Y_binary = 1)
INJ_SEV_EXCLUDE  = {5, 9}            # Unknown severity -- exclude

ALPHA_LEVEL = 0.05
N_BOOTSTRAP = 500         # Bootstrap iterations for CI
BOOTSTRAP_SEED = 42       # Bootstrap seed (fixed before execution)

# ---------------------------------------------------------------------------
# NHTSA 5-Star Safety Ratings API
# ---------------------------------------------------------------------------
NHTSA_API_BASE = "https://api.nhtsa.gov/SafetyRatings"
API_SLEEP_SEC  = 0.15    # Rate limiting sleep between API calls
API_TIMEOUT    = 20       # seconds

# ESC values from NHTSA API
ESC_STANDARD = "Standard"
ESC_NO       = "No"
ESC_OPTIONAL = "Optional"


def _ssl_context() -> ssl.SSLContext:
    """Return SSL context using certifi CA bundle if available, else unverified."""
    try:
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
        return ctx
    except ImportError:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        return ctx


_SSL_CTX = _ssl_context()


def _api_get(url: str) -> Optional[Dict]:
    """Fetch URL as JSON using certifi-verified SSL. Returns None on error."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "CCMP-Research/1.0"})
        with urllib.request.urlopen(req, timeout=API_TIMEOUT, context=_SSL_CTX) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        print(f"  [API ERROR] {url}: {e}", file=sys.stderr)
        return None


def get_vehicle_esc(vehicle_id: int) -> Optional[str]:
    """
    Fetch NHTSAElectronicStabilityControl for a VehicleId.
    Returns 'Standard', 'No', 'Optional', or None (not found/error).
    Caches to CACHE_VEHICLES_DIR/{vehicle_id}.json
    """
    cache_path = CACHE_VEHICLES_DIR / f"{vehicle_id}.json"
    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f).get("esc_value")

    data = _api_get(f"{NHTSA_API_BASE}/VehicleId/{vehicle_id}")
    time.sleep(API_SLEEP_SEC)

    if data is None or data.get("Count", 0) == 0:
        result = {"vehicle_id": vehicle_id, "esc_value": None, "fetched": datetime.utcnow().isoformat()}
        with open(cache_path, "w") as f:
            json.dump(result, f)
        return None

    r = data["Results"][0]
    esc = r.get("NHTSAElectronicStabilityControl", None)
    # Normalize
    if esc and "standard" in esc.lower():
        esc = ESC_STANDARD
    elif esc and "no" == esc.strip().lower():
        esc = ESC_NO
    elif esc and "optional" in esc.lower():
        esc = ESC_OPTIONAL
    else:
        esc = None  # empty string or unrecognized

    result = {
        "vehicle_id": vehicle_id,
        "esc_value": esc,
        "description": r.get("VehicleDescription", ""),
        "fetched": datetime.utcnow().isoformat(),
    }
    with open(cache_path, "w") as f:
        json.dump(result, f)
    return esc


import re as _re
_SAFE_CHARS = _re.compile(r'[/\\:*?"<>|]')


def _safe_name(s: str) -> str:
    """Sanitize string for use as filename component."""
    return _SAFE_CHARS.sub('_', s).replace(' ', '_')


# Level-1 cache: all models for a (make, year) pair from NHTSA API
CACHE_MAKE_YEAR_DIR = CACHE_DIR / "make_year"
CACHE_MAKE_YEAR_DIR.mkdir(parents=True, exist_ok=True)

# Level-2 cache: VehicleIds for a (make, api_model, year) triple
CACHE_VID_LIST_DIR = CACHE_DIR / "vid_lists"
CACHE_VID_LIST_DIR.mkdir(parents=True, exist_ok=True)


def _get_api_models_for_make_year(year: int, make_upper: str) -> Optional[List[str]]:
    """
    Level-1 cache: fetch all model names for (year, make) from NHTSA API.
    Cached to CACHE_MAKE_YEAR_DIR/{year}_{safe(make)}.json.
    Returns list of uppercase model name strings, or None if make not found.
    """
    cache_path = CACHE_MAKE_YEAR_DIR / f"{year}_{_safe_name(make_upper)}.json"
    if cache_path.exists():
        with open(cache_path) as f:
            d = json.load(f)
            return d.get("models")   # None if make not found in API

    make_enc = urllib.parse.quote(make_upper, safe="")
    url      = f"{NHTSA_API_BASE}/modelyear/{year}/make/{make_enc}"
    data     = _api_get(url)
    time.sleep(API_SLEEP_SEC)

    if data is None or data.get("Count", 0) == 0:
        result = {"year": year, "make": make_upper, "models": None,
                  "fetched": datetime.utcnow().isoformat()}
        with open(cache_path, "w") as f:
            json.dump(result, f)
        return None

    models = sorted(set(r.get("Model", "").upper() for r in data["Results"]))
    result = {"year": year, "make": make_upper, "models": models,
              "fetched": datetime.utcnow().isoformat()}
    with open(cache_path, "w") as f:
        json.dump(result, f)
    return models


def _match_crss_model_to_api(crss_model: str, api_models: List[str]) -> List[str]:
    """
    Match a CRSS model name (from vpicdecode) to NHTSA API model names.
    Returns ALL matching API model names (multiple body styles counted together).

    Matching strategy (in priority order):
      1. Forward prefix match (word boundary): CRSS="F-150" matches
         ["F-150", "F-150 EXTENDED CAB", "F-150 SUPER CAB", ...].
         This is the primary strategy -- collects all body styles.
      2. Reverse prefix match: CRSS="F-150 CREW CAB" matches "F-150" if no forward match.
         Returns only the longest API model.

    Conservative: if CRSS model is <3 chars, skip (ambiguous).
    IMPORTANT: Do NOT short-circuit on exact match alone -- include all body styles.
    """
    crss_u = crss_model.upper().strip()
    if not crss_u or len(crss_u) < 3:
        return []

    # Forward prefix: crss_u is an exact match OR crss_u is a word-boundary prefix of api_model
    # "F-150" matches "F-150" (exact) AND "F-150 EXTENDED CAB" (prefix with space)
    forward = [m for m in api_models if m == crss_u or m.startswith(crss_u + " ")]
    if forward:
        return sorted(set(forward))

    # Reverse prefix: api_model is a word-boundary prefix of crss_u
    # "ACCORD SEDAN" (CRSS) -> api has "ACCORD" -> reverse match
    reverse = [m for m in api_models if crss_u == m or crss_u.startswith(m + " ")]
    if reverse:
        return [max(reverse, key=len)]  # most specific API model

    return []


def _get_vehicle_ids_for_api_model(year: int, make_upper: str, api_model: str) -> List[int]:
    """
    Level-2 cache: fetch VehicleIds for (year, make, api_model) from NHTSA API.
    Cached to CACHE_VID_LIST_DIR/{year}_{safe(make)}_{safe(api_model)}.json.
    """
    cache_path = CACHE_VID_LIST_DIR / f"{year}_{_safe_name(make_upper)}_{_safe_name(api_model)}.json"
    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f).get("vehicle_ids", [])

    make_enc  = urllib.parse.quote(make_upper, safe="")
    model_enc = urllib.parse.quote(api_model,  safe="")
    url       = f"{NHTSA_API_BASE}/modelyear/{year}/make/{make_enc}/model/{model_enc}"
    data      = _api_get(url)
    time.sleep(API_SLEEP_SEC)

    if data is None or data.get("Count", 0) == 0:
        result = {"year": year, "make": make_upper, "api_model": api_model,
                  "vehicle_ids": [], "fetched": datetime.utcnow().isoformat()}
        with open(cache_path, "w") as f:
            json.dump(result, f)
        return []

    vids = [r["VehicleId"] for r in data["Results"] if "VehicleId" in r]
    result = {"year": year, "make": make_upper, "api_model": api_model,
              "vehicle_ids": vids, "fetched": datetime.utcnow().isoformat()}
    with open(cache_path, "w") as f:
        json.dump(result, f)
    return vids


def _esc_consensus_from_values(esc_values: List[Optional[str]]) -> str:
    """
    Aggregate ESC values across multiple VehicleId configurations.
    Decision rule (prespecified):
      - All 'Standard' -> 'Standard' (T=1)
      - All 'No'       -> 'No'       (T=0)
      - Any 'Optional' -> 'Optional' (T=unassigned, self-selection layer)
      - Mix of 'No'+'Standard' without 'Optional' -> 'ambiguous' (T=unassigned)
      - All None/empty -> 'not_found' (T=unassigned)
    """
    valid = [v for v in esc_values if v is not None]
    if not valid:
        return "not_found"
    unique = set(valid)
    if ESC_OPTIONAL in unique:
        return ESC_OPTIONAL
    if len(unique) == 1:
        return list(unique)[0]
    return "ambiguous"


def get_make_model_esc(year: int, make: str, model: str) -> Dict:
    """
    Query NHTSA API for all configurations of (year, make, model).
    Uses two-level cache (make_year listing + per-vehicle-id ESC).

    Efficient: makes listing API call only once per (make, year) across all models.

    Returns dict with esc_consensus in:
      'Standard', 'No', 'Optional', 'ambiguous', 'not_found'
    """
    make_upper  = make.upper().strip()
    model_upper = model.upper().strip()
    cache_key   = f"{year}_{_safe_name(make_upper)}_{_safe_name(model_upper)}"
    cache_path  = CACHE_INDEX_DIR / f"{cache_key}.json"

    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)

    # Level-1: get all models for (make, year)
    api_models = _get_api_models_for_make_year(year, make_upper)
    if api_models is None:
        result = {"year": year, "make": make_upper, "model": model_upper,
                  "matched_api_models": [], "vehicle_ids": [],
                  "esc_values": [], "esc_consensus": "not_found",
                  "fetched": datetime.utcnow().isoformat()}
        with open(cache_path, "w") as f:
            json.dump(result, f)
        return result

    # Match CRSS model to API models
    matched = _match_crss_model_to_api(model_upper, api_models)
    if not matched:
        result = {"year": year, "make": make_upper, "model": model_upper,
                  "matched_api_models": [], "vehicle_ids": [],
                  "esc_values": [], "esc_consensus": "not_found",
                  "fetched": datetime.utcnow().isoformat()}
        with open(cache_path, "w") as f:
            json.dump(result, f)
        return result

    # Level-2: get VehicleIds for each matched API model
    all_vids: List[int] = []
    for api_model in matched:
        vids = _get_vehicle_ids_for_api_model(year, make_upper, api_model)
        all_vids.extend(vids)
    all_vids = list(set(all_vids))  # deduplicate

    # Level-3: get ESC for each VehicleId (cached at VehicleId level)
    esc_values = [get_vehicle_esc(vid) for vid in all_vids]
    consensus  = _esc_consensus_from_values(esc_values)

    result = {
        "year": year, "make": make_upper, "model": model_upper,
        "matched_api_models": matched,
        "vehicle_ids": all_vids,
        "esc_values": esc_values,
        "esc_consensus": consensus,
        "fetched": datetime.utcnow().isoformat(),
    }
    with open(cache_path, "w") as f:
        json.dump(result, f)
    return result


def build_esc_cache_for_crss(df_vehicles: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Build NHTSA ESC cache for all unique (MAKE_API, MODEL_API, MOD_YEAR) in CRSS.
    Uses two-level cache: Level-1 = (make, year) listing, Level-2 = (make, api_model, year) VehicleIds.
    This minimises API calls: ~750 Level-1 calls instead of ~5610 per-model calls.

    Returns DataFrame with columns: make, model, year, esc_consensus, T_assign
      T_assign: 1='Standard', 0='No', NaN=unassigned (Optional/ambiguous/not_found)
    """
    # Get unique combinations for Z=0 group (MY 2000-2010)
    pre = df_vehicles[
        (df_vehicles["MOD_YEAR"] >= MY_MIN_COVERAGE) &
        (df_vehicles["MOD_YEAR"] <= MY_MAX_COVERAGE)
    ][["MAKE_API", "MODEL_API", "MOD_YEAR"]].drop_duplicates()
    pre = pre[pre["MAKE_API"].notna() & pre["MODEL_API"].notna()]

    # Phase A: Fetch Level-1 (make, year) listings (deduped, fast)
    make_year_pairs = pre[["MAKE_API", "MOD_YEAR"]].drop_duplicates()
    n_make_year = len(make_year_pairs)
    if verbose:
        print(f"  Phase A: Fetching model listings for {n_make_year} (make, year) pairs...")

    for j, (_, row) in enumerate(make_year_pairs.iterrows()):
        make = str(row["MAKE_API"]).upper().strip()
        year = int(row["MOD_YEAR"])
        if not make or make in ("NAN", "UNKNOWN", ""):
            continue
        _get_api_models_for_make_year(year, make)
        if verbose and (j + 1) % 50 == 0:
            print(f"    ... {j+1}/{n_make_year} make-year pairs processed")

    # Phase B: For each (make, model, year), match and get ESC
    total = len(pre)
    if verbose:
        print(f"  Phase B: Matching {total} unique (make, model, year) combos to NHTSA ESC...")

    rows = []
    for i, (_, row) in enumerate(pre.iterrows()):
        make  = str(row["MAKE_API"]).upper().strip()
        model = str(row["MODEL_API"]).upper().strip()
        year  = int(row["MOD_YEAR"])

        if not make or make in ("NAN", "UNKNOWN", "") or not model or model in ("NAN", ""):
            rows.append({"make": make, "model": model, "year": year,
                         "esc_consensus": "not_found", "T_assign": np.nan})
            continue

        result = get_make_model_esc(year, make, model)
        cons   = result.get("esc_consensus", "not_found")

        t_assign = 1.0 if cons == ESC_STANDARD else (0.0 if cons == ESC_NO else np.nan)
        rows.append({"make": make, "model": model, "year": year,
                     "esc_consensus": cons, "T_assign": t_assign})

        if verbose and (i + 1) % 200 == 0:
            print(f"    ... {i+1}/{total} (make, model, year) combos processed")

    df_cache = pd.DataFrame(rows)
    df_cache.to_csv(ESC_API_CSV, index=False)
    if verbose:
        print(f"  Saved NHTSA API ESC list: {ESC_API_CSV}")
    return df_cache


# ---------------------------------------------------------------------------
# CRSS Data Loading
# ---------------------------------------------------------------------------

def _crss_dir(year: int) -> Path:
    """Return path to CRSS CSV directory for given year."""
    return SCRATCHPAD / f"crss{year}"


def _download_crss(year: int) -> bool:
    """Download and extract CRSS ZIP for given year to scratchpad. Returns True if success."""
    out_dir  = _crss_dir(year)
    zip_path = SCRATCHPAD / f"CRSS{year}CSV.zip"

    # Check if accident CSV already exists (case-insensitive, possibly in subdirectory)
    if _find_csv(out_dir, "accident.csv") is not None if out_dir.exists() else False:
        return True   # Already downloaded and extracted

    out_dir.mkdir(parents=True, exist_ok=True)
    url = f"https://static.nhtsa.gov/nhtsa/downloads/CRSS/{year}/CRSS{year}CSV.zip"
    print(f"  Downloading CRSS {year} from {url}...")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "CCMP-Research/1.0"})
        with urllib.request.urlopen(req, timeout=300, context=_SSL_CTX) as resp, \
             open(str(zip_path), "wb") as out_f:
            while True:
                chunk = resp.read(65536)
                if not chunk:
                    break
                out_f.write(chunk)
    except Exception as e:
        print(f"  [ERROR] Download failed for CRSS {year}: {e}", file=sys.stderr)
        return False

    print(f"  Extracting CRSS {year}...")
    try:
        with zipfile.ZipFile(str(zip_path), "r") as z:
            # Find CSV files (may be in subdirectory)
            names = z.namelist()
            for name in names:
                if name.lower().endswith(".csv"):
                    basename = os.path.basename(name)
                    dest = out_dir / basename.upper()
                    with z.open(name) as src, open(dest, "wb") as dst:
                        dst.write(src.read())
    except Exception as e:
        print(f"  [ERROR] Extract failed for CRSS {year}: {e}", file=sys.stderr)
        return False

    print(f"  CRSS {year} ready: {out_dir}")
    return True


def _find_csv(crss_dir: Path, name: str) -> Optional[Path]:
    """
    Find CSV file case-insensitively in directory or one level of subdirectory.
    Handles CRSS2020 where ZIP extracted into CRSS2020CSV/ subdir.
    """
    # Direct match in top-level directory
    for p in crss_dir.iterdir():
        if p.is_file() and p.name.upper() == name.upper():
            return p
    # One level deep (e.g., crss2020/CRSS2020CSV/accident.csv)
    for sub in crss_dir.iterdir():
        if sub.is_dir():
            for p in sub.iterdir():
                if p.is_file() and p.name.upper() == name.upper():
                    return p
    return None


def load_crss_year(year: int) -> Optional[pd.DataFrame]:
    """
    Load one year of CRSS data. Returns merged (accident + vehicle + person + vpicdecode)
    DataFrame, or None if data not available.

    Columns in returned DataFrame:
      CASENUM, VEH_NO, CRSS_YEAR
      -- from accident: PSU_ACC, PSUSTRAT_ACC, STRATUM, WEIGHT_ACC, REGION, URBANICITY, INT_HWY
      -- from vehicle: MOD_YEAR, MAKE_API (from vpicdecode), MODEL_API (from vpicdecode)
      -- from person: INJ_SEV, WEIGHT_PER (person weight)
    """
    crss_dir = _crss_dir(year)
    ok = _download_crss(year)
    if not ok:
        print(f"  Skipping CRSS {year} (download failed).", file=sys.stderr)
        return None

    # File name patterns (vary by year)
    for acc_name in ["accident.csv", "ACCIDENT.CSV"]:
        acc_path = _find_csv(crss_dir, acc_name)
        if acc_path:
            break
    for veh_name in ["vehicle.csv", "VEHICLE.CSV"]:
        veh_path = _find_csv(crss_dir, veh_name)
        if veh_path:
            break
    for per_name in ["person.csv", "PERSON.CSV"]:
        per_path = _find_csv(crss_dir, per_name)
        if per_path:
            break
    for vpc_name in ["vpicdecode.csv", "VPICDECODE.CSV"]:
        vpc_path = _find_csv(crss_dir, vpc_name)
        if vpc_path:
            break

    missing = [n for n, p in [("accident", acc_path), ("vehicle", veh_path),
                                ("person", per_path), ("vpicdecode", vpc_path)]
               if p is None]
    if missing:
        print(f"  [WARN] CRSS {year} missing files: {missing}", file=sys.stderr)
        return None

    # Load with latin-1 encoding (CRSS uses this)
    print(f"  Loading CRSS {year}...")
    try:
        df_acc = pd.read_csv(str(acc_path),  encoding="latin-1", low_memory=False)
        df_veh = pd.read_csv(str(veh_path),  encoding="latin-1", low_memory=False)
        df_per = pd.read_csv(str(per_path),  encoding="latin-1", low_memory=False)
        df_vpc = pd.read_csv(str(vpc_path),  encoding="latin-1", low_memory=False)
    except Exception as e:
        print(f"  [ERROR] Loading CRSS {year}: {e}", file=sys.stderr)
        return None

    # Normalize column names to uppercase
    df_acc.columns = [c.upper() for c in df_acc.columns]
    df_veh.columns = [c.upper() for c in df_veh.columns]
    df_per.columns = [c.upper() for c in df_per.columns]
    df_vpc.columns = [c.upper() for c in df_vpc.columns]

    # Select relevant columns
    acc_cols = ["CASENUM", "PSU", "PSUSTRAT", "STRATUM", "WEIGHT", "REGION", "URBANICITY", "INT_HWY"]
    acc_cols_avail = [c for c in acc_cols if c in df_acc.columns]
    df_acc = df_acc[acc_cols_avail].rename(columns={"PSU": "PSU_ACC", "PSUSTRAT": "PSUSTRAT_ACC",
                                                      "WEIGHT": "WEIGHT_ACC"})

    veh_cols = ["CASENUM", "VEH_NO", "MOD_YEAR"]
    veh_cols_avail = [c for c in veh_cols if c in df_veh.columns]
    df_veh = df_veh[veh_cols_avail]

    per_cols = ["CASENUM", "VEH_NO", "INJ_SEV", "WEIGHT"]
    per_cols_avail = [c for c in per_cols if c in df_per.columns]
    df_per = df_per[per_cols_avail].rename(columns={"WEIGHT": "WEIGHT_PER"})

    # vpicdecode: Make and Model text
    vpc_cols_needed = ["CASENUM", "VEH_NO"]
    if "MAKE" in df_vpc.columns:
        vpc_cols_needed.append("MAKE")
    if "MODEL" in df_vpc.columns:
        vpc_cols_needed.append("MODEL")
    df_vpc = df_vpc[[c for c in vpc_cols_needed if c in df_vpc.columns]].rename(
        columns={"MAKE": "MAKE_API", "MODEL": "MODEL_API"}
    )

    # Merge: accident × vehicle (CASENUM)
    df = pd.merge(df_per, df_veh, on=["CASENUM", "VEH_NO"], how="inner")
    df = pd.merge(df, df_acc, on="CASENUM", how="left")
    df = pd.merge(df, df_vpc, on=["CASENUM", "VEH_NO"], how="left")

    df["CRSS_YEAR"] = year

    # Numeric coercions
    for col in ["MOD_YEAR", "INJ_SEV", "WEIGHT_PER", "WEIGHT_ACC", "REGION", "URBANICITY", "INT_HWY"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    print(f"  CRSS {year}: {len(df):,} person-vehicle records loaded")
    return df


def pool_crss(years: List[int]) -> pd.DataFrame:
    """Pool CRSS data across multiple years."""
    frames = []
    for year in years:
        df_year = load_crss_year(year)
        if df_year is not None:
            frames.append(df_year)
    if not frames:
        raise RuntimeError("No CRSS data could be loaded.")
    df_pool = pd.concat(frames, ignore_index=True)
    print(f"\nPooled CRSS ({', '.join(str(y) for y in years)}): {len(df_pool):,} records")
    return df_pool


# ---------------------------------------------------------------------------
# Treatment, Instrument, Outcome Assignment
# ---------------------------------------------------------------------------

def assign_variables(df: pd.DataFrame, esc_cache: pd.DataFrame) -> pd.DataFrame:
    """
    Assign Z, T, Y to each person-record.

    Z assignment (pre-specified):
      Z=0 if MOD_YEAR in [2000, 2010]
      Z=1 if MOD_YEAR >= 2012
      Z=NaN otherwise (excluded: MY2011, MY<2000, MY unknown)

    T assignment:
      For Z=1 (MY>=2012): T=1 (FMVSS 126 full mandate)
      For Z=0 (MY in [2000,2010]):
        Look up NHTSA API ESC consensus from esc_cache
        'Standard' -> T=1 (always-taker, early adopter)
        'No'       -> T=0 (never had ESC, potential complier)
        else       -> T=NaN (unassigned: Optional/ambiguous/not_found)
    """
    # Instrument Z
    df = df.copy()
    my = df["MOD_YEAR"]
    df["Z"] = np.nan
    df.loc[(my >= MY_MIN_COVERAGE) & (my <= Z_PRE_ESC_MAX_MY), "Z"] = 0.0
    df.loc[my >= Z_POST_ESC_MIN_MY, "Z"] = 1.0

    # Z_wide (sensitivity)
    df["Z_wide"] = np.nan
    df.loc[(my >= MY_MIN_COVERAGE) & (my <= Z_PRE_ESC_MAX_MY_WIDE), "Z_wide"] = 0.0
    df.loc[my >= Z_POST_ESC_MIN_MY, "Z_wide"] = 1.0

    # Treatment T (start with NaN, fill in below)
    df["T"] = np.nan
    # Post-mandate: T=1 by FMVSS 126
    df.loc[df["Z"] == 1, "T"] = 1.0

    # Pre-mandate: look up from ESC cache
    if "MAKE_API" in df.columns and "MODEL_API" in df.columns:
        # Normalize for merge
        df["_make_up"]  = df["MAKE_API"].fillna("").astype(str).str.upper().str.strip()
        df["_model_up"] = df["MODEL_API"].fillna("").astype(str).str.upper().str.strip()
        df["_year"]     = df["MOD_YEAR"].fillna(0).astype(int)

        esc_lu = esc_cache.copy()
        esc_lu["make"]  = esc_lu["make"].fillna("").astype(str).str.upper().str.strip()
        esc_lu["model"] = esc_lu["model"].fillna("").astype(str).str.upper().str.strip()
        esc_lu["year"]  = esc_lu["year"].astype(int)

        # Merge on make/model/year for pre-mandate rows
        pre_mask = (df["Z"] == 0.0)
        df_pre   = df.loc[pre_mask, ["_make_up", "_model_up", "_year"]].copy()
        df_pre   = df_pre.merge(
            esc_lu[["make", "model", "year", "T_assign"]].rename(
                columns={"make": "_make_up", "model": "_model_up", "year": "_year"}),
            on=["_make_up", "_model_up", "_year"], how="left"
        )
        df.loc[pre_mask, "T"] = df_pre["T_assign"].values

        df.drop(columns=["_make_up", "_model_up", "_year"], inplace=True)

    # Outcome Y
    df["Y_binary"] = np.nan
    df["Y_num"]    = np.nan
    valid_mask = df["INJ_SEV"].isin(INJ_SEV_VALID)
    df.loc[valid_mask, "Y_binary"] = (df.loc[valid_mask, "INJ_SEV"].isin(INJ_SEV_INJURY)).astype(float)
    df.loc[valid_mask, "Y_num"]    = df.loc[valid_mask, "INJ_SEV"].astype(float)

    # Analysis weight (CRSS person weight)
    df["W"] = df["WEIGHT_PER"].fillna(0.0)
    df.loc[df["W"] <= 0, "W"] = np.nan

    return df


def assign_treatment_categories(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add T_category column for reporting:
      'standard'    : T=1, Z=0 (always-taker in pre-mandate)
      'no_esc'      : T=0, Z=0 (potential complier in pre-mandate)
      'mandatory'   : T=1, Z=1 (post-mandate)
      'optional'    : T=NaN, Z=0, because ESC was Optional
      'not_in_api'  : T=NaN, Z=0, not found in NHTSA API
      'excluded_z'  : Z=NaN (MY2011, MY<2000, MY unknown)
      'no_inj_data' : Y=NaN
    """
    # This is for descriptive stats only; not used in estimation
    df = df.copy()
    df["T_category"] = "other"
    df.loc[df["Z"].isna(), "T_category"] = "excluded_z"
    df.loc[(df["Z"] == 0.0) & (df["T"] == 1.0), "T_category"] = "standard"
    df.loc[(df["Z"] == 0.0) & (df["T"] == 0.0), "T_category"] = "no_esc"
    df.loc[(df["Z"] == 1.0) & (df["T"] == 1.0), "T_category"] = "mandatory"
    df.loc[(df["Z"] == 0.0) & (df["T"].isna()), "T_category"] = "unassigned_pre"
    return df


# ---------------------------------------------------------------------------
# IPSW Construction
# ---------------------------------------------------------------------------

def compute_stratum_ipsw(df: pd.DataFrame) -> pd.DataFrame:
    """
    Approximate IPSW for collider correction using URBANICITY x INT_HWY x REGION
    stratum-level crash-rate proxy.

    Approximation: use relative crash frequency within strata (from CRSS design weights)
    as a proxy for P(crash | stratum). Upweight low-crash-probability strata.

    This is a ROUGH approximation since absolute P(crash) requires VMT denominator
    (not available in CRSS). As per preregistration §2-D and §V-E of the paper,
    the test is robust to spatial coarsening.

    Returns df with column W_IPSW_STRATUM added.
    Note: W_IPSW_STRATUM is relative weight -- normalized to mean 1.
    """
    df = df.copy()
    strat_cols = ["URBANICITY", "INT_HWY", "REGION"]
    avail = [c for c in strat_cols if c in df.columns]
    if len(avail) < 2:
        df["W_IPSW_STRATUM"] = 1.0
        return df

    # For each stratum, compute the weighted crash count (proxy for P(crash) * constant)
    # Crash rate proxy: use inverse of (stratum_crash_count / total_crashes) -- relative
    # Lower stratum share -> higher weight (lower P(crash) -> higher IPSW)
    df["_strat_key"] = df[avail].astype(str).apply(lambda x: "_".join(x), axis=1)
    strat_totals = df.groupby("_strat_key")["W"].sum()  # sum of design weights = national crash count
    grand_total  = strat_totals.sum()

    # Relative crash proportion per stratum
    strat_prop = (strat_totals / grand_total)

    # IPSW = 1 / relative_crash_prop (rescaled to mean 1)
    ipsw_map   = 1.0 / strat_prop
    ipsw_mean  = (df["_strat_key"].map(ipsw_map) * df["W"]).sum() / df["W"].sum()
    ipsw_scaled = df["_strat_key"].map(ipsw_map) / ipsw_mean  # normalize

    df["W_IPSW_STRATUM"] = ipsw_scaled.fillna(1.0)
    df.drop(columns=["_strat_key"], inplace=True)
    return df


# ---------------------------------------------------------------------------
# IV Wald Estimator (identical to hc_t2_reduction.py _ipsw_wald)
# ---------------------------------------------------------------------------

def _hajek_wald(
    Y: np.ndarray,
    T: np.ndarray,
    Z: np.ndarray,
    W: np.ndarray,
) -> Dict:
    """
    IPSW-weighted IV Wald estimator (Hajek) with delta-method SE.
    Matches hc_t2_reduction.py:_ipsw_wald exactly.

    Returns: tau, SE_tau, CI_L, CI_U, z_stat, p_value, FS, RF, n_Z0, n_Z1
    """
    m1 = (Z == 1)
    m0 = (Z == 0)
    n1 = int(m1.sum())
    n0 = int(m0.sum())

    if n1 < 10 or n0 < 10:
        nan = float("nan")
        return {"tau": nan, "SE_tau": nan, "CI_L": nan, "CI_U": nan,
                "z_stat": nan, "p_value": nan, "reject": False,
                "FS": nan, "RF": nan, "n_Z0": n0, "n_Z1": n1}

    w1, w0   = W[m1], W[m0]
    sw1, sw0 = w1.sum(), w0.sum()

    EY_Z1 = float(np.dot(Y[m1], w1) / sw1)
    EY_Z0 = float(np.dot(Y[m0], w0) / sw0)
    ET_Z1 = float(np.dot(T[m1], w1) / sw1)
    ET_Z0 = float(np.dot(T[m0], w0) / sw0)

    RF = EY_Z1 - EY_Z0
    FS = ET_Z1 - ET_Z0

    if abs(FS) < 1e-8:
        nan = float("nan")
        return {"tau": nan, "SE_tau": nan, "CI_L": nan, "CI_U": nan,
                "z_stat": nan, "p_value": nan, "reject": False,
                "FS": float(FS), "RF": float(RF), "n_Z0": n0, "n_Z1": n1}

    tau = RF / FS

    resY_Z1 = Y[m1] - EY_Z1
    resY_Z0 = Y[m0] - EY_Z0
    resT_Z1 = T[m1] - ET_Z1
    resT_Z0 = T[m0] - ET_Z0

    var_RF = (np.dot(w1**2, resY_Z1**2) / sw1**2 +
              np.dot(w0**2, resY_Z0**2) / sw0**2)
    var_FS = (np.dot(w1**2, resT_Z1**2) / sw1**2 +
              np.dot(w0**2, resT_Z0**2) / sw0**2)

    SE_tau = float(np.sqrt(max(1e-20, var_RF / FS**2 + RF**2 * var_FS / FS**4)))
    z_stat = tau / SE_tau
    p_val  = float(2.0 * scipy.stats.norm.sf(abs(z_stat)))

    return {
        "tau"    : float(tau),
        "SE_tau" : SE_tau,
        "CI_L"   : float(tau - 1.96 * SE_tau),
        "CI_U"   : float(tau + 1.96 * SE_tau),
        "z_stat" : float(z_stat),
        "p_value": p_val,
        "reject" : bool(p_val < ALPHA_LEVEL),
        "FS"     : float(FS),
        "RF"     : float(RF),
        "n_Z0"   : n0,
        "n_Z1"   : n1,
        "EY_Z1"  : float(EY_Z1),
        "EY_Z0"  : float(EY_Z0),
        "ET_Z1"  : float(ET_Z1),
        "ET_Z0"  : float(ET_Z0),
    }


def _first_stage_fstat(
    T: np.ndarray, Z: np.ndarray, W: np.ndarray
) -> Dict:
    """
    Approximate first-stage F-statistic for instrument strength.
    F = (FS / SE_FS)^2 where SE_FS = Hajek sandwich SE of FS.
    """
    m1 = (Z == 1); m0 = (Z == 0)
    w1, w0 = W[m1], W[m0]
    sw1, sw0 = w1.sum(), w0.sum()

    if sw1 < 1e-10 or sw0 < 1e-10:
        return {"FS": float("nan"), "SE_FS": float("nan"), "F_stat": float("nan")}

    ET_Z1 = float(np.dot(T[m1], w1) / sw1)
    ET_Z0 = float(np.dot(T[m0], w0) / sw0)
    FS    = ET_Z1 - ET_Z0

    resT_Z1 = T[m1] - ET_Z1
    resT_Z0 = T[m0] - ET_Z0
    var_FS  = (np.dot(w1**2, resT_Z1**2) / sw1**2 +
               np.dot(w0**2, resT_Z0**2) / sw0**2)
    SE_FS   = float(np.sqrt(max(1e-20, var_FS)))
    F_stat  = float((FS / SE_FS)**2) if SE_FS > 1e-10 else float("inf")

    return {"FS": float(FS), "SE_FS": SE_FS, "F_stat": F_stat,
            "ET_Z0": float(ET_Z0), "ET_Z1": float(ET_Z1)}


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------

def bootstrap_iv_wald(
    df_analytic: pd.DataFrame,
    Y_col: str = "Y_binary",
    Z_col: str = "Z",
    T_col: str = "T",
    W_col: str = "W",
    B: int = N_BOOTSTRAP,
    seed: int = BOOTSTRAP_SEED,
) -> Dict:
    """
    Stratified cluster bootstrap CI for IV Wald estimator.
    Cluster unit: CASENUM (accident level) within PSUSTRAT stratum.
    Composite PSU key for pooled years: CRSS_YEAR * 100000 + PSU_ACC.

    Returns: bs_tau list, CI_L_boot, CI_U_boot, SE_boot
    """
    rng = np.random.default_rng(seed)

    # Define strata and clusters
    if "PSUSTRAT_ACC" in df_analytic.columns:
        strat_col = "PSUSTRAT_ACC"
    elif "STRATUM" in df_analytic.columns:
        strat_col = "STRATUM"
    else:
        strat_col = None

    if "CASENUM" in df_analytic.columns:
        cluster_col = "CASENUM"
    else:
        cluster_col = None

    # Build composite cluster ID that's unique per year
    if "CRSS_YEAR" in df_analytic.columns:
        df_analytic = df_analytic.copy()
        df_analytic["_cluster_id"] = (
            df_analytic["CRSS_YEAR"].astype(str) + "_" +
            df_analytic.get(cluster_col, pd.Series(range(len(df_analytic)))).astype(str)
        )
        cluster_col = "_cluster_id"

    tau_obs = df_analytic[Y_col].notna() & df_analytic[Z_col].notna() & \
              df_analytic[T_col].notna() & df_analytic[W_col].notna()

    bs_taus = []
    strats = (df_analytic[strat_col].unique()
              if strat_col and strat_col in df_analytic.columns
              else [None])

    for b in range(B):
        frames = []
        for strat in strats:
            if strat is None:
                sub = df_analytic
            else:
                sub = df_analytic[df_analytic[strat_col] == strat]

            if cluster_col and cluster_col in sub.columns:
                clusters = sub[cluster_col].unique()
                sampled  = rng.choice(clusters, size=len(clusters), replace=True)
                chunks   = [sub[sub[cluster_col] == c] for c in sampled]
                bs_sub   = pd.concat(chunks, ignore_index=True) if chunks else sub
            else:
                idx    = rng.integers(0, len(sub), size=len(sub))
                bs_sub = sub.iloc[idx]

            frames.append(bs_sub)

        df_bs = pd.concat(frames, ignore_index=True) if frames else df_analytic
        mask  = df_bs[Y_col].notna() & df_bs[Z_col].notna() & \
                df_bs[T_col].notna() & df_bs[W_col].notna()
        df_bs = df_bs[mask]

        if len(df_bs) < 20:
            continue

        res = _hajek_wald(
            df_bs[Y_col].values,
            df_bs[T_col].values,
            df_bs[Z_col].values,
            df_bs[W_col].values,
        )
        if not np.isnan(res["tau"]):
            bs_taus.append(res["tau"])

    if len(bs_taus) < 10:
        return {"bs_taus": bs_taus, "CI_L_boot": float("nan"),
                "CI_U_boot": float("nan"), "SE_boot": float("nan"),
                "n_boot_valid": len(bs_taus)}

    bs_arr    = np.array(bs_taus)
    CI_L_boot = float(np.percentile(bs_arr, 2.5))
    CI_U_boot = float(np.percentile(bs_arr, 97.5))
    SE_boot   = float(bs_arr.std(ddof=1))

    return {"bs_taus": bs_taus, "CI_L_boot": CI_L_boot,
            "CI_U_boot": CI_U_boot, "SE_boot": SE_boot,
            "n_boot_valid": len(bs_taus)}


# ---------------------------------------------------------------------------
# Dang/Sivinski List Validation
# ---------------------------------------------------------------------------

def validate_against_dang_sivinski(esc_cache: pd.DataFrame) -> Dict:
    """
    Cross-check NHTSA API results against Dang 2007 / Sivinski 2011 list.
    Returns dict with match summary and discrepancies.
    """
    if not ESC_LIST_CSV.exists():
        return {"status": "esc_list_csv_not_found"}

    ds_list = pd.read_csv(ESC_LIST_CSV)
    discrepancies = []
    matches       = []

    for _, row in ds_list.iterrows():
        make = str(row["make"]).upper().strip()
        model_kw  = str(row["model_kw"]).upper().strip()
        esc_std   = int(row["esc_std_year"])
        no_esc_max = int(row.get("no_esc_max_year", esc_std - 1))

        # Check in API cache: year = no_esc_max_year -> should be 'No' or 'Standard' but older
        # Check year = esc_std_year -> should be 'Standard'
        for chk_year, expected in [(no_esc_max, ESC_NO), (esc_std, ESC_STANDARD)]:
            sub = esc_cache[
                (esc_cache["make"].str.upper() == make) &
                (esc_cache["year"] == chk_year) &
                (esc_cache["model"].str.upper().str.contains(model_kw.split()[0], na=False))
            ]
            if sub.empty:
                continue
            api_cons = sub.iloc[0]["esc_consensus"]
            if api_cons == expected:
                matches.append({"make": make, "model_kw": model_kw, "year": chk_year,
                                 "expected": expected, "api": api_cons, "status": "match"})
            else:
                discrepancies.append({"make": make, "model_kw": model_kw, "year": chk_year,
                                       "expected": expected, "api": api_cons, "status": "mismatch"})

    return {
        "n_checks": len(matches) + len(discrepancies),
        "n_matches": len(matches),
        "n_discrepancies": len(discrepancies),
        "discrepancies": discrepancies[:20],  # cap for readability
    }


# ---------------------------------------------------------------------------
# Power Assessment
# ---------------------------------------------------------------------------

def assess_power(n_eff: float, tau_obs: float, se_obs: float) -> Dict:
    """
    Post-hoc power assessment for H0: tau=0 at alpha=0.05 (two-sided).
    Using normal approximation.
    """
    if np.isnan(tau_obs) or np.isnan(se_obs) or se_obs <= 0:
        return {"power_at_obs_tau": float("nan"), "mde_80pct": float("nan")}

    # Power at observed tau
    z_alpha = scipy.stats.norm.ppf(1 - ALPHA_LEVEL / 2)
    ncp     = abs(tau_obs) / se_obs   # non-centrality parameter
    power   = float(1 - scipy.stats.norm.cdf(z_alpha - ncp) + scipy.stats.norm.cdf(-z_alpha - ncp))

    # MDE for 80% power
    z_beta  = scipy.stats.norm.ppf(0.80)
    mde_80  = float((z_alpha + z_beta) * se_obs)   # approx MDE for 80% power

    # From paper Fig 4 reference points (N_rep = effective n, tau = true effect)
    # |tau| = 0.06 -> power 0.80 at N_rep = 20,000; |tau| = 0.03 -> N_rep = 100,000
    power_ref = {}
    for tau_ref in [0.015, 0.030, 0.060]:
        ncp_ref = tau_ref / se_obs if se_obs > 0 else 0
        pwr_ref = float(1 - scipy.stats.norm.cdf(z_alpha - ncp_ref) +
                        scipy.stats.norm.cdf(-z_alpha - ncp_ref))
        power_ref[f"|tau|={tau_ref:.3f}"] = round(pwr_ref, 3)

    return {
        "power_at_obs_tau": round(power, 3),
        "mde_80pct_se_based": round(mde_80, 4),
        "se_obs": round(se_obs, 5),
        "power_at_reference_tau": power_ref,
    }


# ---------------------------------------------------------------------------
# Environment Log
# ---------------------------------------------------------------------------

def get_env_log(years: List[int], n_total: int, n_analytic: int) -> Dict:
    """Collect reproducibility metadata."""
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(BASE_DIR), stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        git_commit = "unavailable"

    return {
        "timestamp"    : datetime.utcnow().isoformat() + "Z",
        "script"       : "realdata_tier2_pointtest.py",
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "scipy_version" : scipy.__version__,
        "platform"     : platform.platform(),
        "hostname"     : socket.gethostname(),
        "git_commit"   : git_commit,
        "GPU"          : "none (CPU-only)",
        "crss_years"   : years,
        "n_total_records": n_total,
        "n_analytic"   : n_analytic,
        "z_pre_esc_max": Z_PRE_ESC_MAX_MY,
        "z_post_esc_min": Z_POST_ESC_MIN_MY,
        "z_transition" : Z_TRANSITION_MY,
        "alpha_level"  : ALPHA_LEVEL,
        "bootstrap_B"  : N_BOOTSTRAP,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "esc_list_csv" : str(ESC_LIST_CSV),
        "esc_api_csv"  : str(ESC_API_CSV),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> Dict:
    print("=" * 72)
    print("Real-Data Tier-2 Point Test: CCMP / Risk Homeostasis")
    print(f"Start: {datetime.utcnow().isoformat()}Z")
    print("=" * 72)

    print("\n[PRE-SPECIFIED ANALYSIS DEFINITION]")
    print(f"  Z=0: MY in [{MY_MIN_COVERAGE}, {Z_PRE_ESC_MAX_MY}] (pre-mandate)")
    print(f"  Z=1: MY >= {Z_POST_ESC_MIN_MY} (FMVSS 126 fully effective)")
    print(f"  MY {Z_TRANSITION_MY} excluded (95% phase-in transition)")
    print(f"  T source (primary): NHTSA 5-Star Safety Ratings API")
    print(f"  T source (validation): Dang 2007 / Sivinski 2011 ({ESC_LIST_CSV.name})")
    print(f"  Y: I(INJ_SEV in {{1,2,3,4}}) = any injury (primary)")
    print(f"  IPSW: CRSS person WEIGHT only (no collider IPSW, GPS unavailable)")
    print(f"  H0: tau = 0 (complete risk homeostasis, kappa_LATE = 1)")
    print(f"  Alpha: {ALPHA_LEVEL}, Bootstrap B={N_BOOTSTRAP}, seed={BOOTSTRAP_SEED}")

    # ------------------------------------------------------------------ STEP 1: Load CRSS
    print(f"\n[STEP 1] Loading CRSS data for years: {CRSS_YEARS_AVAIL}")
    df_raw = pool_crss(CRSS_YEARS_AVAIL)
    n_raw  = len(df_raw)
    print(f"  Total records loaded: {n_raw:,}")

    # ------------------------------------------------------------------ STEP 2: ESC Cache
    print(f"\n[STEP 2] Building NHTSA ESC cache for pre-mandate vehicles")
    # Load or build ESC cache
    if ESC_API_CSV.exists():
        print(f"  Loading existing ESC API cache: {ESC_API_CSV}")
        esc_cache = pd.read_csv(ESC_API_CSV)
        print(f"  Cache entries: {len(esc_cache):,}")
    else:
        print("  Building ESC cache from NHTSA 5-Star Safety Ratings API...")
        esc_cache = build_esc_cache_for_crss(df_raw, verbose=True)
        print(f"  Built cache: {len(esc_cache):,} unique (make, model, year) combos")

    # Summarize cache
    cache_summary = esc_cache["esc_consensus"].value_counts().to_dict() if len(esc_cache) > 0 else {}
    print(f"  ESC cache consensus breakdown: {cache_summary}")

    # ------------------------------------------------------------------ STEP 3: Assign variables
    print(f"\n[STEP 3] Assigning Z, T, Y variables")
    df = assign_variables(df_raw, esc_cache)
    df = assign_treatment_categories(df)
    df = compute_stratum_ipsw(df)

    # Coverage summary
    print("\n  [Coverage Report: T assignment in Z=0 pre-mandate group]")
    z0_mask = (df["Z"] == 0.0) & df["MOD_YEAR"].notna()
    n_z0    = z0_mask.sum()

    t1_mask      = z0_mask & (df["T"] == 1.0)   # Standard ESC (always-taker)
    t0_mask      = z0_mask & (df["T"] == 0.0)   # No ESC (potential complier)
    tna_mask     = z0_mask & df["T"].isna()      # Unassigned
    n_t1         = t1_mask.sum()
    n_t0         = t0_mask.sum()
    n_tna        = tna_mask.sum()

    print(f"  Z=0 (MY {MY_MIN_COVERAGE}-{Z_PRE_ESC_MAX_MY}) total persons: {n_z0:,}")
    print(f"    T=1 (Standard ESC, always-takers):  {n_t1:,} ({100*n_t1/max(n_z0,1):.1f}%)")
    print(f"    T=0 (No ESC, potential compliers):   {n_t0:,} ({100*n_t0/max(n_z0,1):.1f}%)")
    print(f"    T=unassigned (Optional/ambiguous):   {n_tna:,} ({100*n_tna/max(n_z0,1):.1f}%)")

    # Breakdown of T=unassigned reasons (from ESC cache)
    z0_veh = df[z0_mask & df["T"].isna()][["MAKE_API", "MODEL_API", "MOD_YEAR"]].drop_duplicates()
    if len(z0_veh) > 0 and len(esc_cache) > 0:
        z0_veh_merged = z0_veh.merge(
            esc_cache.rename(columns={"make": "MAKE_API", "model": "MODEL_API", "year": "MOD_YEAR"}),
            on=["MAKE_API", "MODEL_API", "MOD_YEAR"], how="left"
        )
        reason_counts = z0_veh_merged["esc_consensus"].fillna("not_found").value_counts().to_dict()
        print(f"    Unassigned breakdown (unique make/model/year): {reason_counts}")

    # Z=1 summary
    z1_mask = (df["Z"] == 1.0) & df["T"].notna()
    n_z1    = z1_mask.sum()
    print(f"\n  Z=1 (MY>={Z_POST_ESC_MIN_MY}) total persons: {n_z1:,} (T=1 by mandate)")

    # Excluded
    n_trans = (df["MOD_YEAR"] == Z_TRANSITION_MY).sum()
    n_old   = (df["MOD_YEAR"] < MY_MIN_COVERAGE).sum()
    print(f"  Excluded MY {Z_TRANSITION_MY} (transition): {n_trans:,}")
    print(f"  Excluded MY<{MY_MIN_COVERAGE} (too old):     {n_old:,}")

    # ------------------------------------------------------------------ STEP 4: Analytic sample
    print(f"\n[STEP 4] Constructing analytic samples")

    # Primary analytic sample: T assigned, Y observed, W positive
    analytic_mask = (
        df["Z"].notna() &
        df["T"].notna() &
        df["Y_binary"].notna() &
        df["W"].notna() &
        (df["W"] > 0)
    )
    df_a = df[analytic_mask].copy()
    n_analytic = len(df_a)

    n_analytic_z0 = (df_a["Z"] == 0).sum()
    n_analytic_z1 = (df_a["Z"] == 1).sum()
    n_analytic_t0 = (df_a["T"] == 0).sum()
    n_analytic_t1 = (df_a["T"] == 1).sum()

    print(f"  Primary analytic sample: N = {n_analytic:,}")
    print(f"    Z=0 (pre-mandate): {n_analytic_z0:,}")
    print(f"    Z=1 (post-mandate): {n_analytic_z1:,}")
    print(f"    T=0 (no ESC):      {n_analytic_t0:,}")
    print(f"    T=1 (has ESC):     {n_analytic_t1:,}")

    # Y distribution
    y_dist = df_a["Y_binary"].value_counts().to_dict()
    print(f"  Y_binary distribution: {{0 (no injury): {y_dist.get(0,0):,}, 1 (injury): {y_dist.get(1,0):,}}}")

    inj_rate_z0 = df_a[df_a["Z"]==0]["Y_binary"].mean()
    inj_rate_z1 = df_a[df_a["Z"]==1]["Y_binary"].mean()
    print(f"  Unweighted injury rate: Z=0: {inj_rate_z0:.3f}, Z=1: {inj_rate_z1:.3f}")

    # ------------------------------------------------------------------ STEP 5: First stage
    print(f"\n[STEP 5] First stage (instrument strength)")
    Y_a = df_a["Y_binary"].values
    T_a = df_a["T"].values
    Z_a = df_a["Z"].values
    W_a = df_a["W"].values

    fs_res = _first_stage_fstat(T_a, Z_a, W_a)
    print(f"  FS = E[T|Z=1] - E[T|Z=0] = {fs_res['FS']:.4f}")
    print(f"  E[T|Z=0] = {fs_res['ET_Z0']:.4f}  E[T|Z=1] = {fs_res['ET_Z1']:.4f}")
    print(f"  SE_FS = {fs_res['SE_FS']:.5f}  F-stat = {fs_res['F_stat']:.1f}")
    if fs_res["F_stat"] < 10:
        print("  [WARN] F-stat < 10: WEAK INSTRUMENT. Wald IV tau has inflated variance.")

    # ------------------------------------------------------------------ STEP 6: IV Wald (primary)
    print(f"\n[STEP 6] IV Wald (primary: Y_binary, W=CRSS WEIGHT, Z=Z_binary)")
    wald_primary = _hajek_wald(Y_a, T_a, Z_a, W_a)
    tau_hat  = wald_primary["tau"]
    se_hat   = wald_primary["SE_tau"]
    ci_l     = wald_primary["CI_L"]
    ci_u     = wald_primary["CI_U"]
    z_stat   = wald_primary["z_stat"]
    p_val    = wald_primary["p_value"]

    print(f"  tau_hat (IV Wald) = {tau_hat:.5f}")
    print(f"  SE     = {se_hat:.5f}")
    print(f"  95% CI = [{ci_l:.5f}, {ci_u:.5f}]")
    print(f"  z-stat = {z_stat:.3f}   p-value = {p_val:.4f}")
    print(f"  H0: tau=0  -> {'REJECT' if wald_primary['reject'] else 'FAIL TO REJECT'}")
    print(f"  RF (reduced form) = {wald_primary['RF']:.5f}")
    print(f"  Interpretation: tau < 0 means ESC reduces injury (kappa_LATE < 1)")

    # Robustness 1: Y_numeric
    print(f"\n  [Robustness: Y_numeric (INJ_SEV 0-4)]")
    mask_yn = df_a["Y_num"].notna()
    if mask_yn.sum() >= 100:
        df_yn = df_a[mask_yn]
        wald_num = _hajek_wald(
            df_yn["Y_num"].values, df_yn["T"].values,
            df_yn["Z"].values,     df_yn["W"].values
        )
        print(f"  tau_hat (numeric) = {wald_num['tau']:.5f}  p = {wald_num['p_value']:.4f}")
    else:
        wald_num = {}
        print("  Insufficient Y_num observations")

    # Robustness 2: Z_wide (MY<=2009 pre-mandate)
    print(f"\n  [Robustness: Z_wide (pre-mandate MY<={Z_PRE_ESC_MAX_MY_WIDE})]")
    if "Z_wide" in df_a.columns:
        df_zw  = df_a[df_a["Z_wide"].notna()].copy()
        wald_zw = _hajek_wald(
            df_zw["Y_binary"].values, df_zw["T"].values,
            df_zw["Z_wide"].values,   df_zw["W"].values
        )
        print(f"  tau_hat (Z_wide) = {wald_zw['tau']:.5f}  p = {wald_zw['p_value']:.4f}")
        print(f"  n_Z0 = {wald_zw['n_Z0']:,}, n_Z1 = {wald_zw['n_Z1']:,}")
    else:
        wald_zw = {}

    # Robustness 3: IPSW stratum-weighted
    print(f"\n  [Robustness: stratum IPSW (URBANICITY x INT_HWY x REGION)]")
    if "W_IPSW_STRATUM" in df_a.columns:
        W_ipsw = df_a["W"].values * df_a["W_IPSW_STRATUM"].values
        wald_ipsw = _hajek_wald(Y_a, T_a, Z_a, W_ipsw)
        print(f"  tau_hat (IPSW) = {wald_ipsw['tau']:.5f}  p = {wald_ipsw['p_value']:.4f}")
    else:
        wald_ipsw = {}

    # ------------------------------------------------------------------ STEP 7: Bootstrap CI
    print(f"\n[STEP 7] Bootstrap CI (B={N_BOOTSTRAP}, cluster=CASENUM within PSUSTRAT)")
    bs_res = bootstrap_iv_wald(df_a, Y_col="Y_binary", Z_col="Z", T_col="T", W_col="W")
    print(f"  Bootstrap CI: [{bs_res['CI_L_boot']:.5f}, {bs_res['CI_U_boot']:.5f}]")
    print(f"  SE_boot = {bs_res['SE_boot']:.5f}  (from {bs_res['n_boot_valid']} valid iterations)")

    # ------------------------------------------------------------------ STEP 8: Power
    print(f"\n[STEP 8] Power assessment")
    pwr = assess_power(n_analytic, tau_hat, se_hat)
    print(f"  Power at observed |tau| = {abs(tau_hat):.4f}: {pwr['power_at_obs_tau']:.3f}")
    print(f"  MDE for 80% power: {pwr['mde_80pct_se_based']:.4f}")
    print(f"  Power at reference |tau| values: {pwr['power_at_reference_tau']}")
    if pwr["power_at_obs_tau"] < 0.5:
        print("  [WARN] Power < 0.50 -- under-powered. Non-rejection is consistent with"
              " both H0 being true and power being insufficient.")

    # ------------------------------------------------------------------ STEP 9: Dang/Sivinski validation
    print(f"\n[STEP 9] Cross-validation against Dang 2007 / Sivinski 2011 list")
    ds_val = validate_against_dang_sivinski(esc_cache)
    print(f"  Total checks: {ds_val.get('n_checks', 0)}")
    print(f"  Matches: {ds_val.get('n_matches', 0)}")
    print(f"  Discrepancies: {ds_val.get('n_discrepancies', 0)}")
    if ds_val.get("discrepancies"):
        for d in ds_val["discrepancies"][:5]:
            print(f"    Discrepancy: {d}")

    # ------------------------------------------------------------------ Save results
    stem = f"realdata_tier2_result_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    env_log = get_env_log(CRSS_YEARS_AVAIL, n_raw, n_analytic)

    aggregate = {
        "env_log"             : env_log,
        "coverage"            : {
            "n_z0_total"     : int(n_z0),
            "n_z0_T1"        : int(n_t1),
            "n_z0_T0"        : int(n_t0),
            "n_z0_Tunassigned": int(n_tna),
            "pct_T1_z0"      : round(100*n_t1/max(n_z0,1), 2),
            "pct_T0_z0"      : round(100*n_t0/max(n_z0,1), 2),
            "pct_Tna_z0"     : round(100*n_tna/max(n_z0,1), 2),
            "n_z1"           : int(n_z1),
            "n_analytic"     : int(n_analytic),
        },
        "first_stage"         : fs_res,
        "wald_primary"        : wald_primary,
        "wald_numeric_Y"      : wald_num,
        "wald_Z_wide"         : wald_zw if wald_zw else {},
        "wald_IPSW_stratum"   : wald_ipsw if wald_ipsw else {},
        "bootstrap"           : {k: v for k, v in bs_res.items() if k != "bs_taus"},
        "power_assessment"    : pwr,
        "dang_sivinski_valid" : ds_val,
        "analysis_definition" : {
            "Z_0_MY_range"     : f"{MY_MIN_COVERAGE}-{Z_PRE_ESC_MAX_MY}",
            "Z_1_MY_min"       : Z_POST_ESC_MIN_MY,
            "Z_transition"     : Z_TRANSITION_MY,
            "T_source_primary" : "NHTSA 5-Star Safety Ratings API",
            "T_source_validate": "Dang 2007 / Sivinski 2011 (esc_equipment_list.csv)",
            "Y_primary"        : "I(INJ_SEV in {1,2,3,4})",
            "W_primary"        : "CRSS person WEIGHT",
            "IPSW"             : "CRSS design weight only (no collider IPSW, GPS unavailable)",
            "note_T_Z_identity": (
                "When T=T(NHTSA_API) and Z=FMVSS126, FS is NOT 1 -- "
                "pre-mandate vehicles with 'No' ESC form T=0, those with 'Standard' form T=1. "
                "This gives a non-trivial first stage F > 10."
            ),
        },
    }

    json_path = RESULTS_DIR / f"{stem}.json"
    with open(json_path, "w") as f:
        json.dump(aggregate, f, indent=2, allow_nan=True)
    print(f"\nSaved JSON: {json_path}")

    # Summary for human reading
    print("\n" + "=" * 72)
    print("FINAL RESULT SUMMARY")
    print("=" * 72)
    print(f"Dataset  : CRSS pooled ({', '.join(str(y) for y in CRSS_YEARS_AVAIL)})")
    print(f"N_analytic: {n_analytic:,}")
    print(f"N_Z0     : {n_analytic_z0:,}  N_Z1: {n_analytic_z1:,}")
    print(f"")
    print(f"First Stage: FS={fs_res['FS']:.4f}  F-stat={fs_res['F_stat']:.1f}")
    print(f"")
    print(f"PRIMARY RESULT (Y_binary, Z_binary, W=CRSS_WEIGHT):")
    print(f"  tau_hat = {tau_hat:.5f}")
    print(f"  SE      = {se_hat:.5f}")
    print(f"  95% CI  = [{ci_l:.5f}, {ci_u:.5f}]  (delta method)")
    print(f"  95% CI  = [{bs_res['CI_L_boot']:.5f}, {bs_res['CI_U_boot']:.5f}]  (bootstrap)")
    print(f"  z-stat  = {z_stat:.3f}   p-value = {p_val:.4f}")
    print(f"  H0: tau=0 -> {'REJECT at alpha=0.05' if wald_primary['reject'] else 'FAIL TO REJECT at alpha=0.05'}")
    print(f"")
    print(f"Power at observed tau: {pwr['power_at_obs_tau']:.3f}")
    print(f"MDE for 80% power: {pwr['mde_80pct_se_based']:.4f}")
    print(f"")
    print(f"HONEST EVALUATION:")
    if wald_primary["reject"]:
        print(f"  Result: H0 REJECTED. Evidence against complete risk homeostasis (kappa_LATE=1).")
        if tau_hat < 0:
            print(f"  tau < 0 implies ESC REDUCES injury, consistent with kappa_LATE < 1.")
        else:
            print(f"  tau > 0 implies ESC INCREASES injury -- check direction carefully.")
    else:
        print(f"  Result: FAILED TO REJECT H0. Cannot distinguish tau=0 from observed data.")
        print(f"  This does NOT confirm risk homeostasis -- power may be insufficient.")
        print(f"  Power estimate: {pwr['power_at_obs_tau']:.3f}")
        print(f"  Even 'meaningful' effects (|tau|=0.06) would need power ~{pwr['power_at_reference_tau'].get('|tau|=0.060', 'N/A')}")

    print(f"\nEnd: {datetime.utcnow().isoformat()}Z")
    print("=" * 72)

    return aggregate


if __name__ == "__main__":
    result = main()
