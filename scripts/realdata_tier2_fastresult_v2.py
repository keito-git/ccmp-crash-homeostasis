"""
Real-Data Tier-2 Point Test v2 -- FMVSS 126 Population Fix
============================================================
Script: realdata_tier2_fastresult_v2.py  (NEW FILE -- originals NOT modified)

BUG FIX (2026-07-14):
  v1 scripts (realdata_tier2_fastresult.py / realdata_tier2_pointtest.py) applied
  NO vehicle-type filter, allowing motorcycles, large trucks, and buses into the
  analytic sample.

  FMVSS 126 (49 CFR 571.126) applies ONLY to passenger cars, multipurpose passenger
  vehicles (MPVs), trucks, and buses with GVWR <= 10,000 lbs.  Two-wheelers, large
  commercial trucks, and buses are never-takers (Z cannot shift T for them) and
  contaminate the first stage.  This fix restricts the analytic population to
  FMVSS 126 scope via CRSS BODY_TYP codes + GVWR where available.

SECONDARY FIX (2026-07-14):
  Original _match_crss_model_to_api() skipped model names with len < 3, silently
  returning "not_found" for all 2-char CRSS model names (AUDI A4/A6/A3, ACURA TL/RL,
  LEXUS ES/GS/RX/IS, etc.).  This fix lowers the threshold to len >= 2 and rebuilds
  ESC consensus for affected entries using the existing API cache.

PRE-SPECIFICATION AMENDMENT NOTE:
  Population restriction to FMVSS 126 scope is a change from the original
  pre-specification.  Reason: FMVSS 126 applicability was the *intended* scope
  (it defines Z), and the absence of a body-type filter was an implementation error
  discovered on 2026-07-14.  This amendment is recorded here and in the results JSON.

ANALYSIS DEFINITION (same as v1 except population filter):
  Z=0: MY in [2000, 2010]; Z=1: MY >= 2012; MY 2011 excluded
  T: from esc_equipment_list_nhtsa_api_v2.csv (NHTSA 5-Star Safety Ratings API, v2)
  Y: I(INJ_SEV in {1,2,3,4})
  W: CRSS person WEIGHT
  H0: tau=0, alpha=0.05, B=500, seed=42
  Bootstrap: stratified cluster bootstrap (cluster=CASENUM within PSUSTRAT)

POPULATION FILTER:
  CRSS BODY_TYP codes included, excluded, and GVWR-conditional are defined in
  FMVSS126_INCLUDE, FMVSS126_EXCLUDE, GVWR_CONDITIONAL below with CRSS code
  book descriptions and rationale.

FABRICATION PROHIBITION:
  All numbers come from actual code execution.  No manual entry.

GPU: none (CPU-only)

PREREQUISITE:
  - CRSS CSVs extracted to SCRATCHPAD/crss{year}/ (same as v1)
  - NHTSA API cache in data/nhtsa_safetyratings_cache/ (built by v1 pointtest)
"""

from __future__ import annotations

import json
import os
import platform
import socket
import ssl
import subprocess
import sys
import time
import urllib.parse
import urllib.request
import urllib.error
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd
import scipy.stats

sys.stdout.reconfigure(line_buffering=True)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR    = Path(__file__).resolve().parents[1]
DATA_DIR    = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

ESC_API_CSV_V1 = DATA_DIR / "esc_equipment_list_nhtsa_api.csv"    # original (read-only)
ESC_API_CSV_V2 = DATA_DIR / "esc_equipment_list_nhtsa_api_v2.csv"  # v2 (write)

CACHE_DIR          = DATA_DIR / "nhtsa_safetyratings_cache"
CACHE_VEHICLES_DIR = CACHE_DIR / "vehicles"
CACHE_MAKE_YEAR_DIR= CACHE_DIR / "make_year"
CACHE_VID_LIST_DIR = CACHE_DIR / "vid_lists"
CACHE_INDEX_DIR    = CACHE_DIR / "index"

SCRATCHPAD = Path("/private/tmp/claude-501/-Users-redacted-user/"
                  "373ef59a-33a6-4195-ad55-30c70723863d/scratchpad")

CRSS_YEARS = [2016, 2017, 2018, 2019, 2020, 2021]

# Pre-specified constants (identical to v1)
MY_MIN_COVERAGE    = 2000
Z_PRE_ESC_MAX_MY   = 2010
Z_POST_ESC_MIN_MY  = 2012
Z_TRANSITION_MY    = 2011
Z_PRE_ESC_MAX_MY_WIDE = 2009

INJ_SEV_VALID  = {0, 1, 2, 3, 4}
INJ_SEV_INJURY = {1, 2, 3, 4}

ALPHA_LEVEL    = 0.05
N_BOOTSTRAP    = 500
BOOTSTRAP_SEED = 42

NHTSA_API_BASE = "https://api.nhtsa.gov/SafetyRatings"
API_SLEEP_SEC  = 0.15
API_TIMEOUT    = 20

import re as _re
_SAFE_CHARS = _re.compile(r'[/\\:*?"<>|]')

def _safe_name(s: str) -> str:
    return _SAFE_CHARS.sub('_', s).replace(' ', '_')


# ---------------------------------------------------------------------------
# FMVSS 126 Population Filter -- BODY_TYP Classification
# (Reference: 49 CFR 571.126; CRSS Analytical User's Manual)
# ---------------------------------------------------------------------------

# BODY_TYP codes definitively in scope for FMVSS 126 (GVWR <= 10,000 lbs implied)
FMVSS126_INCLUDE: Set[int] = frozenset({
    # Passenger cars (automobiles)
    1,   # Convertible (excludes sun-roof, t-bar)
    2,   # 2-door sedan, hardtop, coupe
    3,   # 3-door / 2-door hatchback
    4,   # 4-door sedan, hardtop
    5,   # 5-door / 4-door hatchback
    6,   # Station Wagon (excluding van and truck based)
    7,   # Hatchback, number of doors unknown
    8,   # Sedan / Hardtop, number of doors unknown
    9,   # Other or Unknown automobile type
    10,  # Auto-based pickup (El Camino, Caballero, Ranchero, SSR, G8-ST, Subaru Brat)
    11,  # Auto-based panel (cargo station wagon, auto-based ambulance or hearse)
    12,  # Large Limousine (>4 side doors or stretched chassis) -- car-based, typically <=10k lbs
    17,  # 3-door coupe
    # SUV / Utility vehicles (all categories here are <=10k lbs by CRSS definition)
    14,  # Compact Utility (SUV "Small" and "Midsize" categories)
    15,  # Large utility (SUV "Full Size" and "Large" categories -- GVWR dist confirms <=10k dominant)
    16,  # Utility station wagon (Suburban, Travelall, Grand Wagoneer)
    19,  # Utility Vehicle, Unknown body type
    # Light vans (GVWR <=10k confirmed by code name or GVWR distribution)
    20,  # Minivan (Town & Country, Caravan, Grand Caravan, Voyager, Odyssey, Sienna, Sedona)
    22,  # Step-van or walk-in van with GVWR <= 10,000 lbs (explicitly in CRSS code name)
    # Light trucks / pickups
    34,  # Light Pickup
    39,  # Unknown (pickup style) light conventional truck type -- "light" implies <=10k
    42,  # Light Truck Based Motorhome (chassis mounted) -- "light" implies <=10k
    45,  # Other light conventional truck type -- "light" implies <=10k
    48,  # Unknown light truck type -- "light" implies <=10k
    # Unknown light vehicles (catch-all; GVWR distribution shows <=10k dominant)
    49,  # Unknown light vehicle type (automobile, utility vehicle, van, or light truck)
})

# BODY_TYP codes conditionally included: GVWR <= 10,000 lbs must be confirmed from GVWR field.
# If GVWR is unknown/not-reported, EXCLUDE (conservative).
GVWR_CONDITIONAL: Set[int] = frozenset({
    21,  # Large Van (B150-B350, E150-E350, Sportsman, Royal Maxiwagon, Ram, Tradesman) --
         # Lighter models (E150, E250 standard) may be <=10k lbs; E350/E450/E550 exceed 10k.
         # GVWR distribution: 2113 confirmed <=10k, 21 confirmed >10k, 3734 unknown.
         # Strategy: include only when GVWR_NORM == 1 (<=10k lbs confirmed).
    40,  # Cab Chassis Based (includes Rescue Vehicle, Light Stake, Dump, Tow Truck) --
         # May include heavy commercial vehicles; GVWR distribution uncertain (mostly unknown).
         # Strategy: include only when GVWR_NORM == 1 (<=10k lbs confirmed).
})

# BODY_TYP codes definitively OUT of FMVSS 126 scope
# (not needed to define explicitly, but listed for documentation)
FMVSS126_EXCLUDE: Set[int] = frozenset({
    # Vans > 10,000 lbs or undefined classification
    28,  # Other van type (Hi-Cube Van, Kary) -- large commercial vans
    29,  # Unknown van type -- conservative exclusion
    # Buses (all sizes, all types)
    50,  # School Bus
    51,  # Cross Country / Intercity Bus
    52,  # Transit Bus (City Bus)
    55,  # Van-Based Bus GVWR > 10,000 lbs (explicitly stated in name)
    58,  # Other Bus Type
    59,  # Unknown Bus Type
    # Medium and heavy trucks (all GVWR ranges)
    60,  # Step van (GVWR > 10,000 lbs -- explicitly stated in name)
    61,  # Single-unit straight truck (GVWR 10,001-19,500 lbs)
    62,  # Single-unit straight truck (GVWR 19,501-26,000 lbs)
    63,  # Single-unit straight truck (GVWR > 26,000 lbs)
    64,  # Single-unit straight truck (GVWR unknown) -- conservative
    65,  # Medium/Heavy Vehicle Based Motor Home
    66,  # Truck-tractor (Cab only, or with trailing units)
    67,  # Medium/heavy Pickup (GVWR > 10,000 lbs -- explicitly stated in name)
    71,  # Unknown if single-unit or combination (GVWR 10k-26k)
    72,  # Unknown if single-unit or combination (GVWR > 26k)
    73,  # Camper or motorhome, unknown truck type
    78,  # Unknown medium/heavy truck type
    79,  # Unknown truck type (light/medium/heavy)
    # Motorcycles, mopeds, motor scooters, ATVs (not motor vehicles in FMVSS 126 sense)
    80,  # Two Wheel Motorcycle (excluding motor scooters)
    81,  # Moped or motorized bicycle
    82,  # Three-wheel Motorcycle (2 Rear Wheels)
    83,  # Off-road Motorcycle
    84,  # Motor Scooter
    85,  # Unenclosed Three Wheel Motorcycle / Autocycle (1 Rear Wheel)
    86,  # Enclosed Three Wheel Motorcycle / Autocycle (1 Rear Wheel)
    87,  # Unknown Three Wheel Motorcycle Type
    88,  # Other motored cycle type (mini-bikes, pocket bikes)
    89,  # Unknown motored cycle type
    # Other non-FMVSS 126 vehicles
    90,  # ATV / ATC (All-Terrain Cycle)
    91,  # Snowmobile
    92,  # Farm equipment other than trucks
    93,  # Construction equipment other than trucks
    94,  # Low Speed Vehicle (LSV) / Neighborhood Electric Vehicle (NEV)
    95,  # Golf Cart
    96,  # Recreational Off-Highway Vehicle
    97,  # Other vehicle type (go-cart, fork-lift, street sweeper, dune buggy)
    # Undefined / unclassifiable (conservative exclusion)
    13,  # Code not documented in available codebook (N=1)
    30,  # Code not documented in available codebook (N=2065, GVWR mostly unknown)
    31,  # Code not documented in available codebook (N=7492, some >10k GVWR)
    32,  # Code not documented in available codebook (N=6, GVWR unknown)
    41,  # Code not documented in available codebook (N=1)
    # Unknown / not reported (cannot classify)
    98,  # Not Reported
    99,  # Unknown body type
})


def _normalize_gvwr(df_veh: pd.DataFrame) -> pd.Series:
    """
    Harmonize GVWR coding across CRSS years into a 3-category code.

    2016-2019 GVWR field:
      0 = Not Applicable
      1 = <= 10,000 lbs  (FMVSS 126 in scope)
      2 = 10,001-26,000 lbs
      3 = > 26,000 lbs
      8 = Not Reported
      9 = Reported as Unknown

    2020-2021 GVWR_FROM field:
      11 = Class 1: <= 6,000 lbs
      12 = Class 2: 6,001-10,000 lbs
      13 = Class 3: 10,001-14,000 lbs
      14 = Class 4: 14,001-16,000 lbs
      15 = Class 5: 16,001-19,500 lbs
      16 = Class 6: 19,501-26,000 lbs
      17 = Class 7: 26,001-33,000 lbs
      18 = Class 8: >= 33,001 lbs
      98 = Not Reported
      99 = Reported as Unknown

    Returns Series with harmonized codes:
      1 = GVWR <= 10,000 lbs (FMVSS 126 scope confirmed)
      2 = GVWR > 10,000 lbs, <= 26,000 lbs
      3 = GVWR > 26,000 lbs
      0 = Not Applicable
      8 = Not Reported / Unknown (conservative: treated as out-of-scope for conditional codes)
    """
    if "GVWR_FROM" in df_veh.columns:
        # 2020-2021 format
        mapping = {11: 1, 12: 1, 13: 2, 14: 2, 15: 2, 16: 2, 17: 3, 18: 3, 98: 8, 99: 8}
        return df_veh["GVWR_FROM"].map(mapping).fillna(8).astype(int)
    elif "GVWR" in df_veh.columns:
        # 2016-2019 format: 1=<=10k, 2=10k-26k, 3=>26k, 0=NA, 8=NR, 9=Unk
        return df_veh["GVWR"].fillna(8).replace({9: 8}).astype(int)
    else:
        return pd.Series(8, index=df_veh.index, dtype=int)


def is_fmvss126_scope(body_typ: pd.Series, gvwr_norm: pd.Series) -> pd.Series:
    """
    Return boolean Series: True if vehicle is in scope for FMVSS 126.

    Logic:
      - Codes in FMVSS126_INCLUDE: always in scope
      - Codes in GVWR_CONDITIONAL: in scope only if GVWR_NORM == 1 (confirmed <=10k)
      - All other codes: out of scope
    """
    in_scope = body_typ.isin(FMVSS126_INCLUDE)
    conditional = body_typ.isin(GVWR_CONDITIONAL) & (gvwr_norm == 1)
    return in_scope | conditional


# ---------------------------------------------------------------------------
# NHTSA 5-Star Safety Ratings API helpers (verbatim from v1, with one fix)
# ---------------------------------------------------------------------------

def _ssl_context() -> ssl.SSLContext:
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
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "CCMP-Research/1.0"})
        with urllib.request.urlopen(req, timeout=API_TIMEOUT, context=_SSL_CTX) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        print(f"  [API ERROR] {url}: {e}", file=sys.stderr)
        return None


def get_vehicle_esc_cached(vehicle_id: int) -> Optional[str]:
    """Return cached ESC value for a VehicleId. Queries API if not cached."""
    cache_path = CACHE_VEHICLES_DIR / f"{vehicle_id}.json"
    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f).get("esc_value")
    data = _api_get(f"{NHTSA_API_BASE}/VehicleId/{vehicle_id}")
    time.sleep(API_SLEEP_SEC)
    if data is None or data.get("Count", 0) == 0:
        result = {"vehicle_id": vehicle_id, "esc_value": None,
                  "fetched": datetime.utcnow().isoformat()}
        with open(cache_path, "w") as f:
            json.dump(result, f)
        return None
    r = data["Results"][0]
    esc = r.get("NHTSAElectronicStabilityControl", None)
    if esc and "standard" in esc.lower():
        esc = "Standard"
    elif esc and esc.strip().lower() == "no":
        esc = "No"
    elif esc and "optional" in esc.lower():
        esc = "Optional"
    else:
        esc = None
    result = {"vehicle_id": vehicle_id, "esc_value": esc,
              "description": r.get("VehicleDescription", ""),
              "fetched": datetime.utcnow().isoformat()}
    with open(cache_path, "w") as f:
        json.dump(result, f)
    return esc


def _get_api_models_cached(year: int, make_upper: str) -> Optional[List[str]]:
    """Return model list from Level-1 cache (make, year). None if make not in API."""
    cache_path = CACHE_MAKE_YEAR_DIR / f"{year}_{_safe_name(make_upper)}.json"
    if not cache_path.exists():
        return None
    with open(cache_path) as f:
        return json.load(f).get("models")


def _get_vehicle_ids_cached(year: int, make_upper: str, api_model: str) -> List[int]:
    """Return VehicleIds from Level-2 cache. Queries API if cache miss."""
    cache_path = CACHE_VID_LIST_DIR / (
        f"{year}_{_safe_name(make_upper)}_{_safe_name(api_model)}.json"
    )
    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f).get("vehicle_ids", [])
    make_enc  = urllib.parse.quote(make_upper, safe="")
    model_enc = urllib.parse.quote(api_model,  safe="")
    url  = f"{NHTSA_API_BASE}/modelyear/{year}/make/{make_enc}/model/{model_enc}"
    data = _api_get(url)
    time.sleep(API_SLEEP_SEC)
    if data is None or data.get("Count", 0) == 0:
        vids = []
    else:
        vids = [r["VehicleId"] for r in data["Results"] if "VehicleId" in r]
    result = {"year": year, "make": make_upper, "api_model": api_model,
              "vehicle_ids": vids, "fetched": datetime.utcnow().isoformat()}
    with open(cache_path, "w") as f:
        json.dump(result, f)
    return vids


def _esc_consensus(esc_values: List[Optional[str]]) -> str:
    valid = [v for v in esc_values if v is not None]
    if not valid:
        return "not_found"
    unique = set(valid)
    if "Optional" in unique:
        return "Optional"
    if len(unique) == 1:
        return list(unique)[0]
    return "ambiguous"


def _match_model_v2(crss_model: str, api_models: List[str]) -> List[str]:
    """
    v2: model name matching with min length = 2 (fixed from v1's min=3).

    This rescues 2-char model names like AUDI A4, A6; ACURA TL, RL;
    LEXUS ES, GS, IS, RX; BMW X3, X5, Z4; etc.

    Matching strategy:
      1. Forward prefix (word boundary): "A4" matches "A4", "A4 AVANT", "A4 CABRIOLET"
      2. Reverse prefix (word boundary): "A4 CABRIOLET" matches "A4" if no forward hit
    """
    crss_u = crss_model.upper().strip()
    if not crss_u or len(crss_u) < 2:   # v2: min length 2 (v1 was 3)
        return []
    forward = [m for m in api_models if m == crss_u or m.startswith(crss_u + " ")]
    if forward:
        return sorted(set(forward))
    reverse = [m for m in api_models if crss_u == m or crss_u.startswith(m + " ")]
    if reverse:
        return [max(reverse, key=len)]
    return []


def _lookup_esc_v2(year: int, make: str, model: str) -> Dict:
    """
    ESC lookup for (year, make, model) using existing cache + v2 matching.

    Does NOT use the index cache (which bakes in v1's buggy matching results).
    Always re-runs matching against Level-1 (make, year) model list.

    Returns dict with esc_consensus, matched_api_models, vehicle_ids, esc_values.
    """
    make_u  = make.upper().strip()
    model_u = model.upper().strip()

    # Level-1: get all models for (make, year)
    api_models = _get_api_models_cached(year, make_u)
    if api_models is None:
        return {"esc_consensus": "not_found", "matched_api_models": [],
                "vehicle_ids": [], "esc_values": []}

    # v2 matching (min len 2)
    matched = _match_model_v2(model_u, api_models)
    if not matched:
        return {"esc_consensus": "not_found", "matched_api_models": [],
                "vehicle_ids": [], "esc_values": []}

    # Level-2: get VehicleIds for each matched model
    all_vids: List[int] = []
    for api_model in matched:
        vids = _get_vehicle_ids_cached(year, make_u, api_model)
        all_vids.extend(vids)
    all_vids = list(set(all_vids))

    # Level-3: get ESC for each VehicleId
    esc_values = [get_vehicle_esc_cached(vid) for vid in all_vids]
    consensus  = _esc_consensus(esc_values)

    return {
        "esc_consensus": consensus,
        "matched_api_models": matched,
        "vehicle_ids": all_vids,
        "esc_values": esc_values,
    }


def build_esc_cache_v2(df_vehicles_premandate: pd.DataFrame) -> pd.DataFrame:
    """
    Build/update ESC cache using v2 matching.

    Steps:
      1. Load existing v1 ESC API CSV
      2. For entries with not_found AND model_len <= 2 (v1 matching bug),
         re-run _lookup_esc_v2() to attempt rescue
      3. For any (make, model, year) in the passenger-car-filtered CRSS data
         that is missing from v1 cache, run _lookup_esc_v2()
      4. Save result as ESC_API_CSV_V2 (new file, does not overwrite v1)

    Returns DataFrame with columns: make, model, year, esc_consensus, T_assign
    """
    print("  Loading v1 ESC API CSV...", flush=True)
    df_v1 = pd.read_csv(ESC_API_CSV_V1)
    df_v1.columns = [c.lower() for c in df_v1.columns]
    print(f"  v1 entries: {len(df_v1):,}", flush=True)

    v1_lookup = set(
        zip(df_v1["make"].str.upper().str.strip(),
            df_v1["model"].str.upper().str.strip(),
            df_v1["year"].astype(int))
    )

    rows: List[Dict] = []
    rescued = 0
    carried = 0
    new_entries = 0

    # Unique (make, model, year) in the pre-mandate passenger-car data
    pre = df_vehicles_premandate[
        ["MAKE_API", "MODEL_API", "MOD_YEAR"]
    ].drop_duplicates()
    pre = pre[pre["MAKE_API"].notna() & pre["MODEL_API"].notna()]

    total = len(pre)
    print(f"  Rebuilding ESC cache for {total} unique (make, model, year) combos "
          f"in passenger-car pre-mandate data...", flush=True)

    for i, (_, row) in enumerate(pre.iterrows()):
        make  = str(row["MAKE_API"]).upper().strip()
        model = str(row["MODEL_API"]).upper().strip()
        year  = int(row["MOD_YEAR"])

        if not make or make in ("NAN", "UNKNOWN", "") or not model or model in ("NAN",):
            rows.append({"make": make, "model": model, "year": year,
                         "esc_consensus": "not_found", "T_assign": np.nan,
                         "v2_action": "skipped_empty"})
            continue

        key = (make, model, year)

        # Check v1 result
        v1_row = df_v1[(df_v1["make"].str.upper().str.strip() == make) &
                        (df_v1["model"].str.upper().str.strip() == model) &
                        (df_v1["year"].astype(int) == year)]

        if v1_row.empty:
            # New entry (vehicle appeared in passenger-car data but not in v1 cache)
            res = _lookup_esc_v2(year, make, model)
            cons = res["esc_consensus"]
            action = "new"
            new_entries += 1
        else:
            v1_cons = str(v1_row.iloc[0]["esc_consensus"])
            model_len = len(model)

            if v1_cons == "not_found" and model_len <= 2:
                # Potentially caused by v1 matching bug (len < 3 skip)
                res = _lookup_esc_v2(year, make, model)
                cons = res["esc_consensus"]
                action = "rescued" if cons != "not_found" else "rescue_failed"
                if cons != "not_found":
                    rescued += 1
            else:
                cons = v1_cons
                action = "carried"
                carried += 1

        t_assign = 1.0 if cons == "Standard" else (0.0 if cons == "No" else np.nan)
        rows.append({"make": make, "model": model, "year": year,
                     "esc_consensus": cons, "T_assign": t_assign,
                     "v2_action": action})

        if (i + 1) % 200 == 0 or i == 0:
            print(f"    ... {i+1}/{total} processed "
                  f"(rescued={rescued}, new={new_entries})", flush=True)

    df_v2 = pd.DataFrame(rows)
    df_v2.to_csv(ESC_API_CSV_V2, index=False)
    print(f"  Saved v2 ESC cache: {ESC_API_CSV_V2}", flush=True)
    print(f"  Summary: carried={carried}, rescued={rescued}, "
          f"new={new_entries}, total={len(df_v2)}", flush=True)
    print(f"  esc_consensus v2: {df_v2['esc_consensus'].value_counts().to_dict()}", flush=True)
    return df_v2


# ---------------------------------------------------------------------------
# CRSS Data Loading (v2: includes BODY_TYP + GVWR, applies population filter)
# ---------------------------------------------------------------------------

def _find_csv(d: Path, name: str) -> Optional[Path]:
    for p in d.iterdir():
        if p.is_file() and p.name.upper() == name.upper():
            return p
    for sub in d.iterdir():
        if sub.is_dir():
            for p in sub.iterdir():
                if p.is_file() and p.name.upper() == name.upper():
                    return p
    return None


def load_crss_year_v2(year: int) -> Optional[pd.DataFrame]:
    """
    Load one year of CRSS data with FMVSS 126 population filter applied.

    Changes from v1:
    - Loads BODY_TYP and GVWR (or GVWR_FROM) from vehicle.csv
    - Applies is_fmvss126_scope() filter BEFORE merging with person.csv
    - Reports excluded vehicle counts by BODY_TYP

    Returns merged (accident + vehicle + person + vpicdecode) DataFrame
    restricted to FMVSS 126 applicable vehicles.
    """
    crss_dir = SCRATCHPAD / f"crss{year}"
    if not crss_dir.exists():
        print(f"  CRSS {year} directory not found", flush=True)
        return None

    acc_path = _find_csv(crss_dir, "accident.csv")
    veh_path = _find_csv(crss_dir, "vehicle.csv")
    per_path = _find_csv(crss_dir, "person.csv")
    vpc_path = _find_csv(crss_dir, "vpicdecode.csv")

    missing = [n for n, p in [("accident", acc_path), ("vehicle", veh_path),
                               ("person", per_path), ("vpicdecode", vpc_path)]
               if p is None]
    if missing:
        print(f"  CRSS {year} missing: {missing}", flush=True)
        return None

    print(f"  Loading CRSS {year}...", flush=True)
    try:
        df_acc = pd.read_csv(str(acc_path), encoding="latin-1", low_memory=False)
        df_veh = pd.read_csv(str(veh_path), encoding="latin-1", low_memory=False)
        df_per = pd.read_csv(str(per_path), encoding="latin-1", low_memory=False)
        df_vpc = pd.read_csv(str(vpc_path), encoding="latin-1", low_memory=False)
    except Exception as e:
        print(f"  [ERROR] {e}", flush=True)
        return None

    for d in [df_acc, df_veh, df_per, df_vpc]:
        d.columns = [c.upper() for c in d.columns]

    # ------------------------------------------------------------------
    # v2 ADDITION: BODY_TYP and GVWR from vehicle.csv
    # ------------------------------------------------------------------
    df_veh["GVWR_NORM"] = _normalize_gvwr(df_veh)
    if "BODY_TYP" not in df_veh.columns:
        print(f"  [WARN] CRSS {year}: BODY_TYP column missing -- "
              f"no population filter applied for this year", flush=True)
        df_veh["BODY_TYP"] = np.nan
        scope_mask = pd.Series(True, index=df_veh.index)
    else:
        df_veh["BODY_TYP"] = pd.to_numeric(df_veh["BODY_TYP"], errors="coerce")
        scope_mask = is_fmvss126_scope(df_veh["BODY_TYP"], df_veh["GVWR_NORM"])

    n_veh_total   = len(df_veh)
    n_veh_in      = int(scope_mask.sum())
    n_veh_out     = n_veh_total - n_veh_in
    pct_out       = 100.0 * n_veh_out / max(n_veh_total, 1)

    # Distribution of excluded BODY_TYP codes
    if "BODY_TYP" in df_veh.columns and "BODY_TYPNAME" in df_veh.columns:
        excl_breakdown = (df_veh[~scope_mask]
                          .groupby(["BODY_TYP", "BODY_TYPNAME"])
                          .size()
                          .sort_values(ascending=False)
                          .head(20)
                          .to_dict())
    else:
        excl_breakdown = (df_veh[~scope_mask]["BODY_TYP"]
                          .value_counts()
                          .head(20)
                          .to_dict() if "BODY_TYP" in df_veh.columns else {})

    print(f"  CRSS {year}: {n_veh_total:,} vehicles total; "
          f"in-scope {n_veh_in:,} ({100-pct_out:.1f}%), "
          f"excluded {n_veh_out:,} ({pct_out:.1f}%)", flush=True)

    df_veh_filtered = df_veh[scope_mask].copy()

    # ------------------------------------------------------------------
    # Standard column selection (same as v1 except added BODY_TYP/GVWR)
    # ------------------------------------------------------------------
    acc_cols = [c for c in ["CASENUM", "PSU", "PSUSTRAT", "STRATUM",
                             "WEIGHT", "REGION", "URBANICITY", "INT_HWY"]
                if c in df_acc.columns]
    df_acc = df_acc[acc_cols].rename(columns={"PSU": "PSU_ACC",
                                               "PSUSTRAT": "PSUSTRAT_ACC",
                                               "WEIGHT": "WEIGHT_ACC"})

    veh_cols = [c for c in ["CASENUM", "VEH_NO", "MOD_YEAR", "BODY_TYP", "GVWR_NORM"]
                if c in df_veh_filtered.columns]
    df_veh_filtered = df_veh_filtered[veh_cols]

    per_cols = [c for c in ["CASENUM", "VEH_NO", "INJ_SEV", "WEIGHT"]
                if c in df_per.columns]
    df_per = df_per[per_cols].rename(columns={"WEIGHT": "WEIGHT_PER"})

    vpc_cols = ["CASENUM", "VEH_NO"]
    if "MAKE" in df_vpc.columns:
        vpc_cols.append("MAKE")
    if "MODEL" in df_vpc.columns:
        vpc_cols.append("MODEL")
    df_vpc = df_vpc[vpc_cols].rename(columns={"MAKE": "MAKE_API", "MODEL": "MODEL_API"})

    # Merge (same structure as v1, but with filtered vehicle table)
    df = pd.merge(df_per, df_veh_filtered, on=["CASENUM", "VEH_NO"], how="inner")
    df = pd.merge(df, df_acc, on="CASENUM", how="left")
    df = pd.merge(df, df_vpc, on=["CASENUM", "VEH_NO"], how="left")
    df["CRSS_YEAR"] = year

    for col in ["MOD_YEAR", "INJ_SEV", "WEIGHT_PER", "WEIGHT_ACC", "BODY_TYP"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    print(f"  CRSS {year}: {len(df):,} person-vehicle records (post-filter)", flush=True)

    # Attach exclusion stats for reporting
    df._excl_by_bodytyp = excl_breakdown
    df._n_veh_total = n_veh_total
    df._n_veh_out   = n_veh_out
    return df


# ---------------------------------------------------------------------------
# Treatment, Instrument, Outcome Assignment (identical to v1)
# ---------------------------------------------------------------------------

def assign_variables(df: pd.DataFrame, esc_cache: pd.DataFrame) -> pd.DataFrame:
    """Assign Z, T, Y (identical logic to v1)."""
    df = df.copy()
    my = df["MOD_YEAR"]
    df["Z"] = np.nan
    df.loc[(my >= MY_MIN_COVERAGE) & (my <= Z_PRE_ESC_MAX_MY), "Z"] = 0.0
    df.loc[my >= Z_POST_ESC_MIN_MY, "Z"] = 1.0
    df["Z_wide"] = np.nan
    df.loc[(my >= MY_MIN_COVERAGE) & (my <= Z_PRE_ESC_MAX_MY_WIDE), "Z_wide"] = 0.0
    df.loc[my >= Z_POST_ESC_MIN_MY, "Z_wide"] = 1.0

    df["T"] = np.nan
    df.loc[df["Z"] == 1, "T"] = 1.0  # Post-mandate: T=1 by FMVSS 126

    if "MAKE_API" in df.columns and "MODEL_API" in df.columns:
        df["_make_up"]  = df["MAKE_API"].fillna("").astype(str).str.upper().str.strip()
        df["_model_up"] = df["MODEL_API"].fillna("").astype(str).str.upper().str.strip()
        df["_year"]     = df["MOD_YEAR"].fillna(0).astype(int)
        esc_lu = esc_cache.copy()
        esc_lu["make"]  = esc_lu["make"].fillna("").astype(str).str.upper().str.strip()
        esc_lu["model"] = esc_lu["model"].fillna("").astype(str).str.upper().str.strip()
        esc_lu["year"]  = esc_lu["year"].astype(int)
        pre_mask = (df["Z"] == 0.0)
        df_pre   = df.loc[pre_mask, ["_make_up", "_model_up", "_year"]].copy()
        df_pre   = df_pre.merge(
            esc_lu[["make", "model", "year", "T_assign"]].rename(
                columns={"make": "_make_up", "model": "_model_up", "year": "_year"}),
            on=["_make_up", "_model_up", "_year"], how="left")
        df.loc[pre_mask, "T"] = df_pre["T_assign"].values
        df.drop(columns=["_make_up", "_model_up", "_year"], inplace=True)

    df["Y_binary"] = np.nan
    valid_mask = df["INJ_SEV"].isin(INJ_SEV_VALID)
    df.loc[valid_mask, "Y_binary"] = (
        df.loc[valid_mask, "INJ_SEV"].isin(INJ_SEV_INJURY)
    ).astype(float)
    df["W"] = df["WEIGHT_PER"].fillna(0.0)
    df.loc[df["W"] <= 0, "W"] = np.nan
    return df


# ---------------------------------------------------------------------------
# IV Wald Estimator (line-for-line match with v1 _hajek_wald)
# ---------------------------------------------------------------------------

def _hajek_wald(Y: np.ndarray, T: np.ndarray, Z: np.ndarray, W: np.ndarray) -> Dict:
    m1 = (Z == 1); m0 = (Z == 0)
    n1, n0 = int(m1.sum()), int(m0.sum())
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
    resY_Z1 = Y[m1] - EY_Z1; resY_Z0 = Y[m0] - EY_Z0
    resT_Z1 = T[m1] - ET_Z1; resT_Z0 = T[m0] - ET_Z0
    var_RF = (np.dot(w1**2, resY_Z1**2) / sw1**2 +
              np.dot(w0**2, resY_Z0**2) / sw0**2)
    var_FS = (np.dot(w1**2, resT_Z1**2) / sw1**2 +
              np.dot(w0**2, resT_Z0**2) / sw0**2)
    SE_tau = float(np.sqrt(max(1e-20, var_RF / FS**2 + RF**2 * var_FS / FS**4)))
    z_stat = tau / SE_tau
    p_val  = float(2.0 * scipy.stats.norm.sf(abs(z_stat)))
    return {
        "tau": float(tau), "SE_tau": SE_tau,
        "CI_L": float(tau - 1.96 * SE_tau), "CI_U": float(tau + 1.96 * SE_tau),
        "z_stat": float(z_stat), "p_value": p_val,
        "reject": bool(p_val < ALPHA_LEVEL),
        "FS": float(FS), "RF": float(RF),
        "EY_Z1": float(EY_Z1), "EY_Z0": float(EY_Z0),
        "ET_Z1": float(ET_Z1), "ET_Z0": float(ET_Z0),
        "n_Z0": n0, "n_Z1": n1,
    }


def _first_stage_fstat(T: np.ndarray, Z: np.ndarray, W: np.ndarray) -> Dict:
    m1 = (Z == 1); m0 = (Z == 0)
    w1, w0 = W[m1], W[m0]
    sw1, sw0 = w1.sum(), w0.sum()
    if sw1 < 1e-10 or sw0 < 1e-10:
        return {"FS": float("nan"), "SE_FS": float("nan"), "F_stat": float("nan")}
    ET_Z1 = float(np.dot(T[m1], w1) / sw1)
    ET_Z0 = float(np.dot(T[m0], w0) / sw0)
    FS    = ET_Z1 - ET_Z0
    resT_Z1 = T[m1] - ET_Z1; resT_Z0 = T[m0] - ET_Z0
    var_FS  = (np.dot(w1**2, resT_Z1**2) / sw1**2 +
               np.dot(w0**2, resT_Z0**2) / sw0**2)
    SE_FS   = float(np.sqrt(max(1e-20, var_FS)))
    F_stat  = float((FS / SE_FS)**2) if SE_FS > 1e-10 else float("inf")
    return {"FS": float(FS), "SE_FS": SE_FS, "F_stat": F_stat,
            "ET_Z0": float(ET_Z0), "ET_Z1": float(ET_Z1)}


# ---------------------------------------------------------------------------
# Bootstrap CI (numpy-based, identical scheme to v1)
# ---------------------------------------------------------------------------

def bootstrap_iv_wald_np(
    Y: np.ndarray, T: np.ndarray, Z: np.ndarray, W: np.ndarray,
    caseid: np.ndarray, stratid: np.ndarray,
    B: int = N_BOOTSTRAP, seed: int = BOOTSTRAP_SEED,
) -> Dict:
    rng = np.random.default_rng(seed)
    unique_strats = np.unique(stratid)
    strat_cluster_obs: Dict = {}
    for s in unique_strats:
        s_mask   = (stratid == s)
        s_idx    = np.where(s_mask)[0]
        s_cases  = caseid[s_mask]
        unique_c = np.unique(s_cases)
        strat_cluster_obs[s] = {c: s_idx[s_cases == c] for c in unique_c}

    bs_taus = []
    for b in range(B):
        resampled_idx = []
        for s, cluster_map in strat_cluster_obs.items():
            clusters = list(cluster_map.keys())
            sampled  = rng.choice(clusters, size=len(clusters), replace=True)
            for c in sampled:
                resampled_idx.append(cluster_map[c])
        if not resampled_idx:
            continue
        idx  = np.concatenate(resampled_idx)
        res  = _hajek_wald(Y[idx], T[idx], Z[idx], W[idx])
        if not np.isnan(res["tau"]):
            bs_taus.append(res["tau"])
        if (b + 1) % 100 == 0:
            print(f"  Bootstrap: {b+1}/{B} done", flush=True)

    if len(bs_taus) < 10:
        return {"CI_L_boot": float("nan"), "CI_U_boot": float("nan"),
                "SE_boot": float("nan"), "n_boot_valid": len(bs_taus)}
    arr = np.array(bs_taus)
    return {"CI_L_boot": float(np.percentile(arr, 2.5)),
            "CI_U_boot": float(np.percentile(arr, 97.5)),
            "SE_boot": float(arr.std(ddof=1)),
            "n_boot_valid": len(bs_taus)}


def assess_power(tau_hat: float, se_hat: float) -> Dict:
    if np.isnan(tau_hat) or np.isnan(se_hat) or se_hat <= 0:
        return {"power_at_obs_tau": float("nan"), "mde_80pct": float("nan"),
                "power_at_reference_tau": {}}
    z_alpha = scipy.stats.norm.ppf(1 - ALPHA_LEVEL / 2)
    z_beta  = scipy.stats.norm.ppf(0.80)
    ncp     = abs(tau_hat) / se_hat
    power   = float(1 - scipy.stats.norm.cdf(z_alpha - ncp) +
                    scipy.stats.norm.cdf(-z_alpha - ncp))
    mde     = float((z_alpha + z_beta) * se_hat)
    ref = {}
    for t in [0.015, 0.030, 0.060]:
        ncp2 = t / se_hat
        ref[f"|tau|={t:.3f}"] = round(float(
            1 - scipy.stats.norm.cdf(z_alpha - ncp2) +
            scipy.stats.norm.cdf(-z_alpha - ncp2)), 3)
    return {"power_at_obs_tau": round(power, 3), "mde_80pct": round(mde, 4),
            "power_at_reference_tau": ref}


def get_env_log(n_raw: int, n_analytic: int) -> Dict:
    try:
        git = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(BASE_DIR), stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        git = "unavailable"
    return {
        "timestamp"      : datetime.utcnow().isoformat() + "Z",
        "script"         : "realdata_tier2_fastresult_v2.py",
        "python_version" : sys.version,
        "numpy_version"  : np.__version__,
        "pandas_version" : pd.__version__,
        "platform"       : platform.platform(),
        "hostname"       : socket.gethostname(),
        "git_commit"     : git,
        "GPU"            : "none (CPU-only)",
        "crss_years"     : CRSS_YEARS,
        "n_raw"          : n_raw,
        "n_analytic"     : n_analytic,
        "T_source"       : "esc_equipment_list_nhtsa_api_v2.csv (NHTSA 5-Star Safety Ratings API, v2 matching)",
        "analysis_version": "v2",
        "v2_changes"     : [
            "Population restricted to FMVSS 126 scope via BODY_TYP filter (CRSS vehicle.csv)",
            "GVWR-conditional inclusion for BODY_TYP 21 (Large Van) and 40 (Cab Chassis)",
            "_match_crss_model_to_api min_len lowered from 3 to 2 (fixes 2-char model silencing)",
            "ESC cache rebuilt for passenger-car pre-mandate vehicles using v2 matching",
            "ESC API CSV saved as esc_equipment_list_nhtsa_api_v2.csv (v1 not overwritten)",
        ],
        "amendment_note" : (
            "Population restriction is a pre-specification amendment. "
            "Original pre-spec did not apply BODY_TYP filter. "
            "Rationale: FMVSS 126 (Z definition) applies only to GVWR<=10k vehicles. "
            "Motorcycles and large trucks are never-takers (dilute first stage). "
            "This is an implementation error correction, not a post-hoc result change."
        ),
    }


# ---------------------------------------------------------------------------
# Coverage reporting helpers
# ---------------------------------------------------------------------------

def report_esc_cache_passenger_coverage(esc_v2: pd.DataFrame, prefix: str = "") -> Dict:
    """Report esc_consensus breakdown for passenger-car pre-mandate vehicles."""
    cs = esc_v2["esc_consensus"].value_counts().to_dict()
    t1 = int((esc_v2["esc_consensus"] == "Standard").sum())
    t0 = int((esc_v2["esc_consensus"] == "No").sum())
    topt = int((esc_v2["esc_consensus"] == "Optional").sum())
    tamb = int((esc_v2["esc_consensus"] == "ambiguous").sum())
    tnf  = int((esc_v2["esc_consensus"] == "not_found").sum())
    total = len(esc_v2)

    print(f"{prefix}ESC cache consensus (passenger-car pre-mandate):", flush=True)
    print(f"{prefix}  Standard   (T=1): {t1:,} ({100*t1/max(total,1):.1f}%)", flush=True)
    print(f"{prefix}  No         (T=0): {t0:,} ({100*t0/max(total,1):.1f}%)", flush=True)
    print(f"{prefix}  Optional (T=undef): {topt:,} ({100*topt/max(total,1):.1f}%)", flush=True)
    print(f"{prefix}  ambiguous (T=undef): {tamb:,} ({100*tamb/max(total,1):.1f}%)", flush=True)
    print(f"{prefix}  not_found (T=undef): {tnf:,} ({100*tnf/max(total,1):.1f}%)", flush=True)

    # Top not_found makes (passenger cars)
    if tnf > 0:
        nf_df = esc_v2[esc_v2["esc_consensus"] == "not_found"]
        top_makes = nf_df["make"].value_counts().head(15).to_dict()
        top_pairs = nf_df.groupby(["make", "model"]).size().sort_values(ascending=False).head(20).to_dict()
        print(f"{prefix}  not_found top makes: {top_makes}", flush=True)
        print(f"{prefix}  not_found top (make,model) pairs: {top_pairs}", flush=True)
    else:
        top_makes = {}
        top_pairs = {}

    return {"Standard": t1, "No": t0, "Optional": topt, "ambiguous": tamb,
            "not_found": tnf, "total": total,
            "not_found_top_makes": top_makes}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> Dict:
    print("=" * 72, flush=True)
    print("Real-Data Tier-2 Point Test v2 -- FMVSS 126 Population Fix", flush=True)
    print(f"Start: {datetime.utcnow().isoformat()}Z", flush=True)
    print("=" * 72, flush=True)

    print("\n[PRE-SPECIFICATION AMENDMENT]", flush=True)
    print("  v2 adds BODY_TYP filter to restrict analysis to FMVSS 126 scope.", flush=True)
    print("  This is an implementation error correction (not a post-hoc result change).", flush=True)
    print("  See env_log['amendment_note'] for full rationale.", flush=True)
    print(f"\n[BODY_TYP CLASSIFICATION]", flush=True)
    print(f"  FMVSS126_INCLUDE  ({len(FMVSS126_INCLUDE)} codes): "
          f"{sorted(FMVSS126_INCLUDE)}", flush=True)
    print(f"  GVWR_CONDITIONAL  ({len(GVWR_CONDITIONAL)} codes): "
          f"{sorted(GVWR_CONDITIONAL)} (include only if GVWR_NORM=1, i.e. <=10k lbs)", flush=True)
    print(f"  All other BODY_TYP codes: EXCLUDED", flush=True)

    # ------------------------------------------------------------------
    # STEP 1: Load CRSS (with population filter applied inside loader)
    # ------------------------------------------------------------------
    print(f"\n[STEP 1] Load CRSS data (FMVSS 126 population filter applied)", flush=True)
    frames = []
    excl_by_bodytyp_all: Dict = {}
    n_veh_total_all = 0
    n_veh_out_all   = 0

    for year in CRSS_YEARS:
        df_year = load_crss_year_v2(year)
        if df_year is not None:
            # Collect exclusion stats (stored as attribute on DataFrame)
            if hasattr(df_year, "_excl_by_bodytyp"):
                for k, v in df_year._excl_by_bodytyp.items():
                    excl_by_bodytyp_all[k] = excl_by_bodytyp_all.get(k, 0) + v
            if hasattr(df_year, "_n_veh_total"):
                n_veh_total_all += df_year._n_veh_total
            if hasattr(df_year, "_n_veh_out"):
                n_veh_out_all   += df_year._n_veh_out
            frames.append(df_year)

    if not frames:
        raise RuntimeError("No CRSS data loaded.")

    df_raw = pd.concat(frames, ignore_index=True)
    n_raw  = len(df_raw)
    print(f"\n  Pooled (FMVSS 126 scope only): {n_raw:,} person-vehicle records", flush=True)
    print(f"  Total vehicles before filter: {n_veh_total_all:,}", flush=True)
    print(f"  Vehicles excluded (out of FMVSS 126 scope): {n_veh_out_all:,} "
          f"({100*n_veh_out_all/max(n_veh_total_all,1):.1f}%)", flush=True)
    print(f"  Top excluded BODY_TYP codes: {dict(list(excl_by_bodytyp_all.items())[:20])}", flush=True)

    # ------------------------------------------------------------------
    # STEP 2: Build v2 ESC cache (rescue 2-char model names)
    # ------------------------------------------------------------------
    print(f"\n[STEP 2] Build v2 ESC cache", flush=True)
    if ESC_API_CSV_V2.exists():
        print(f"  Loading existing v2 cache: {ESC_API_CSV_V2}", flush=True)
        esc_v2 = pd.read_csv(ESC_API_CSV_V2)
    else:
        # Only build for pre-mandate vehicles in the passenger-car scope
        pre_mask = ((df_raw["MOD_YEAR"] >= MY_MIN_COVERAGE) &
                    (df_raw["MOD_YEAR"] <= Z_PRE_ESC_MAX_MY))
        df_pre = df_raw[pre_mask]
        esc_v2 = build_esc_cache_v2(df_pre)

    esc_v2_summary = report_esc_cache_passenger_coverage(esc_v2, prefix="  ")

    # ------------------------------------------------------------------
    # STEP 3: Assign Z, T, Y
    # ------------------------------------------------------------------
    print(f"\n[STEP 3] Assign Z, T, Y", flush=True)
    df = assign_variables(df_raw, esc_v2)

    z0 = (df["Z"] == 0.0) & df["MOD_YEAR"].notna()
    n_z0  = int(z0.sum())
    n_t1  = int((z0 & (df["T"] == 1.0)).sum())
    n_t0  = int((z0 & (df["T"] == 0.0)).sum())
    n_tna = int((z0 & df["T"].isna()).sum())

    # Breakdown of T=NA: Optional vs not_found vs ambiguous
    z0_veh = df[z0 & df["T"].isna()][["MAKE_API", "MODEL_API", "MOD_YEAR"]].drop_duplicates()
    if len(z0_veh) > 0:
        z0_merged = z0_veh.merge(
            esc_v2.rename(columns={"make": "MAKE_API", "model": "MODEL_API", "year": "MOD_YEAR"}),
            on=["MAKE_API", "MODEL_API", "MOD_YEAR"], how="left"
        )
        tna_breakdown = z0_merged["esc_consensus"].fillna("not_found").value_counts().to_dict()
    else:
        tna_breakdown = {}

    n_z1 = int((df["Z"] == 1.0).sum())

    # Weighted coverage (CRSS design weights)
    w_z0 = df[z0]["W"].fillna(0)
    w_z0_sum  = float(w_z0.sum()) or 1.0
    w_t1 = df[z0 & (df["T"] == 1.0)]["W"].fillna(0).sum()
    w_t0 = df[z0 & (df["T"] == 0.0)]["W"].fillna(0).sum()
    w_tna= df[z0 & df["T"].isna()]["W"].fillna(0).sum()

    print(f"\n  [Coverage: Z=0 pre-mandate group -- PASSENGER CARS ONLY]", flush=True)
    print(f"  Z=0 total persons: {n_z0:,}", flush=True)
    print(f"    T=1 Standard (always-takers):    {n_t1:,} ({100*n_t1/max(n_z0,1):.1f}% unweighted)"
          f"  [{100*w_t1/w_z0_sum:.1f}% weighted]", flush=True)
    print(f"    T=0 No ESC (potential compliers): {n_t0:,} ({100*n_t0/max(n_z0,1):.1f}% unweighted)"
          f"  [{100*w_t0/w_z0_sum:.1f}% weighted]", flush=True)
    print(f"    T=NA Optional (self-selection):  "
          f"{int((z0 & df['T'].isna() & (df.get('esc_v2_Optional', False) if False else df['T'].isna())).sum()):,}",
          flush=True)
    print(f"    T=NA total unassigned:            {n_tna:,} ({100*n_tna/max(n_z0,1):.1f}% unweighted)"
          f"  [{100*w_tna/w_z0_sum:.1f}% weighted]", flush=True)
    print(f"    T=NA breakdown (unique make/model/year): {tna_breakdown}", flush=True)
    print(f"  Z=1 total: {n_z1:,} (T=1 by FMVSS 126 mandate)", flush=True)

    # 4-category breakdown with esc_consensus labels
    # Re-merge esc_consensus for Z=0 rows to get Optional vs not_found split
    df_z0 = df[z0].copy()
    df_z0["_make_up"]  = df_z0["MAKE_API"].fillna("").astype(str).str.upper().str.strip()
    df_z0["_model_up"] = df_z0["MODEL_API"].fillna("").astype(str).str.upper().str.strip()
    df_z0["_year"]     = df_z0["MOD_YEAR"].fillna(0).astype(int)
    esc_lu = esc_v2.copy()
    esc_lu["make"]  = esc_lu["make"].fillna("").astype(str).str.upper().str.strip()
    esc_lu["model"] = esc_lu["model"].fillna("").astype(str).str.upper().str.strip()
    esc_lu["year"]  = esc_lu["year"].astype(int)
    df_z0 = df_z0.merge(
        esc_lu[["make","model","year","esc_consensus"]].rename(
            columns={"make":"_make_up","model":"_model_up","year":"_year"}),
        on=["_make_up","_model_up","_year"], how="left"
    )
    df_z0["esc_consensus"] = df_z0["esc_consensus"].fillna("not_found")

    n_standard  = int((df_z0["T"] == 1.0).sum())
    n_no        = int((df_z0["T"] == 0.0).sum())
    n_optional  = int((df_z0["T"].isna() & (df_z0["esc_consensus"] == "Optional")).sum())
    n_nf_amb    = int((df_z0["T"].isna() & (df_z0["esc_consensus"].isin(["not_found","ambiguous"]))).sum())

    w_standard = float(df_z0[df_z0["T"] == 1.0]["W"].fillna(0).sum())
    w_no       = float(df_z0[df_z0["T"] == 0.0]["W"].fillna(0).sum())
    w_optional = float(df_z0[(df_z0["T"].isna()) & (df_z0["esc_consensus"] == "Optional")]["W"].fillna(0).sum())
    w_nf_amb   = float(df_z0[(df_z0["T"].isna()) & (df_z0["esc_consensus"].isin(["not_found","ambiguous"]))]["W"].fillna(0).sum())
    w_total    = float(df_z0["W"].fillna(0).sum()) or 1.0

    print(f"\n  [4-Category Coverage (Z=0 group, passenger cars only)]", flush=True)
    print(f"    Standard   (T=1 certain):    {n_standard:,} ({100*n_standard/max(n_z0,1):.1f}% uw, {100*w_standard/w_total:.1f}% w)", flush=True)
    print(f"    No ESC     (T=0 certain):    {n_no:,} ({100*n_no/max(n_z0,1):.1f}% uw, {100*w_no/w_total:.1f}% w)", flush=True)
    print(f"    Optional   (self-selection): {n_optional:,} ({100*n_optional/max(n_z0,1):.1f}% uw, {100*w_optional/w_total:.1f}% w)", flush=True)
    print(f"    not_found/ambiguous (API gap): {n_nf_amb:,} ({100*n_nf_amb/max(n_z0,1):.1f}% uw, {100*w_nf_amb/w_total:.1f}% w)", flush=True)

    # ------------------------------------------------------------------
    # STEP 4: Analytic sample
    # ------------------------------------------------------------------
    print(f"\n[STEP 4] Analytic sample", flush=True)
    mask = (df["Z"].notna() & df["T"].notna() & df["Y_binary"].notna() &
            df["W"].notna() & (df["W"] > 0))
    df_a = df[mask].copy()
    n_analytic = len(df_a)
    naz0 = int((df_a["Z"] == 0).sum())
    naz1 = int((df_a["Z"] == 1).sum())
    naz_t0 = int((df_a["T"] == 0).sum())
    naz_t1 = int((df_a["T"] == 1).sum())
    print(f"  N_analytic = {n_analytic:,}  (Z=0: {naz0:,}, Z=1: {naz1:,})", flush=True)
    print(f"  T=0: {naz_t0:,}  T=1: {naz_t1:,}", flush=True)
    inj_z0 = df_a[df_a["Z"] == 0]["Y_binary"].mean()
    inj_z1 = df_a[df_a["Z"] == 1]["Y_binary"].mean()
    print(f"  Unweighted injury rate: Z=0={inj_z0:.3f} Z=1={inj_z1:.3f}", flush=True)

    Y_a = df_a["Y_binary"].values.astype(float)
    T_a = df_a["T"].values.astype(float)
    Z_a = df_a["Z"].values.astype(float)
    W_a = df_a["W"].values.astype(float)

    strat_col = "PSUSTRAT_ACC" if "PSUSTRAT_ACC" in df_a.columns else "STRATUM"
    casenum   = (df_a["CASENUM"].fillna(0).astype(int).values
                 if "CASENUM" in df_a.columns else np.arange(len(df_a)))
    year_arr  = df_a["CRSS_YEAR"].fillna(0).astype(int).values
    strat_arr = (df_a[strat_col].fillna(0).astype(int).values
                 if strat_col in df_a.columns else np.zeros(len(df_a), int))
    comp_strat = year_arr * 1000 + strat_arr
    comp_case  = year_arr * 1_000_000 + casenum

    # ------------------------------------------------------------------
    # STEP 5: First stage
    # ------------------------------------------------------------------
    print(f"\n[STEP 5] First stage (instrument strength)", flush=True)
    fs = _first_stage_fstat(T_a, Z_a, W_a)
    print(f"  E[T|Z=0] = {fs['ET_Z0']:.4f}  E[T|Z=1] = {fs['ET_Z1']:.4f}", flush=True)
    print(f"  FS = {fs['FS']:.4f}  SE_FS = {fs.get('SE_FS', float('nan')):.5f}"
          f"  F-stat = {fs['F_stat']:.1f}", flush=True)
    if fs["F_stat"] < 10:
        print("  [WARN] F-stat < 10: WEAK INSTRUMENT. IV Wald tau has inflated variance.",
              flush=True)

    # ------------------------------------------------------------------
    # STEP 6: IV Wald (primary)
    # ------------------------------------------------------------------
    print(f"\n[STEP 6] IV Wald (primary: Y_binary, W=CRSS WEIGHT, Z=Z_binary)", flush=True)
    wald = _hajek_wald(Y_a, T_a, Z_a, W_a)
    tau, se = wald["tau"], wald["SE_tau"]
    print(f"  tau_hat = {tau:.5f}  SE = {se:.5f}", flush=True)
    print(f"  95% CI  = [{wald['CI_L']:.5f}, {wald['CI_U']:.5f}]  (delta method)", flush=True)
    print(f"  z-stat  = {wald['z_stat']:.3f}  p-value = {wald['p_value']:.4f}", flush=True)
    print(f"  H0: tau=0 -> {'REJECT' if wald['reject'] else 'FAIL TO REJECT'} "
          f"at alpha=0.05", flush=True)
    print(f"  RF (reduced form) = {wald['RF']:.5f}", flush=True)

    # Robustness: Z_wide
    print(f"\n  [Robustness: Z_wide (pre-mandate MY<={Z_PRE_ESC_MAX_MY_WIDE})]", flush=True)
    mask_w = (df["Z_wide"].notna() & df["T"].notna() & df["Y_binary"].notna() &
              df["W"].notna() & (df["W"] > 0))
    df_w = df[mask_w]
    if len(df_w) >= 20:
        wald_w = _hajek_wald(df_w["Y_binary"].values.astype(float),
                             df_w["T"].values.astype(float),
                             df_w["Z_wide"].values.astype(float),
                             df_w["W"].values.astype(float))
        print(f"  tau_hat (Z_wide) = {wald_w['tau']:.5f}  p = {wald_w['p_value']:.4f}"
              f"  n_Z0={wald_w['n_Z0']:,}", flush=True)
    else:
        wald_w = {}
        print("  Insufficient observations for Z_wide", flush=True)

    # ------------------------------------------------------------------
    # STEP 7: Bootstrap CI
    # ------------------------------------------------------------------
    print(f"\n[STEP 7] Bootstrap CI (B={N_BOOTSTRAP}, numpy cluster bootstrap, "
          f"seed={BOOTSTRAP_SEED})", flush=True)
    bs = bootstrap_iv_wald_np(Y_a, T_a, Z_a, W_a, comp_case, comp_strat)
    print(f"  95% CI (bootstrap) = [{bs['CI_L_boot']:.5f}, {bs['CI_U_boot']:.5f}]",
          flush=True)
    print(f"  SE_boot = {bs['SE_boot']:.5f}  (n_valid={bs['n_boot_valid']})", flush=True)

    # ------------------------------------------------------------------
    # STEP 8: Power
    # ------------------------------------------------------------------
    print(f"\n[STEP 8] Power assessment", flush=True)
    pwr = assess_power(tau, se)
    print(f"  Power at |tau|={abs(tau):.4f}: {pwr['power_at_obs_tau']:.3f}", flush=True)
    print(f"  MDE for 80% power: {pwr['mde_80pct']:.4f}", flush=True)
    print(f"  Power at reference tau: {pwr['power_at_reference_tau']}", flush=True)
    if pwr["power_at_obs_tau"] < 0.5:
        print("  [WARN] Power < 0.50. Non-rejection is consistent with both "
              "tau=0 (risk homeostasis) and insufficient power to detect a "
              "meaningful effect. Do NOT interpret non-rejection as confirmation "
              "of homeostasis.", flush=True)

    # ------------------------------------------------------------------
    # STEP 9: Honest evaluation
    # ------------------------------------------------------------------
    print(f"\n[STEP 9] Honest evaluation", flush=True)
    if wald["reject"]:
        print(f"  RESULT: REJECT H0 (tau=0) at alpha={ALPHA_LEVEL}.", flush=True)
        if tau < 0:
            print(f"  tau < 0: ESC reduces injury probability (kappa_LATE < 1).", flush=True)
        else:
            print(f"  tau > 0: unexpected direction -- check carefully.", flush=True)
    else:
        print(f"  RESULT: FAIL TO REJECT H0. Cannot rule out tau=0 from observed data.",
              flush=True)
        print(f"  Estimated power at observed |tau|={abs(tau):.4f}: "
              f"{pwr['power_at_obs_tau']:.3f}", flush=True)
        print(f"  Plausible true effects of |tau|=0.03 would have power "
              f"{pwr['power_at_reference_tau'].get('|tau|=0.030','N/A')}.", flush=True)
        print(f"  Non-rejection does NOT confirm risk homeostasis.", flush=True)

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    stem = f"realdata_tier2_v2_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    aggregate = {
        "env_log"       : get_env_log(n_raw, n_analytic),
        "population_filter": {
            "applied"              : True,
            "FMVSS126_INCLUDE"     : sorted(FMVSS126_INCLUDE),
            "GVWR_CONDITIONAL"     : sorted(GVWR_CONDITIONAL),
            "n_vehicles_pre_filter": n_veh_total_all,
            "n_vehicles_excluded"  : n_veh_out_all,
            "pct_excluded"         : round(100*n_veh_out_all/max(n_veh_total_all,1), 2),
        },
        "esc_cache_v2_summary": esc_v2_summary,
        "coverage_z0"   : {
            "n_z0"            : n_z0,
            "n_Standard_T1"   : n_standard,
            "n_No_T0"         : n_no,
            "n_Optional_Tna"  : n_optional,
            "n_NotFound_Tna"  : n_nf_amb,
            "pct_Standard_uw" : round(100*n_standard/max(n_z0,1), 2),
            "pct_No_uw"       : round(100*n_no/max(n_z0,1), 2),
            "pct_Optional_uw" : round(100*n_optional/max(n_z0,1), 2),
            "pct_NotFound_uw" : round(100*n_nf_amb/max(n_z0,1), 2),
            "pct_Standard_w"  : round(100*w_standard/w_total, 2),
            "pct_No_w"        : round(100*w_no/w_total, 2),
            "pct_Optional_w"  : round(100*w_optional/w_total, 2),
            "pct_NotFound_w"  : round(100*w_nf_amb/w_total, 2),
            "n_z1"            : n_z1,
            "n_analytic"      : n_analytic,
            "n_az0"           : naz0,
            "n_az1"           : naz1,
        },
        "first_stage"   : fs,
        "wald_primary"  : wald,
        "wald_Z_wide"   : wald_w,
        "bootstrap"     : bs,
        "power"         : pwr,
        "analysis_note" : (
            "v2 adds BODY_TYP population filter (FMVSS 126 scope) and fixes "
            "2-char model matching bug.  ESC cache rebuilt as v2 CSV."
        ),
    }

    json_path = RESULTS_DIR / f"{stem}.json"
    with open(json_path, "w") as f:
        json.dump(aggregate, f, indent=2, allow_nan=True)
    print(f"\nSaved: {json_path}", flush=True)

    print("\n" + "=" * 72, flush=True)
    print("FINAL SUMMARY -- v2 (FMVSS 126 Population Fix)", flush=True)
    print("=" * 72, flush=True)
    print(f"Dataset     : CRSS pooled {CRSS_YEARS}", flush=True)
    print(f"Population  : FMVSS 126 scope (passenger cars, light trucks, SUVs, minivans)", flush=True)
    print(f"N_analytic  : {n_analytic:,}  (Z=0: {naz0:,}, Z=1: {naz1:,})", flush=True)
    print(f"", flush=True)
    print(f"Coverage (Z=0 group, passenger cars, unweighted / weighted):", flush=True)
    print(f"  T=1 Standard  : {n_standard:,} ({100*n_standard/max(n_z0,1):.1f}% uw / {100*w_standard/w_total:.1f}% w)", flush=True)
    print(f"  T=0 No ESC    : {n_no:,} ({100*n_no/max(n_z0,1):.1f}% uw / {100*w_no/w_total:.1f}% w)", flush=True)
    print(f"  T=? Optional  : {n_optional:,} ({100*n_optional/max(n_z0,1):.1f}% uw / {100*w_optional/w_total:.1f}% w)", flush=True)
    print(f"  T=? not_found : {n_nf_amb:,} ({100*n_nf_amb/max(n_z0,1):.1f}% uw / {100*w_nf_amb/w_total:.1f}% w)", flush=True)
    print(f"", flush=True)
    print(f"First Stage : FS={fs['FS']:.4f}  F-stat={fs['F_stat']:.1f}", flush=True)
    print(f"", flush=True)
    print(f"PRIMARY RESULT (Y_binary, Z_binary, W=CRSS_WEIGHT, H0: tau=0):", flush=True)
    print(f"  tau_hat = {tau:.5f}", flush=True)
    print(f"  SE      = {se:.5f}", flush=True)
    print(f"  95% CI  = [{wald['CI_L']:.5f}, {wald['CI_U']:.5f}]  (delta method)", flush=True)
    print(f"  95% CI  = [{bs['CI_L_boot']:.5f}, {bs['CI_U_boot']:.5f}]  (bootstrap)", flush=True)
    print(f"  z-stat  = {wald['z_stat']:.3f}  p-value = {wald['p_value']:.4f}", flush=True)
    print(f"  Decision: {'REJECT H0' if wald['reject'] else 'FAIL TO REJECT H0'} at alpha=0.05", flush=True)
    print(f"", flush=True)
    print(f"Power at observed |tau|: {pwr['power_at_obs_tau']:.3f}", flush=True)
    print(f"MDE for 80% power: {pwr['mde_80pct']:.4f}", flush=True)
    print(f"\nEnd: {datetime.utcnow().isoformat()}Z", flush=True)
    print("=" * 72, flush=True)

    return aggregate


if __name__ == "__main__":
    result = main()
