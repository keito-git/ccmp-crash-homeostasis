"""
realdata_changeover_design.py -- Within Make-Model Changeover Design
=======================================================================
Script: realdata_changeover_design.py  (NEW FILE -- existing scripts NOT modified)
Authors: Keito Inoshita, Akira Kawai
Date  : 2026-07-14

PURPOSE
-------
The naive IV design (Z = {MY>=2012} vs {MY<=2010}) was shown by falsification_test.py
to violate the exclusion restriction: placebo tau_hat = -0.0585 (p<0.0001), indicating
that vehicle-generation confounding (crashworthiness improvements, airbag upgrades, etc.)
drives the main result, not ESC.

This script implements the within-make-model changeover design following NHTSA Dang
(2007, DOT HS 810 794):
  For make/models that switched from No ESC -> Standard ESC,
  compare MY in [y*, y*+k-1] (T=1, ESC standard) vs MY in [y*-k, y*-1] (T=0, No ESC)
  where y* = first MY with Standard ESC in the NHTSA API equipment list.

This design:
  - Controls for make/model fixed effects (same vehicle line)
  - Limits vehicle generation drift to k years within the same model line
  - Uses DID control group (non-switcher vehicles) to remove k-year secular trend

PRE-SPECIFIED DEFINITIONS (FIXED BEFORE RUNNING -- 2026-07-14)
---------------------------------------------------------------
1. Switcher: (make,model) with >=1 'No' AND >=1 'Standard' year in ESC list
2. Optional handling: if any year in pre-window or post-window has status != expected
   (not 'No' for pre, not 'Standard' for post), EXCLUDE that (make,model) for this k.
   Optional years are thus disqualifying if they fall within a required window.
3. Pre-window: years in [y*-k, y*-1], all must have esc_consensus=='No'
4. Post-window: years in [y*, y*+k-1], all must have esc_consensus=='Standard'
5. Both windows within [2000, 2010] (list coverage)
6. T: pre-window -> T=0, post-window -> T=1 (FS = 1.0 exactly by construction)
7. Z_within: pre-window -> 0, post-window -> 1
8. Estimator: identical _hajek_wald from v2 (NOT changed).
   Note: since T = Z_within exactly (FS=1), tau_hat = RF = E[Y|Z=1] - E[Y|Z=0].
9. DID reference year: R = round(median y* across active switchers for each k)
10. DID control group: non-switcher (all-No or all-Standard) make/models in same MY range
    Trend = E[Y|control, MY in post-window of R] - E[Y|control, MY in pre-window of R]
    DID_tau = tau_hat_switcher - Trend
11. Placebo: P_CRASH1==5 AND MAN_COLL==1 (identical to falsification_test.py)
12. Rollover: ROLLOVER in {1, 2, 9} (identical to falsification_test.py)
13. k sensitivity: k = 1, 2, 3
14. H0 test: tau=0 at alpha=0.05

DECISION CRITERION (pre-specified):
  Placebo p >= 0.05 AND tau not significantly negative -> PASS (design valid)
  Placebo p <  0.05 AND tau significantly negative    -> FAIL (design still confounded)

INTEGRITY REQUIREMENTS
----------------------
- No result fabrication.
- Group definitions fixed before data is loaded (this header documents the pre-spec).
- If placebo still shows effect: report clearly and honestly.
- Existing data/scripts are NOT modified.
- GPU: none (CPU-only).
"""

from __future__ import annotations

import json
import platform
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

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

ESC_API_CSV_V2 = DATA_DIR / "esc_equipment_list_nhtsa_api_v2.csv"

SCRATCHPAD = Path("/private/tmp/claude-501/-Users-redacted-user/"
                  "373ef59a-33a6-4195-ad55-30c70723863d/scratchpad")

CRSS_YEARS: List[int] = [2016, 2017, 2018, 2019, 2020, 2021]

# ---------------------------------------------------------------------------
# Pre-specified constants
# ---------------------------------------------------------------------------
MY_MIN_COVERAGE = 2000
MY_MAX_COVERAGE = 2010       # ESC list upper bound

INJ_SEV_VALID   : Set[int] = {0, 1, 2, 3, 4}
INJ_SEV_INJURY  : Set[int] = {1, 2, 3, 4}
ALPHA_LEVEL     : float    = 0.05
K_VALUES        : List[int] = [1, 2, 3]

# Placebo / ESC-active (identical to falsification_test.py)
PLACEBO_P_CRASH1_VALUES   : Set[int] = frozenset({5})      # "Stopped in Roadway"
PLACEBO_MAN_COLL_VALUES   : Set[int] = frozenset({1})      # "Front-to-Rear"
ESC_ACTIVE_ROLLOVER_VALUES: Set[int] = frozenset({1, 2, 9})

# FMVSS 126 population filter (identical to v2)
FMVSS126_INCLUDE: Set[int] = frozenset({
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
    14, 15, 16, 17, 19, 20, 22, 34, 39, 42, 45, 48, 49,
})
GVWR_CONDITIONAL: Set[int] = frozenset({21, 40})

NAN = float("nan")


# ---------------------------------------------------------------------------
# FMVSS 126 scope helpers (identical to v2)
# ---------------------------------------------------------------------------

def _normalize_gvwr(df_veh: pd.DataFrame) -> pd.Series:
    if "GVWR_FROM" in df_veh.columns:
        mapping = {11: 1, 12: 1, 13: 2, 14: 2, 15: 2, 16: 2, 17: 3, 18: 3, 98: 8, 99: 8}
        return df_veh["GVWR_FROM"].map(mapping).fillna(8).astype(int)
    elif "GVWR" in df_veh.columns:
        return df_veh["GVWR"].fillna(8).replace({9: 8}).astype(int)
    return pd.Series(8, index=df_veh.index, dtype=int)


def _is_fmvss126_scope(body_typ: pd.Series, gvwr_norm: pd.Series) -> pd.Series:
    return body_typ.isin(FMVSS126_INCLUDE) | (body_typ.isin(GVWR_CONDITIONAL) & (gvwr_norm == 1))


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


# ---------------------------------------------------------------------------
# Switcher / Control Identification
# ---------------------------------------------------------------------------

def build_changeover_lookups(
    esc_df: pd.DataFrame,
    k_values: List[int],
) -> Dict[int, Dict]:
    """
    Build per-k changeover lookups.  Pre-specified rules in module docstring.

    Returns dict k -> {
        'lookup'       : {(make_up, model_up, MY) -> Z_within (0 or 1)},
        'switchers'    : {(make_up, model_up) -> {'y_star', 'pre_years', 'post_years'}},
        'n_switchers'  : int,
        'y_star_values': List[int],
        'median_y_star': float,
    }
    """
    esc = esc_df.copy()
    esc["make_up"]  = esc["make"].fillna("").str.upper().str.strip()
    esc["model_up"] = esc["model"].fillna("").str.upper().str.strip()

    out: Dict[int, Dict] = {}

    for k in k_values:
        lookup   : Dict[Tuple, int]  = {}
        switchers: Dict[Tuple, Dict] = {}
        y_stars  : List[int]         = []

        for (mu, mo), grp in esc.groupby(["make_up", "model_up"]):
            ys_map: Dict[int, str] = {
                int(r["year"]): r["esc_consensus"] for _, r in grp.iterrows()
            }
            std_yrs = [y for y, s in ys_map.items() if s == "Standard"]
            no_yrs  = [y for y, s in ys_map.items() if s == "No"]

            if not std_yrs or not no_yrs:
                continue

            y_star = min(std_yrs)

            pre_yrs  = list(range(y_star - k, y_star))
            post_yrs = list(range(y_star, y_star + k))

            # Both windows within list coverage
            if not (all(MY_MIN_COVERAGE <= y <= MY_MAX_COVERAGE for y in pre_yrs + post_yrs)):
                continue
            # All pre years must be 'No'
            if not all(ys_map.get(y, "missing") == "No" for y in pre_yrs):
                continue
            # All post years must be 'Standard'
            if not all(ys_map.get(y, "missing") == "Standard" for y in post_yrs):
                continue

            key = (mu, mo)
            switchers[key] = {"y_star": y_star, "pre_years": pre_yrs, "post_years": post_yrs}
            y_stars.append(y_star)
            for y in pre_yrs:
                lookup[(mu, mo, y)] = 0
            for y in post_yrs:
                lookup[(mu, mo, y)] = 1

        med = float(np.median(y_stars)) if y_stars else NAN
        print(
            f"  k={k}: {len(switchers)} switchers, "
            f"{len(lookup)} (make,model,MY) entries, "
            f"median y*={med:.1f}",
            flush=True,
        )
        out[k] = {
            "lookup":        lookup,
            "switchers":     switchers,
            "n_switchers":   len(switchers),
            "y_star_values": y_stars,
            "median_y_star": med,
        }

    return out


def build_control_sets(esc_df: pd.DataFrame) -> Tuple[Set[Tuple], Set[Tuple]]:
    """
    Return (all_no_set, all_std_set) — make/models that never switched.
    all_no:  esc_consensus in {No, not_found, ambiguous} only -> all-No throughout
    all_std: esc_consensus == Standard only -> all-Standard throughout
    """
    esc = esc_df.copy()
    esc["make_up"]  = esc["make"].fillna("").str.upper().str.strip()
    esc["model_up"] = esc["model"].fillna("").str.upper().str.strip()

    all_no : Set[Tuple] = set()
    all_std: Set[Tuple] = set()

    for (mu, mo), grp in esc.groupby(["make_up", "model_up"]):
        statuses = set(grp["esc_consensus"].tolist())
        clean    = statuses - {"not_found", "ambiguous"}
        if "No" in statuses and "Standard" not in clean and "Optional" not in clean:
            all_no.add((mu, mo))
        elif clean == {"Standard"}:
            all_std.add((mu, mo))

    print(f"  Control non-switchers: all-No={len(all_no)}, all-Standard={len(all_std)}", flush=True)
    return all_no, all_std


# ---------------------------------------------------------------------------
# CRSS Loading (identical structure to falsification_test.py)
# ---------------------------------------------------------------------------

def load_crss_year(year: int) -> Optional[pd.DataFrame]:
    crss_dir = SCRATCHPAD / f"crss{year}"
    if not crss_dir.exists():
        print(f"  CRSS {year}: directory not found, skipping", flush=True)
        return None

    acc_path = _find_csv(crss_dir, "accident.csv")
    veh_path = _find_csv(crss_dir, "vehicle.csv")
    per_path = _find_csv(crss_dir, "person.csv")
    vpc_path = _find_csv(crss_dir, "vpicdecode.csv")

    missing = [n for n, p in [
        ("accident", acc_path), ("vehicle", veh_path),
        ("person",   per_path), ("vpicdecode", vpc_path),
    ] if p is None]
    if missing:
        print(f"  CRSS {year}: missing {missing}, skipping", flush=True)
        return None

    print(f"  Loading CRSS {year}...", flush=True)
    try:
        df_acc = pd.read_csv(str(acc_path), encoding="latin-1", low_memory=False)
        df_veh = pd.read_csv(str(veh_path), encoding="latin-1", low_memory=False)
        df_per = pd.read_csv(str(per_path), encoding="latin-1", low_memory=False)
        df_vpc = pd.read_csv(str(vpc_path), encoding="latin-1", low_memory=False)
    except Exception as e:
        print(f"  CRSS {year}: load error {e}, skipping", flush=True)
        return None

    for d in [df_acc, df_veh, df_per, df_vpc]:
        d.columns = [c.upper() for c in d.columns]

    # FMVSS 126 filter
    df_veh["GVWR_NORM"] = _normalize_gvwr(df_veh)
    if "BODY_TYP" in df_veh.columns:
        df_veh["BODY_TYP"] = pd.to_numeric(df_veh["BODY_TYP"], errors="coerce")
        scope = _is_fmvss126_scope(df_veh["BODY_TYP"], df_veh["GVWR_NORM"])
    else:
        scope = pd.Series(True, index=df_veh.index)
    df_veh_f = df_veh[scope].copy()
    print(f"  CRSS {year}: {len(df_veh):,} vehicles -> {scope.sum():,} in FMVSS126 scope",
          flush=True)

    # Column selection (vehicle: include falsification cols)
    veh_want = ["CASENUM", "VEH_NO", "MOD_YEAR", "BODY_TYP", "GVWR_NORM",
                "P_CRASH1", "MAN_COLL", "ROLLOVER"]
    veh_cols = [c for c in veh_want if c in df_veh_f.columns]
    df_v = df_veh_f[veh_cols]

    acc_want = ["CASENUM", "WEIGHT"]
    acc_cols = [c for c in acc_want if c in df_acc.columns]
    df_a = df_acc[acc_cols].rename(columns={"WEIGHT": "WEIGHT_ACC"})

    per_want = ["CASENUM", "VEH_NO", "INJ_SEV", "WEIGHT"]
    per_cols = [c for c in per_want if c in df_per.columns]
    df_p = df_per[per_cols].rename(columns={"WEIGHT": "WEIGHT_PER"})

    vpc_want = ["CASENUM", "VEH_NO"]
    if "MAKE"  in df_vpc.columns: vpc_want.append("MAKE")
    if "MODEL" in df_vpc.columns: vpc_want.append("MODEL")
    df_vpc = df_vpc[vpc_want].rename(columns={"MAKE": "MAKE_API", "MODEL": "MODEL_API"})

    df = pd.merge(df_p, df_v,   on=["CASENUM", "VEH_NO"], how="inner")
    df = pd.merge(df,   df_a,   on="CASENUM",              how="left")
    df = pd.merge(df,   df_vpc, on=["CASENUM", "VEH_NO"],  how="left")
    df["CRSS_YEAR"] = year

    for col in ["MOD_YEAR", "INJ_SEV", "WEIGHT_PER", "WEIGHT_ACC",
                "BODY_TYP", "P_CRASH1", "MAN_COLL", "ROLLOVER"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    print(f"  CRSS {year}: {len(df):,} person-vehicle records (post-filter)", flush=True)
    return df


# ---------------------------------------------------------------------------
# Variable Assignment
# ---------------------------------------------------------------------------

def assign_variables(
    df_raw: pd.DataFrame,
    lookup: Dict[Tuple, int],
    all_no_set:  Set[Tuple],
    all_std_set: Set[Tuple],
) -> pd.DataFrame:
    """
    For a given k, assign Z_within, T, Y_binary, W, IS_CONTROL, MY_INT.

    Z_within: 0 (pre-window/T=0) or 1 (post-window/T=1), NaN if not in any switcher window
    T:        same as Z_within (T = Z_within exactly; FS = 1.0 by construction)
    IS_CONTROL: True for non-switcher (all-No or all-Standard) make/models
    MY_INT:   integer model year (for DID window lookup)
    """
    df = df_raw.copy()

    make_up  = df["MAKE_API"].fillna("").astype(str).str.upper().str.strip()
    model_up = df["MODEL_API"].fillna("").astype(str).str.upper().str.strip()
    my_int   = df["MOD_YEAR"].fillna(-1).astype(int)

    # Z_within from lookup
    z_vals = [lookup.get((mu, mo, m), NAN) for mu, mo, m in zip(make_up, model_up, my_int)]
    df["Z_within"] = z_vals
    df["T"]        = df["Z_within"].copy()   # T = Z_within exactly
    df["MY_INT"]   = my_int.values

    # Y_binary
    df["Y_binary"] = NAN
    valid = df["INJ_SEV"].isin(INJ_SEV_VALID)
    df.loc[valid, "Y_binary"] = df.loc[valid, "INJ_SEV"].isin(INJ_SEV_INJURY).astype(float)

    # Survey weight
    df["W"] = df["WEIGHT_PER"].fillna(0.0)
    df.loc[df["W"] <= 0, "W"] = NAN

    # Control group flags
    df["IS_CONTROL"] = [(mu, mo) in all_no_set or (mu, mo) in all_std_set
                        for mu, mo in zip(make_up, model_up)]

    return df


# ---------------------------------------------------------------------------
# Crash-type masks (identical definitions to falsification_test.py)
# ---------------------------------------------------------------------------

def make_crash_masks(df: pd.DataFrame) -> Dict[str, pd.Series]:
    """Return boolean masks: 'placebo' and 'rollover'."""
    p_ok = df["P_CRASH1"].isin(PLACEBO_P_CRASH1_VALUES) if "P_CRASH1" in df.columns \
           else pd.Series(False, index=df.index)
    m_ok = df["MAN_COLL"].isin(PLACEBO_MAN_COLL_VALUES) if "MAN_COLL" in df.columns \
           else pd.Series(False, index=df.index)
    r_ok = df["ROLLOVER"].isin(ESC_ACTIVE_ROLLOVER_VALUES) if "ROLLOVER" in df.columns \
           else pd.Series(False, index=df.index)
    return {
        "placebo":  (p_ok & m_ok),
        "rollover": r_ok,
    }


# ---------------------------------------------------------------------------
# Hajek-Weighted IV Wald (IDENTICAL to v2 -- NOT changed)
# ---------------------------------------------------------------------------

def _hajek_wald(
    Y: np.ndarray, T: np.ndarray, Z: np.ndarray, W: np.ndarray,
) -> Dict:
    """Hajek-weighted IV Wald estimator (line-for-line from v2 _hajek_wald)."""
    m1 = (Z == 1); m0 = (Z == 0)
    n1, n0 = int(m1.sum()), int(m0.sum())
    if n1 < 10 or n0 < 10:
        return {"tau": NAN, "SE_tau": NAN, "CI_L": NAN, "CI_U": NAN,
                "z_stat": NAN, "p_value": NAN, "reject": False,
                "FS": NAN, "RF": NAN,
                "EY_Z1": NAN, "EY_Z0": NAN, "ET_Z1": NAN, "ET_Z0": NAN,
                "n_Z0": n0, "n_Z1": n1}
    w1, w0   = W[m1], W[m0]
    sw1, sw0 = w1.sum(), w0.sum()
    EY_Z1 = float(np.dot(Y[m1], w1) / sw1)
    EY_Z0 = float(np.dot(Y[m0], w0) / sw0)
    ET_Z1 = float(np.dot(T[m1], w1) / sw1)
    ET_Z0 = float(np.dot(T[m0], w0) / sw0)
    RF = EY_Z1 - EY_Z0
    FS = ET_Z1 - ET_Z0
    if abs(FS) < 1e-8:
        return {"tau": NAN, "SE_tau": NAN, "CI_L": NAN, "CI_U": NAN,
                "z_stat": NAN, "p_value": NAN, "reject": False,
                "FS": float(FS), "RF": float(RF),
                "EY_Z1": float(EY_Z1), "EY_Z0": float(EY_Z0),
                "ET_Z1": float(ET_Z1), "ET_Z0": float(ET_Z0),
                "n_Z0": n0, "n_Z1": n1}
    tau = RF / FS
    rY1 = Y[m1] - EY_Z1; rY0 = Y[m0] - EY_Z0
    rT1 = T[m1] - ET_Z1; rT0 = T[m0] - ET_Z0
    var_RF = (np.dot(w1**2, rY1**2) / sw1**2 + np.dot(w0**2, rY0**2) / sw0**2)
    var_FS = (np.dot(w1**2, rT1**2) / sw1**2 + np.dot(w0**2, rT0**2) / sw0**2)
    SE_tau = float(np.sqrt(max(1e-20, var_RF / FS**2 + RF**2 * var_FS / FS**4)))
    z_stat = tau / SE_tau
    p_val  = float(2.0 * scipy.stats.norm.sf(abs(z_stat)))
    return {
        "tau": float(tau), "SE_tau": SE_tau,
        "CI_L": float(tau - 1.96 * SE_tau), "CI_U": float(tau + 1.96 * SE_tau),
        "z_stat": float(z_stat), "p_value": p_val, "reject": bool(p_val < ALPHA_LEVEL),
        "FS": float(FS), "RF": float(RF),
        "EY_Z1": float(EY_Z1), "EY_Z0": float(EY_Z0),
        "ET_Z1": float(ET_Z1), "ET_Z0": float(ET_Z0),
        "n_Z0": n0, "n_Z1": n1,
    }


def run_wald(df_sub: pd.DataFrame, label: str = "") -> Dict:
    """Run Hajek Wald on a pre-filtered DataFrame with Z_within, T, Y_binary, W."""
    n = len(df_sub)
    if n < 50:
        print(f"  [{label}] n={n} < 50 -> insufficient, skipping", flush=True)
        return {"tau": NAN, "SE_tau": NAN, "CI_L": NAN, "CI_U": NAN,
                "z_stat": NAN, "p_value": NAN, "reject": False,
                "FS": NAN, "RF": NAN, "n_Z0": 0, "n_Z1": 0,
                "n_subset": n, "label": label, "insufficient_n": True}
    Y = df_sub["Y_binary"].values.astype(float)
    T = df_sub["T"].values.astype(float)
    Z = df_sub["Z_within"].values.astype(float)
    W = df_sub["W"].values.astype(float)
    res = _hajek_wald(Y, T, Z, W)
    res["n_subset"] = n
    res["label"]    = label
    res["insufficient_n"] = False
    def _fv(v: float, fmt: str = "+.5f") -> str:
        return format(v, fmt) if not np.isnan(v) else "N/A"

    print(
        f"  [{label}] n={n:,}  Z=0:{res['n_Z0']:,}  Z=1:{res['n_Z1']:,}  "
        f"tau={_fv(res['tau'])}  SE={_fv(res['SE_tau'], '.5f')}  "
        f"p={_fv(res['p_value'], '.4f')}  RF={_fv(res['RF'])}  FS={_fv(res['FS'], '.5f')}",
        flush=True,
    )
    return res


# ---------------------------------------------------------------------------
# DID Secular Trend Estimation
# ---------------------------------------------------------------------------

def estimate_secular_trend(
    df: pd.DataFrame,
    k: int,
    median_y_star: float,
    crash_mask: Optional[pd.Series] = None,
    label: str = "",
) -> Dict:
    """
    Estimate the secular k-year MY drift in Y from non-switcher control vehicles.

    Reference year R = round(median_y_star).
    Control-pre:  MY in [R-k, R-1]  (same width as switcher pre-window)
    Control-post: MY in [R, R+k-1]  (same width as switcher post-window)

    Both clipped to [MY_MIN_COVERAGE, MY_MAX_COVERAGE].
    """
    R = int(round(median_y_star))
    pre_yrs  = [y for y in range(R - k, R)   if MY_MIN_COVERAGE <= y <= MY_MAX_COVERAGE]
    post_yrs = [y for y in range(R, R + k)   if MY_MIN_COVERAGE <= y <= MY_MAX_COVERAGE]

    if not pre_yrs or not post_yrs:
        return {"trend": NAN, "se_trend": NAN, "n_ctrl_pre": 0, "n_ctrl_post": 0,
                "ctrl_pre_years": pre_yrs, "ctrl_post_years": post_yrs, "R": R, "label": label}

    # Apply crash mask if given
    d = df if crash_mask is None else df[crash_mask.values]

    # Filter to control vehicles only
    d_ctrl = d[d["IS_CONTROL"]]

    # Analytic filter: valid Y and W
    valid  = d_ctrl["Y_binary"].notna() & d_ctrl["W"].notna() & (d_ctrl["W"] > 0)
    d_v    = d_ctrl[valid]

    d_pre  = d_v[d_v["MY_INT"].isin(pre_yrs)]
    d_post = d_v[d_v["MY_INT"].isin(post_yrs)]
    n_pre, n_post = len(d_pre), len(d_post)

    if n_pre < 10 or n_post < 10:
        print(
            f"  [DID/{label}] R={R} pre_yrs={pre_yrs} post_yrs={post_yrs} "
            f"n_ctrl_pre={n_pre} n_ctrl_post={n_post} -> insufficient N",
            flush=True,
        )
        return {"trend": NAN, "se_trend": NAN, "n_ctrl_pre": n_pre, "n_ctrl_post": n_post,
                "ctrl_pre_years": pre_yrs, "ctrl_post_years": post_yrs, "R": R, "label": label}

    w_pre, w_post = d_pre["W"].values, d_post["W"].values
    y_pre, y_post = d_pre["Y_binary"].values, d_post["Y_binary"].values

    m_pre  = float(np.dot(y_pre, w_pre)  / w_pre.sum())
    m_post = float(np.dot(y_post, w_post) / w_post.sum())
    trend  = m_post - m_pre

    var_pre  = float(np.dot(w_pre**2,  (y_pre  - m_pre)**2)  / w_pre.sum()**2)
    var_post = float(np.dot(w_post**2, (y_post - m_post)**2) / w_post.sum()**2)
    se_trend = float(np.sqrt(var_pre + var_post))

    print(
        f"  [DID/{label}] R={R} pre={pre_yrs} post={post_yrs} "
        f"n_pre={n_pre:,} n_post={n_post:,} "
        f"trend={trend:+.5f} (SE={se_trend:.5f})",
        flush=True,
    )
    return {
        "R": R,
        "ctrl_pre_years": pre_yrs, "ctrl_post_years": post_yrs,
        "mean_pre": m_pre, "mean_post": m_post,
        "trend": trend, "se_trend": se_trend,
        "n_ctrl_pre": n_pre, "n_ctrl_post": n_post,
        "label": label,
    }


def compute_did(tau: float, se_tau: float, trend: float, se_trend: float) -> Dict:
    """DID = tau - trend, SE via independence assumption."""
    if any(np.isnan(v) for v in [tau, se_tau, trend, se_trend]):
        return {"DID_tau": NAN, "DID_SE": NAN, "DID_CI_L": NAN, "DID_CI_U": NAN,
                "DID_p": NAN, "DID_reject": False}
    did     = tau - trend
    se_did  = float(np.sqrt(se_tau**2 + se_trend**2))
    z_stat  = did / se_did if se_did > 1e-10 else NAN
    p_val   = float(2.0 * scipy.stats.norm.sf(abs(z_stat))) if not np.isnan(z_stat) else NAN
    return {
        "DID_tau": float(did), "DID_SE": se_did,
        "DID_CI_L": float(did - 1.96 * se_did),
        "DID_CI_U": float(did + 1.96 * se_did),
        "DID_p": p_val, "DID_reject": bool(not np.isnan(p_val) and p_val < ALPHA_LEVEL),
    }


# ---------------------------------------------------------------------------
# Power Analysis
# ---------------------------------------------------------------------------

def compute_power(n0: int, n1: int, mean_y: float) -> Dict:
    """Minimum detectable effect and approximate power at 80% for target delta=-0.024."""
    if n0 < 2 or n1 < 2:
        return {"MDE_at_80pct": NAN, "approx_power_for_RF_0024": NAN}
    p  = max(0.01, min(0.99, float(mean_y) if not np.isnan(mean_y) else 0.38))
    se = float(np.sqrt(p * (1 - p) / n0 + p * (1 - p) / n1))
    z_a = scipy.stats.norm.ppf(1 - ALPHA_LEVEL / 2)   # 1.96
    z_b = scipy.stats.norm.ppf(0.80)                   # 0.84
    MDE = (z_a + z_b) * se
    # Power to detect RF=-0.024 (v2 main result reduced-form magnitude)
    ref_delta = 0.024
    pwr = float(scipy.stats.norm.cdf(ref_delta / se - z_a))
    return {
        "MDE_at_80pct": float(MDE),
        "approx_power_for_RF_0024": max(0.0, pwr),
        "p_assumed": p, "se_null": se, "n0": n0, "n1": n1,
    }


# ---------------------------------------------------------------------------
# Environment Log
# ---------------------------------------------------------------------------

def get_env_log(n_raw: int) -> Dict:
    try:
        git = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(BASE_DIR), stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        git = "unavailable"
    return {
        "timestamp":      datetime.utcnow().isoformat() + "Z",
        "script":         "realdata_changeover_design.py",
        "python_version": sys.version,
        "numpy_version":  np.__version__,
        "pandas_version": pd.__version__,
        "platform":       platform.platform(),
        "hostname":       socket.gethostname(),
        "git_commit":     git,
        "GPU":            "none (CPU-only)",
        "crss_years":     CRSS_YEARS,
        "n_raw_loaded":   n_raw,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 72, flush=True)
    print("Within Make-Model Changeover Design", flush=True)
    print(f"Start: {datetime.utcnow().isoformat()}Z", flush=True)
    print("=" * 72, flush=True)

    print("\n[PRE-ANALYSIS DECLARATION -- definitions fixed before data load]", flush=True)
    print(
        "  Switcher: (make,model) with >=1 'No' AND >=1 'Standard' in ESC list\n"
        "  Pre-window: [y*-k, y*-1] -- all years must be 'No'\n"
        "  Post-window: [y*, y*+k-1] -- all years must be 'Standard'\n"
        "  Optional in required window -> EXCLUDE (make,model) for this k\n"
        "  T = Z_within exactly -> FS = 1.0 by construction\n"
        "  Estimator: identical _hajek_wald from v2 (NOT changed)\n"
        "  DID reference: R = round(median y* for each k)\n"
        "  Placebo: P_CRASH1==5 AND MAN_COLL==1 (identical to falsification_test.py)\n"
        "  Rollover: ROLLOVER in {1,2,9}\n"
        "  Decision: placebo p >= 0.05 -> PASS; placebo p < 0.05 -> FAIL",
        flush=True,
    )

    # ------------------------------------------------------------------
    # STEP 1: ESC lookups
    # ------------------------------------------------------------------
    print("\n[STEP 1] Build changeover lookups", flush=True)
    if not ESC_API_CSV_V2.exists():
        raise FileNotFoundError(f"ESC list not found: {ESC_API_CSV_V2}")
    esc_df = pd.read_csv(ESC_API_CSV_V2)
    print(f"  Loaded {len(esc_df):,} ESC list entries", flush=True)

    co_info    = build_changeover_lookups(esc_df, K_VALUES)
    all_no, all_std = build_control_sets(esc_df)

    # ------------------------------------------------------------------
    # STEP 2: Load CRSS
    # ------------------------------------------------------------------
    print(f"\n[STEP 2] Load CRSS ({CRSS_YEARS})", flush=True)
    frames = []
    for yr in CRSS_YEARS:
        df_y = load_crss_year(yr)
        if df_y is not None:
            frames.append(df_y)
    if not frames:
        raise RuntimeError("No CRSS data loaded.")
    df_raw = pd.concat(frames, ignore_index=True)
    n_raw  = len(df_raw)
    print(f"\n  Pooled: {n_raw:,} person-vehicle records", flush=True)

    for col in ["P_CRASH1", "MAN_COLL", "ROLLOVER"]:
        if col in df_raw.columns:
            n_ok = int(df_raw[col].notna().sum())
            print(f"  {col} valid: {n_ok:,}", flush=True)
        else:
            print(f"  {col}: NOT IN DATA", flush=True)

    # ------------------------------------------------------------------
    # STEP 3: Per-k analysis
    # ------------------------------------------------------------------
    print("\n[STEP 3] Analysis by k", flush=True)
    all_results: Dict[int, Dict] = {}

    for k in K_VALUES:
        print(f"\n{'='*60}", flush=True)
        print(f"k = {k}", flush=True)
        print(f"{'='*60}", flush=True)

        info      = co_info[k]
        lookup    = info["lookup"]
        n_sw      = info["n_switchers"]
        med_ys    = info["median_y_star"]

        if n_sw == 0:
            print("  No valid switchers. Skipping.", flush=True)
            all_results[k] = {"error": "no_valid_switchers"}
            continue

        print(f"  Switcher make/models: {n_sw},  median y*={med_ys}", flush=True)

        # Assign variables for this k
        df = assign_variables(df_raw, lookup, all_no, all_std)

        # Crash masks on the FULL df (needed for DID control vehicles too)
        full_masks = make_crash_masks(df)

        # ---------------------
        # SWITCHER ANALYTIC SAMPLE
        # ---------------------
        # Rows where Z_within is defined AND Y valid AND W valid
        sw_mask = (
            df["Z_within"].notna() &
            df["Y_binary"].notna() &
            df["W"].notna() & (df["W"] > 0)
        )
        df_sw = df[sw_mask].copy()
        n_a   = len(df_sw)
        n_z0  = int((df_sw["Z_within"] == 0).sum())
        n_z1  = int((df_sw["Z_within"] == 1).sum())

        print(f"\n  SWITCHER ANALYTIC SAMPLE: N={n_a:,}  Z=0(pre):{n_z0:,}  Z=1(post):{n_z1:,}",
              flush=True)

        # FS sanity check (should be 1.0)
        if n_a > 0:
            fs_check = (df_sw["T"] == df_sw["Z_within"]).all()
            print(f"  T == Z_within always (FS=1 sanity): {fs_check}", flush=True)

        # Crash masks for switcher analytic sample
        sw_crash = make_crash_masks(df_sw)

        # ---- Full sample Wald ----
        print(f"\n  --- FULL switcher sample ---", flush=True)
        wald_full = run_wald(df_sw, label=f"k={k}/full")

        # ---- Placebo (MOST IMPORTANT) ----
        print(f"\n  --- PLACEBO test (stopped + rear-ended) [DECISION TEST] ---", flush=True)
        df_pl = df_sw[sw_crash["placebo"]].copy()
        n_pl  = int(sw_crash["placebo"].sum())
        print(f"  Placebo subset: {n_pl:,} records ({100*n_pl/n_a:.1f}% of analytic)",
              flush=True)
        wald_placebo = run_wald(df_pl, label=f"k={k}/placebo")

        # ---- Rollover ----
        print(f"\n  --- ROLLOVER group (ESC active) ---", flush=True)
        df_ro = df_sw[sw_crash["rollover"]].copy()
        n_ro  = int(sw_crash["rollover"].sum())
        print(f"  Rollover subset: {n_ro:,} records ({100*n_ro/n_a:.1f}% of analytic)",
              flush=True)
        wald_rollover = run_wald(df_ro, label=f"k={k}/rollover")

        # ---- DID secular trend ----
        print(f"\n  --- DID secular trend (non-switcher control group) ---", flush=True)

        # Must pass crash masks aligned to FULL df (all rows, not just switchers)
        trend_full     = estimate_secular_trend(df, k, med_ys, crash_mask=None,
                                                label=f"k={k}/full")
        trend_placebo  = estimate_secular_trend(df, k, med_ys,
                                                crash_mask=full_masks["placebo"],
                                                label=f"k={k}/placebo")
        trend_rollover = estimate_secular_trend(df, k, med_ys,
                                                crash_mask=full_masks["rollover"],
                                                label=f"k={k}/rollover")

        # DID estimates
        DID_full     = compute_did(wald_full.get("tau", NAN),
                                   wald_full.get("SE_tau", NAN),
                                   trend_full.get("trend", NAN),
                                   trend_full.get("se_trend", NAN))
        DID_placebo  = compute_did(wald_placebo.get("tau", NAN),
                                   wald_placebo.get("SE_tau", NAN),
                                   trend_placebo.get("trend", NAN),
                                   trend_placebo.get("se_trend", NAN))
        DID_rollover = compute_did(wald_rollover.get("tau", NAN),
                                   wald_rollover.get("SE_tau", NAN),
                                   trend_rollover.get("trend", NAN),
                                   trend_rollover.get("se_trend", NAN))

        # ---- Power ----
        pwr = compute_power(n_z0, n_z1, df_sw["Y_binary"].mean() if n_a > 0 else NAN)

        # ---- Falsification verdict ----
        p_pl = wald_placebo.get("p_value", NAN)
        t_pl = wald_placebo.get("tau",     NAN)
        if np.isnan(p_pl):
            verdict = "INCONCLUSIVE: insufficient N in placebo group"
        elif p_pl < ALPHA_LEVEL and t_pl < 0:
            verdict = f"FAIL: placebo tau={t_pl:+.5f} significantly negative (p={p_pl:.4f}) -> exclusion restriction STILL violated"
        elif p_pl >= ALPHA_LEVEL:
            verdict = f"PASS: placebo p={p_pl:.4f} >= {ALPHA_LEVEL} -> no significant exclusion restriction violation"
        else:
            verdict = f"PASS (positive direction): placebo tau={t_pl:+.5f} p={p_pl:.4f}"

        print(f"\n  FALSIFICATION VERDICT (k={k}): {verdict}", flush=True)
        if not np.isnan(DID_full.get("DID_tau", NAN)):
            print(
                f"  DID tau (full)     = {DID_full['DID_tau']:+.5f}  "
                f"(SE={DID_full['DID_SE']:.5f}  p={DID_full['DID_p']:.4f})",
                flush=True,
            )

        all_results[k] = {
            "k": k,
            "n_switchers":    n_sw,
            "median_y_star":  med_ys,
            "y_star_values":  info["y_star_values"],
            "n_analytic":     n_a,
            "n_Z0":           n_z0,
            "n_Z1":           n_z1,
            "mean_Y":         float(df_sw["Y_binary"].mean()) if n_a > 0 else NAN,
            "wald_full":      wald_full,
            "wald_placebo":   wald_placebo,
            "wald_rollover":  wald_rollover,
            "trend_full":     trend_full,
            "trend_placebo":  trend_placebo,
            "trend_rollover": trend_rollover,
            "DID_full":       DID_full,
            "DID_placebo":    DID_placebo,
            "DID_rollover":   DID_rollover,
            "power":          pwr,
            "falsification_verdict": verdict,
        }

    # ------------------------------------------------------------------
    # STEP 4: Summary table
    # ------------------------------------------------------------------
    print(f"\n{'='*72}", flush=True)
    print("CHANGEOVER DESIGN SUMMARY TABLE", flush=True)
    print(f"{'='*72}", flush=True)

    def _s(v: float, fmt: str = "+.5f") -> str:
        return format(v, fmt) if not np.isnan(v) else "   N/A  "

    header = (f"{'k':>2}  {'Group':<12}  {'n_subset':>9}  {'tau_hat':>10}  "
              f"{'SE':>7}  {'p':>8}  {'RF':>9}  {'DID_tau':>9}  {'DID_p':>8}")
    print(header, flush=True)
    print("-" * 85, flush=True)

    for k in K_VALUES:
        r = all_results.get(k, {})
        if "error" in r:
            print(f"{k:>2}  {'ERROR':>12}", flush=True)
            continue
        for grp, wkey, dkey in [
            ("full",     "wald_full",     "DID_full"),
            ("placebo",  "wald_placebo",  "DID_placebo"),
            ("rollover", "wald_rollover", "DID_rollover"),
        ]:
            w = r.get(wkey, {})
            d = r.get(dkey, {})
            tau = w.get("tau", NAN); se = w.get("SE_tau", NAN)
            p   = w.get("p_value", NAN); rf = w.get("RF", NAN)
            n   = w.get("n_subset", 0) or 0
            did = d.get("DID_tau", NAN); dp = d.get("DID_p", NAN)
            sig = "**" if (not np.isnan(p) and p < ALPHA_LEVEL) else "  "
            print(
                f"{k:>2}  {grp:<12}  {n:>9,}  {_s(tau):>8}{sig}  "
                f"{_s(se, '.5f'):>7}  {_s(p, '.4f'):>8}  "
                f"{_s(rf):>9}  {_s(did):>9}  {_s(dp, '.4f'):>8}",
                flush=True,
            )

    print("-" * 85, flush=True)
    print("** = significant at alpha=0.05", flush=True)

    # Headline
    print(f"\n{'='*72}", flush=True)
    print("HEADLINE: Does the changeover design pass the placebo test?", flush=True)
    print(f"{'='*72}", flush=True)
    for k in K_VALUES:
        r = all_results.get(k, {})
        if "error" in r:
            print(f"  k={k}: ERROR", flush=True)
            continue
        print(f"  k={k}: {r.get('falsification_verdict', 'N/A')}", flush=True)

    print(f"\n  Reference: v2 placebo tau=-0.0585 (p<0.0001) [FAILED]", flush=True)
    print(f"  Reference: v2 main   tau=-0.0397 (p<0.0001)", flush=True)

    # ------------------------------------------------------------------
    # STEP 5: Save
    # ------------------------------------------------------------------
    print("\n[STEP 5] Saving results", flush=True)
    ts      = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    jpath   = RESULTS_DIR / f"changeover_design_{ts}.json"

    payload = {
        "env_log": get_env_log(n_raw),
        "design": {
            "name":                "Within make-model changeover (NHTSA Dang 2007 approach)",
            "k_values":            K_VALUES,
            "pre_window":          "[y*-k, y*-1], all 'No'",
            "post_window":         "[y*, y*+k-1], all 'Standard'",
            "optional_handling":   "disqualifies make/model for this k",
            "list_coverage":       f"[{MY_MIN_COVERAGE}, {MY_MAX_COVERAGE}]",
            "T_assignment":        "T = Z_within (pre=0, post=1); FS = 1.0 by construction",
            "estimator":           "_hajek_wald (identical to v2, unchanged)",
            "DID_reference":       "R = round(median y* for each k); control = non-switcher",
            "placebo_def":         "P_CRASH1==5 AND MAN_COLL==1",
            "rollover_def":        "ROLLOVER in {1,2,9}",
            "alpha":               ALPHA_LEVEL,
            "decision_rule":       f"placebo p >= {ALPHA_LEVEL} -> PASS, p < {ALPHA_LEVEL} -> FAIL",
            "n_control_no":        len(all_no),
            "n_control_std":       len(all_std),
        },
        "k_results": {str(k): v for k, v in all_results.items()},
        "v2_reference":         {"tau": -0.03971, "RF": -0.02411, "FS": 0.60724},
        "v2_placebo_reference": {"tau": -0.0585,  "p_value": "<0.0001", "verdict": "FAILED"},
    }

    with open(jpath, "w") as f:
        json.dump(payload, f, indent=2, allow_nan=True)
    print(f"\nSaved: {jpath}", flush=True)
    print(f"End: {datetime.utcnow().isoformat()}Z", flush=True)
    print("=" * 72, flush=True)


if __name__ == "__main__":
    main()
