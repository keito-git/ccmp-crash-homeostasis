"""
Real-Data Tier-2 Point Test -- Dang/Sivinski Sensitivity Analysis (v2: fast)
===========================================================================
Script: realdata_ds_sensitivity.py (NEW file, does NOT modify existing scripts)

T SOURCE: Dang 2007 / Sivinski 2011 strict list (esc_equipment_list.csv)
This is the CROSS-VALIDATION arm of the Tier-2 analysis.
The primary analysis (NHTSA API T source) is realdata_tier2_pointtest.py.

Changes from v1:
  - Vectorized T assignment via pd.merge (not row-wise apply)
  - Numpy-based bootstrap (no pandas overhead)
  - sys.stdout.flush() after each print (force output in buffered mode)

ANALYSIS DEFINITION (frozen 2026-07-14, identical to primary except T source):
  Z=0: MY in [2000, 2010]
  Z=1: MY >= 2012
  MY 2011 excluded
  T: from esc_equipment_list.csv (Dang 2007 / Sivinski 2011)
     Vectorized: expand each DS entry to all MY in [no_esc_min_year, esc_std_year-1]->T=0
                 expand each DS entry to all MY in [esc_std_year, 2010]->T=1
     Merge on (make_upper, model_kw, MOD_YEAR)
  Y = I(INJ_SEV in {1,2,3,4}) [primary]
  W = CRSS person WEIGHT
  H0: tau=0  alpha=0.05  B=500 bootstrap  seed=42

FABRICATION PROHIBITION:
  All numbers come from actual execution. No results manually entered.
"""

from __future__ import annotations

import json
import platform
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import scipy.stats

# Force line-buffered output even when redirected
sys.stdout.reconfigure(line_buffering=True)  # Python 3.7+

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR    = Path(__file__).resolve().parents[1]
DATA_DIR    = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

ESC_LIST_CSV = DATA_DIR / "esc_equipment_list.csv"

SCRATCHPAD = Path("/private/tmp/claude-501/-Users-redacted-user/"
                  "373ef59a-33a6-4195-ad55-30c70723863d/scratchpad")

CRSS_YEARS = [2016, 2017, 2018, 2019, 2020, 2021]

# ---------------------------------------------------------------------------
# Pre-specified constants (identical to primary script)
# ---------------------------------------------------------------------------
Z_PRE_ESC_MAX_MY  = 2010
Z_POST_ESC_MIN_MY = 2012
MY_MIN_COVERAGE   = 2000

INJ_SEV_VALID  = {0, 1, 2, 3, 4}
INJ_SEV_INJURY = {1, 2, 3, 4}

ALPHA_LEVEL    = 0.05
N_BOOTSTRAP    = 500
BOOTSTRAP_SEED = 42


# ---------------------------------------------------------------------------
# Dang/Sivinski lookup table: expand to per-(make, model_kw, MOD_YEAR) rows
# ---------------------------------------------------------------------------

def build_ds_lookup(ds_list: pd.DataFrame) -> pd.DataFrame:
    """
    Expand Dang/Sivinski list to a flat lookup table with one row per
    (make_upper, model_kw, MOD_YEAR, T_assign).
    T_assign = 0.0 for MY in [no_esc_min_year, no_esc_max_year]
    T_assign = 1.0 for MY in [esc_std_year, 2010]  (always-taker in pre-mandate)

    This allows O(1) merge-based T assignment instead of O(N) row-wise apply.
    """
    rows = []
    for _, row in ds_list.iterrows():
        make     = str(row["make"]).upper().strip()
        model_kw = str(row["model_kw"]).upper().strip()
        no_min   = int(row["no_esc_min_year"])
        no_max   = int(row["no_esc_max_year"])
        std      = int(row["esc_std_year"])

        # T=0 years: [no_esc_min_year, no_esc_max_year]
        for my in range(no_min, no_max + 1):
            rows.append({"MAKE_DS": make, "MODEL_KW": model_kw, "MOD_YEAR": my, "T_DS": 0.0})

        # T=1 years (pre-mandate, always-taker): [esc_std_year, 2010]
        for my in range(std, Z_PRE_ESC_MAX_MY + 1):
            rows.append({"MAKE_DS": make, "MODEL_KW": model_kw, "MOD_YEAR": my, "T_DS": 1.0})

    return pd.DataFrame(rows)


def assign_t_vectorized(df: pd.DataFrame, ds_lookup: pd.DataFrame) -> pd.DataFrame:
    """
    Vectorized T assignment: cross-join on MAKE_API + MODEL_KW prefix match + MOD_YEAR.

    Strategy: for each DS model_kw, find all CRSS rows where:
      1. MAKE_API.upper() == MAKE_DS
      2. MODEL_API.upper().startswith(MODEL_KW)  (word-boundary prefix match)
      3. MOD_YEAR matches

    We implement this by iterating over unique (MAKE_DS, MODEL_KW) pairs and
    doing a vectorized merge for each pair. Number of DS entries = 52, so this is
    52 merge operations (not 600k row-wise calls).
    """
    df = df.copy()
    df["T"] = np.nan

    # Normalize CRSS make/model
    df["_make_up"]  = df["MAKE_API"].fillna("").str.upper().str.strip()
    df["_model_up"] = df["MODEL_API"].fillna("").str.upper().str.strip()

    # For post-mandate Z=1: T=1 by FMVSS 126
    df.loc[df["Z"] == 1.0, "T"] = 1.0

    # For pre-mandate Z=0: merge with DS lookup
    pre_mask = (df["Z"] == 0.0)

    # Get unique (MAKE_DS, MODEL_KW) pairs in DS lookup
    unique_pairs = ds_lookup[["MAKE_DS", "MODEL_KW"]].drop_duplicates()

    for _, pair in unique_pairs.iterrows():
        make_ds   = pair["MAKE_DS"]
        model_kw  = pair["MODEL_KW"]

        # Match: CRSS make == DS make AND CRSS model startswith DS keyword
        match_mask = (
            pre_mask &
            (df["_make_up"] == make_ds) &
            (
                (df["_model_up"] == model_kw) |
                df["_model_up"].str.startswith(model_kw + " ", na=False)
            )
        )

        if not match_mask.any():
            continue

        # Get T values for this (MAKE_DS, MODEL_KW, MOD_YEAR) from DS lookup
        pair_lu = ds_lookup[
            (ds_lookup["MAKE_DS"] == make_ds) &
            (ds_lookup["MODEL_KW"] == model_kw)
        ][["MOD_YEAR", "T_DS"]].copy()
        pair_lu["MOD_YEAR"] = pair_lu["MOD_YEAR"].astype(int)

        # Merge on MOD_YEAR for matching rows
        matched_idx = df[match_mask].index
        sub = df.loc[matched_idx, ["MOD_YEAR"]].copy()
        sub["MOD_YEAR"] = pd.to_numeric(sub["MOD_YEAR"], errors="coerce").astype("Int64")
        sub = sub.merge(pair_lu, on="MOD_YEAR", how="left")
        df.loc[matched_idx, "T"] = sub["T_DS"].values

    df.drop(columns=["_make_up", "_model_up"], inplace=True)
    return df


# ---------------------------------------------------------------------------
# CRSS Data Loading
# ---------------------------------------------------------------------------

def _find_csv(crss_dir: Path, name: str) -> Optional[Path]:
    for p in crss_dir.iterdir():
        if p.is_file() and p.name.upper() == name.upper():
            return p
    for sub in crss_dir.iterdir():
        if sub.is_dir():
            for p in sub.iterdir():
                if p.is_file() and p.name.upper() == name.upper():
                    return p
    return None


def load_crss_year(year: int) -> Optional[pd.DataFrame]:
    crss_dir = SCRATCHPAD / f"crss{year}"
    if not crss_dir.exists():
        print(f"  CRSS {year} directory not found", flush=True)
        return None

    acc_path = _find_csv(crss_dir, "accident.csv")
    veh_path = _find_csv(crss_dir, "vehicle.csv")
    per_path = _find_csv(crss_dir, "person.csv")
    vpc_path = _find_csv(crss_dir, "vpicdecode.csv")

    missing = [n for n, p in [("accident",acc_path),("vehicle",veh_path),
                                ("person",per_path),("vpicdecode",vpc_path)] if p is None]
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

    acc_cols = [c for c in ["CASENUM","PSU","PSUSTRAT","STRATUM","WEIGHT","REGION","URBANICITY","INT_HWY"] if c in df_acc.columns]
    df_acc = df_acc[acc_cols].rename(columns={"PSU":"PSU_ACC","PSUSTRAT":"PSUSTRAT_ACC","WEIGHT":"WEIGHT_ACC"})

    veh_cols = [c for c in ["CASENUM","VEH_NO","MOD_YEAR"] if c in df_veh.columns]
    df_veh   = df_veh[veh_cols]

    per_cols = [c for c in ["CASENUM","VEH_NO","INJ_SEV","WEIGHT"] if c in df_per.columns]
    df_per   = df_per[per_cols].rename(columns={"WEIGHT":"WEIGHT_PER"})

    vpc_cols = ["CASENUM","VEH_NO"]
    if "MAKE" in df_vpc.columns:  vpc_cols.append("MAKE")
    if "MODEL" in df_vpc.columns: vpc_cols.append("MODEL")
    df_vpc = df_vpc[vpc_cols].rename(columns={"MAKE":"MAKE_API","MODEL":"MODEL_API"})

    df = pd.merge(df_per, df_veh, on=["CASENUM","VEH_NO"], how="inner")
    df = pd.merge(df, df_acc, on="CASENUM", how="left")
    df = pd.merge(df, df_vpc, on=["CASENUM","VEH_NO"], how="left")
    df["CRSS_YEAR"] = year

    for col in ["MOD_YEAR","INJ_SEV","WEIGHT_PER","WEIGHT_ACC"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    print(f"  CRSS {year}: {len(df):,} records", flush=True)
    return df


def pool_crss() -> pd.DataFrame:
    frames = [f for f in (load_crss_year(y) for y in CRSS_YEARS) if f is not None]
    if not frames:
        raise RuntimeError("No CRSS data.")
    df = pd.concat(frames, ignore_index=True)
    print(f"Pooled: {len(df):,} records across years {[int(f['CRSS_YEAR'].iloc[0]) for f in frames]}", flush=True)
    return df


# ---------------------------------------------------------------------------
# Variable Assignment
# ---------------------------------------------------------------------------

def assign_Z_Y(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    my = df["MOD_YEAR"]
    df["Z"] = np.nan
    df.loc[(my >= MY_MIN_COVERAGE) & (my <= Z_PRE_ESC_MAX_MY), "Z"] = 0.0
    df.loc[my >= Z_POST_ESC_MIN_MY, "Z"] = 1.0

    df["Z_wide"] = np.nan
    df.loc[(my >= MY_MIN_COVERAGE) & (my <= 2009), "Z_wide"] = 0.0
    df.loc[my >= Z_POST_ESC_MIN_MY, "Z_wide"] = 1.0

    df["Y_binary"] = np.nan
    valid = df["INJ_SEV"].isin(INJ_SEV_VALID)
    df.loc[valid, "Y_binary"] = df.loc[valid, "INJ_SEV"].isin(INJ_SEV_INJURY).astype(float)

    df["W"] = df["WEIGHT_PER"].fillna(0.0)
    df.loc[df["W"] <= 0, "W"] = np.nan
    return df


# ---------------------------------------------------------------------------
# IV Wald (faithful copy of hc_t2_reduction.py _ipsw_wald)
# ---------------------------------------------------------------------------

def _hajek_wald(Y: np.ndarray, T: np.ndarray, Z: np.ndarray, W: np.ndarray) -> Dict:
    m1 = (Z == 1); m0 = (Z == 0)
    n1, n0 = int(m1.sum()), int(m0.sum())
    if n1 < 10 or n0 < 10:
        nan = float("nan")
        return {"tau": nan, "SE_tau": nan, "CI_L": nan, "CI_U": nan,
                "z_stat": nan, "p_value": nan, "reject": False,
                "FS": nan, "RF": nan, "n_Z0": n0, "n_Z1": n1}
    w1, w0 = W[m1], W[m0]
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
    var_RF = np.dot(w1**2, resY_Z1**2)/sw1**2 + np.dot(w0**2, resY_Z0**2)/sw0**2
    var_FS = np.dot(w1**2, resT_Z1**2)/sw1**2 + np.dot(w0**2, resT_Z0**2)/sw0**2
    SE_tau = float(np.sqrt(max(1e-20, var_RF/FS**2 + RF**2*var_FS/FS**4)))
    z_stat = tau / SE_tau
    p_val  = float(2.0 * scipy.stats.norm.sf(abs(z_stat)))
    return {
        "tau": float(tau), "SE_tau": SE_tau,
        "CI_L": float(tau - 1.96*SE_tau), "CI_U": float(tau + 1.96*SE_tau),
        "z_stat": float(z_stat), "p_value": p_val, "reject": bool(p_val < ALPHA_LEVEL),
        "FS": float(FS), "RF": float(RF), "RF": float(RF),
        "EY_Z1": float(EY_Z1), "EY_Z0": float(EY_Z0),
        "ET_Z1": float(ET_Z1), "ET_Z0": float(ET_Z0),
        "n_Z0": n0, "n_Z1": n1,
    }


def first_stage_fstat(T: np.ndarray, Z: np.ndarray, W: np.ndarray) -> Dict:
    m1 = (Z == 1); m0 = (Z == 0)
    w1, w0 = W[m1], W[m0]
    sw1, sw0 = w1.sum(), w0.sum()
    if sw1 < 1e-10 or sw0 < 1e-10:
        return {"FS": float("nan"), "F_stat": float("nan")}
    ET_Z1 = float(np.dot(T[m1], w1) / sw1)
    ET_Z0 = float(np.dot(T[m0], w0) / sw0)
    FS    = ET_Z1 - ET_Z0
    resT_Z1 = T[m1] - ET_Z1; resT_Z0 = T[m0] - ET_Z0
    var_FS  = np.dot(w1**2, resT_Z1**2)/sw1**2 + np.dot(w0**2, resT_Z0**2)/sw0**2
    SE_FS   = float(np.sqrt(max(1e-20, var_FS)))
    F_stat  = float((FS / SE_FS)**2) if SE_FS > 1e-10 else float("inf")
    return {"FS": float(FS), "SE_FS": SE_FS, "F_stat": F_stat,
            "ET_Z0": float(ET_Z0), "ET_Z1": float(ET_Z1)}


# ---------------------------------------------------------------------------
# Bootstrap CI (numpy-based: much faster than pandas-based)
# ---------------------------------------------------------------------------

def bootstrap_wald_np(
    Y: np.ndarray, T: np.ndarray, Z: np.ndarray, W: np.ndarray,
    caseid: np.ndarray, stratid: np.ndarray,
    B: int = N_BOOTSTRAP, seed: int = BOOTSTRAP_SEED,
) -> Dict:
    """
    Stratified cluster bootstrap (numpy-based, no pandas overhead).
    Clusters = caseid (accident), strata = stratid (PSUSTRAT × CRSS_YEAR).

    For each bootstrap iteration:
    1. For each stratum, resample clusters with replacement
    2. Pool all resampled observations
    3. Compute IV Wald estimator
    """
    rng = np.random.default_rng(seed)

    unique_strats = np.unique(stratid)
    # Build index arrays: for each stratum, list of observation indices per cluster
    # This avoids rebuilding these lists in each iteration
    strat_cluster_obs: Dict = {}  # stratum -> {cluster_id -> obs_indices}
    for s in unique_strats:
        s_mask = (stratid == s)
        s_idx  = np.where(s_mask)[0]
        s_cases = caseid[s_mask]
        unique_cases = np.unique(s_cases)
        cluster_map  = {c: s_idx[s_cases == c] for c in unique_cases}
        strat_cluster_obs[s] = cluster_map

    bs_taus = []
    for b in range(B):
        resampled_idx = []
        for s, cluster_map in strat_cluster_obs.items():
            clusters = list(cluster_map.keys())
            n_c = len(clusters)
            sampled = rng.choice(clusters, size=n_c, replace=True)
            for c in sampled:
                resampled_idx.append(cluster_map[c])

        if not resampled_idx:
            continue
        idx   = np.concatenate(resampled_idx)
        Y_bs  = Y[idx]; T_bs = T[idx]; Z_bs = Z[idx]; W_bs = W[idx]
        res   = _hajek_wald(Y_bs, T_bs, Z_bs, W_bs)
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
    power   = float(1 - scipy.stats.norm.cdf(z_alpha - ncp) + scipy.stats.norm.cdf(-z_alpha - ncp))
    mde     = float((z_alpha + z_beta) * se_hat)
    ref = {}
    for t in [0.015, 0.030, 0.060]:
        ncp2 = t / se_hat
        ref[f"|tau|={t:.3f}"] = round(float(
            1 - scipy.stats.norm.cdf(z_alpha - ncp2) + scipy.stats.norm.cdf(-z_alpha - ncp2)), 3)
    return {"power_at_obs_tau": round(power, 3), "mde_80pct": round(mde, 4),
            "power_at_reference_tau": ref}


def get_env_log(n_raw: int, n_analytic: int) -> Dict:
    try:
        git = subprocess.check_output(["git","rev-parse","HEAD"],
                    cwd=str(BASE_DIR), stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        git = "unavailable"
    import numpy as _np
    import scipy as _sc
    return {
        "timestamp": datetime.utcnow().isoformat()+"Z",
        "script": "realdata_ds_sensitivity.py (v2)",
        "python_version": sys.version,
        "numpy_version": _np.__version__,
        "pandas_version": pd.__version__,
        "scipy_version": _sc.__version__,
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "git_commit": git,
        "GPU": "none (CPU-only)",
        "crss_years": CRSS_YEARS,
        "n_raw": n_raw,
        "n_analytic": n_analytic,
        "T_source": "Dang 2007 / Sivinski 2011 (esc_equipment_list.csv)",
        "analysis_arm": "Cross-validation / sensitivity",
        "analysis_freeze_date": "2026-07-14",
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> Dict:
    print("="*72, flush=True)
    print("Dang/Sivinski Sensitivity Analysis v2 -- CCMP Tier-2 Point Test", flush=True)
    print(f"Start: {datetime.utcnow().isoformat()}Z", flush=True)
    print("="*72, flush=True)
    print("\n[NOTE] T source = Dang 2007 + Sivinski 2011 (cross-validation arm)", flush=True)
    print("       Primary T source (NHTSA API) is still caching in background.", flush=True)

    print(f"\n[STEP 1] Load Dang/Sivinski ESC list", flush=True)
    ds_raw  = pd.read_csv(ESC_LIST_CSV)
    ds_raw["make"]     = ds_raw["make"].str.upper().str.strip()
    ds_raw["model_kw"] = ds_raw["model_kw"].str.upper().str.strip()
    ds_raw["esc_std_year"]    = ds_raw["esc_std_year"].astype(int)
    ds_raw["no_esc_min_year"] = ds_raw["no_esc_min_year"].astype(int)
    ds_raw["no_esc_max_year"] = ds_raw["no_esc_max_year"].astype(int)
    print(f"  {len(ds_raw)} model entries loaded", flush=True)

    ds_lookup = build_ds_lookup(ds_raw)
    print(f"  DS lookup expanded: {len(ds_lookup):,} (make, model_kw, MY, T) rows", flush=True)

    print(f"\n[STEP 2] Load CRSS pooled data", flush=True)
    df_raw = pool_crss()
    n_raw  = len(df_raw)
    print(f"  Total records: {n_raw:,}", flush=True)

    print(f"\n[STEP 3] Assign Z, Y", flush=True)
    df = assign_Z_Y(df_raw)
    print(f"  Z=0: {(df['Z']==0).sum():,}  Z=1: {(df['Z']==1).sum():,}  Z=NA: {df['Z'].isna().sum():,}", flush=True)

    print(f"\n[STEP 4] Vectorized T assignment from Dang/Sivinski", flush=True)
    df = assign_t_vectorized(df, ds_lookup)

    # Coverage report
    z0 = (df["Z"] == 0.0) & df["MOD_YEAR"].notna()
    n_z0  = int(z0.sum())
    n_t0  = int((z0 & (df["T"] == 0.0)).sum())
    n_t1  = int((z0 & (df["T"] == 1.0)).sum())
    n_tna = int((z0 & df["T"].isna()).sum())
    n_z1  = int((df["Z"] == 1.0).sum())

    print(f"  Z=0 total persons:  {n_z0:,}", flush=True)
    print(f"    T=0 (no ESC):     {n_t0:,} ({100*n_t0/max(n_z0,1):.1f}%)", flush=True)
    print(f"    T=1 (std ESC):    {n_t1:,} ({100*n_t1/max(n_z0,1):.1f}%)", flush=True)
    print(f"    T=unassigned:     {n_tna:,} ({100*n_tna/max(n_z0,1):.1f}%)", flush=True)
    print(f"  Z=1 total (T=1):    {n_z1:,}", flush=True)

    # Analytic sample
    mask = (df["Z"].notna() & df["T"].notna() & df["Y_binary"].notna() &
            df["W"].notna() & (df["W"] > 0))
    df_a = df[mask].copy()
    n_analytic = len(df_a)
    naz0 = int((df_a["Z"] == 0).sum())
    naz1 = int((df_a["Z"] == 1).sum())
    print(f"\n  Analytic sample: N={n_analytic:,}  (Z=0: {naz0:,}, Z=1: {naz1:,})", flush=True)

    # Extract arrays for analysis
    Y_a = df_a["Y_binary"].values.astype(float)
    T_a = df_a["T"].values.astype(float)
    Z_a = df_a["Z"].values.astype(float)
    W_a = df_a["W"].values.astype(float)

    # Composite strat id for bootstrap
    strat_col = "PSUSTRAT_ACC" if "PSUSTRAT_ACC" in df_a.columns else "STRATUM"
    casenum   = df_a["CASENUM"].fillna(0).astype(int).values if "CASENUM" in df_a.columns else np.arange(len(df_a))
    year_arr  = df_a["CRSS_YEAR"].fillna(0).astype(int).values
    strat_arr = df_a[strat_col].fillna(0).astype(int).values if strat_col in df_a.columns else np.zeros(len(df_a), int)
    # Composite strat: year*1000 + PSUSTRAT (unique across years)
    comp_strat = year_arr * 1000 + strat_arr
    # Composite case id: year*1_000_000 + CASENUM (unique across years)
    comp_case  = year_arr * 1_000_000 + casenum

    print(f"\n[STEP 5] First stage", flush=True)
    fs = first_stage_fstat(T_a, Z_a, W_a)
    print(f"  FS = {fs['FS']:.4f}  SE_FS = {fs.get('SE_FS',float('nan')):.5f}  F = {fs['F_stat']:.1f}", flush=True)
    if fs["F_stat"] < 10:
        print("  [WARN] F-stat < 10: weak instrument", flush=True)

    print(f"\n[STEP 6] IV Wald primary", flush=True)
    wald = _hajek_wald(Y_a, T_a, Z_a, W_a)
    tau = wald["tau"]; se = wald["SE_tau"]
    print(f"  tau_hat = {tau:.5f}", flush=True)
    print(f"  SE      = {se:.5f}", flush=True)
    print(f"  95% CI  = [{wald['CI_L']:.5f}, {wald['CI_U']:.5f}]", flush=True)
    print(f"  z-stat  = {wald['z_stat']:.3f}  p-value = {wald['p_value']:.4f}", flush=True)
    print(f"  H0: tau=0 -> {'REJECT' if wald['reject'] else 'FAIL TO REJECT'} at alpha=0.05", flush=True)

    # Z_wide robustness
    print(f"\n  [Robustness: Z_wide (pre-mandate MY<=2009)]", flush=True)
    mask_w = (df["Z_wide"].notna() & df["T"].notna() & df["Y_binary"].notna() &
              df["W"].notna() & (df["W"] > 0))
    df_w   = df[mask_w]
    wald_w = _hajek_wald(df_w["Y_binary"].values.astype(float),
                         df_w["T"].values.astype(float),
                         df_w["Z_wide"].values.astype(float),
                         df_w["W"].values.astype(float))
    print(f"  tau_hat (Z_wide) = {wald_w['tau']:.5f}  p = {wald_w['p_value']:.4f}  n_Z0={wald_w['n_Z0']:,}", flush=True)

    print(f"\n[STEP 7] Bootstrap CI (B={N_BOOTSTRAP}, numpy cluster bootstrap)", flush=True)
    bs = bootstrap_wald_np(Y_a, T_a, Z_a, W_a, comp_case, comp_strat)
    print(f"  Bootstrap 95% CI: [{bs['CI_L_boot']:.5f}, {bs['CI_U_boot']:.5f}]", flush=True)
    print(f"  SE_boot = {bs['SE_boot']:.5f}  (n_valid={bs['n_boot_valid']})", flush=True)

    print(f"\n[STEP 8] Power", flush=True)
    pwr = assess_power(tau, se)
    print(f"  Power at |tau|={abs(tau):.4f}: {pwr['power_at_obs_tau']:.3f}", flush=True)
    print(f"  MDE 80% power: {pwr['mde_80pct']:.4f}", flush=True)
    print(f"  Power at reference tau: {pwr['power_at_reference_tau']}", flush=True)

    print(f"\n  [HONEST EVALUATION]", flush=True)
    if wald["reject"]:
        print(f"  H0 REJECTED at alpha=0.05. tau={'<' if tau<0 else '>'}0.", flush=True)
        if tau < 0:
            print(f"  ESC reduces injury risk: kappa_LATE < 1.", flush=True)
        else:
            print(f"  [UNEXPECTED] ESC increases injury risk: kappa_LATE > 1 -- check carefully.", flush=True)
    else:
        print(f"  FAIL TO REJECT H0 at alpha=0.05.", flush=True)
        print(f"  Power = {pwr['power_at_obs_tau']:.3f}. Non-rejection is consistent with", flush=True)
        print(f"  BOTH tau=0 (true homeostasis) AND insufficient power.", flush=True)

    # Save
    stem = f"ds_sensitivity_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    aggregate = {
        "env_log": get_env_log(n_raw, n_analytic),
        "coverage": {
            "n_z0": n_z0, "n_t0": n_t0, "n_t1": n_t1,
            "n_tna": n_tna, "n_z1": n_z1, "n_analytic": n_analytic,
            "pct_T0_z0": round(100*n_t0/max(n_z0,1),2),
            "pct_T1_z0": round(100*n_t1/max(n_z0,1),2),
            "pct_Tna_z0": round(100*n_tna/max(n_z0,1),2),
        },
        "first_stage": fs,
        "wald_primary": wald,
        "wald_Z_wide": wald_w,
        "bootstrap": bs,
        "power": pwr,
        "analysis_arm": "Dang/Sivinski cross-validation (T from Dang 2007 / Sivinski 2011)",
        "note": "Primary analysis (NHTSA API T source) still caching.",
    }

    json_path = RESULTS_DIR / f"{stem}.json"
    with open(json_path, "w") as f:
        json.dump(aggregate, f, indent=2, allow_nan=True)
    print(f"\nSaved: {json_path}", flush=True)

    print("\n" + "="*72, flush=True)
    print("FINAL SUMMARY (Dang/Sivinski cross-validation arm)", flush=True)
    print("="*72, flush=True)
    print(f"Dataset    : CRSS pooled {CRSS_YEARS}", flush=True)
    print(f"N_analytic : {n_analytic:,}", flush=True)
    print(f"FS (1st stage): {fs['FS']:.4f}  F={fs['F_stat']:.1f}", flush=True)
    print(f"tau_hat  = {tau:.5f}  SE={se:.5f}", flush=True)
    print(f"95% CI   = [{wald['CI_L']:.5f}, {wald['CI_U']:.5f}]  (delta method)", flush=True)
    print(f"95% CI   = [{bs['CI_L_boot']:.5f}, {bs['CI_U_boot']:.5f}]  (bootstrap)", flush=True)
    print(f"z-stat   = {wald['z_stat']:.3f}  p-value = {wald['p_value']:.4f}", flush=True)
    print(f"Decision : {'REJECT H0' if wald['reject'] else 'FAIL TO REJECT H0'} at alpha=0.05", flush=True)
    print(f"Power    : {pwr['power_at_obs_tau']:.3f} at observed |tau|", flush=True)
    print(f"\nEnd: {datetime.utcnow().isoformat()}Z", flush=True)
    print("="*72, flush=True)

    return aggregate


if __name__ == "__main__":
    result = main()
