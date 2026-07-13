"""
Confirmatory re-run post-freeze for HC_T2(b)(c), HC_NI, and HC3(G4).
Imports computation functions from frozen scripts, runs with frozen parameters,
saves results to new output files, and applies Bonferroni-Holm correction (K=5 hypotheses).
Outputs: results/confirmatory_rerun_postfreeze_2026-07-11.csv, results/confirmatory_rerun_postfreeze_2026-07-11.json
"""

from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

# Paths  (same convention as existing frozen scripts — hardcoded project root)
BASE_DIR    = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = BASE_DIR / "scripts"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Pilot (existing) result files to verify against
PILOT_HC_T2_CSV   = RESULTS_DIR / "hc_t2_reduction_per_cell.csv"
PILOT_HC_T2_SP    = RESULTS_DIR / "hc_t2_reduction_size_power.csv"
PILOT_HC_NI_CSV   = RESULTS_DIR / "non_identification_counterexample.csv"
PILOT_HC_NI_JSON  = RESULTS_DIR / "non_identification_counterexample.json"
PILOT_G4_JSON     = RESULTS_DIR / "kappa_g4_recalibration.json"

# New output files
OUT_CSV  = RESULTS_DIR / "confirmatory_rerun_postfreeze_2026-07-11.csv"
OUT_JSON = RESULTS_DIR / "confirmatory_rerun_postfreeze_2026-07-11.json"

# Import from frozen scripts (NO main() calls)
sys.path.insert(0, str(SCRIPTS_DIR))

# HC_T2(b)(c) — hc_t2_reduction.py
from hc_t2_reduction import (
    run_calibration_seed as hc_t2_calib_seed,
    run_size_power as hc_t2_size_power,
    aggregate_cell as hc_t2_aggregate,
    sigmoid,
    SEEDS     as HC_T2_SEEDS,
    N_FINAL   as HC_T2_N,
    M_REPS    as HC_T2_M,
    N_REP     as HC_T2_NREP,
    TIER2_CELLS,
    ALPHA_LEVEL,
)

# HC_NI — non_identification_counterexample.py
from non_identification_counterexample import (
    run_t3i_one_seed,
    run_t3ii_one_seed,
    SEEDS   as NI_SEEDS,
    N_FINAL as NI_N,
)

# HC3(G4) — kappa_grid_g4_recalibration.py
from kappa_grid_g4_recalibration import (
    compute_kappa as g4_compute_kappa,
    binary_search_alpha_b as g4_binary_search,
    SEEDS   as G4_SEEDS,
    N_FINAL as G4_N,
)


# Helpers

def get_git_commit() -> str:
    """Return git HEAD hash of pilot/ sub-repo."""
    try:
        pilot_dir = str(BASE_DIR / "pilot")
        h = subprocess.check_output(
            ["git", "-C", pilot_dir, "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        return h
    except Exception:
        return "unknown"


def bonferroni_holm(p_values: List[float], alpha: float = 0.05) -> Dict[str, Any]:
    """
    Bonferroni-Holm stepdown correction (Holm 1979).
    Inputs: list of p-values (NaN means not available, excluded from ranking).
    Returns dict with sorted order, adjusted p-values, and FWER-controlled decisions.
    """
    valid = [(i, p) for i, p in enumerate(p_values) if not np.isnan(p)]
    K = len(valid)
    if K == 0:
        return {"error": "no valid p-values", "adjusted": p_values}

    # Sort ascending by p-value
    valid_sorted = sorted(valid, key=lambda x: x[1])
    adj_p = {}
    running_max = 0.0
    reject = {}
    for rank, (orig_idx, p) in enumerate(valid_sorted):
        remaining = K - rank
        adj = min(1.0, p * remaining)
        running_max = max(running_max, adj)
        adj_p[orig_idx] = running_max
        reject[orig_idx] = running_max <= alpha

    result = {
        "alpha": alpha,
        "K_available": K,
        "K_total_prereg": 5,
        "sorted_order": [(idx, p) for idx, p in valid_sorted],
        "adjusted_p_values": {i: adj_p.get(i, float("nan")) for i in range(len(p_values))},
        "reject_H0": {i: reject.get(i, None) for i in range(len(p_values))},
        "method": "Bonferroni-Holm stepdown (Holm 1979)",
    }
    return result


# Section 1: HC_T2(b)(c) re-run

def run_hc_t2_rerun() -> Tuple[List[Dict], List[Dict]]:
    """Re-run HC_T2(b)(c) calibration + size/power. Returns (per_seed, size_power)."""
    print("\n" + "=" * 68)
    print("SECTION 1: HC_T2(b)(c) Re-run (hc_t2_reduction logic)")
    print(f"  Seeds: {HC_T2_SEEDS}  N/seed: {HC_T2_N:,}")
    print(f"  M_REPS: {HC_T2_M}  N_REP: {HC_T2_NREP:,}")
    print("=" * 68)

    all_per_seed: List[Dict] = []
    all_sp: List[Dict] = []

    for spec in TIER2_CELLS:
        cell, rho, alpha_b = spec["cell"], spec["rho"], spec["alpha_b"]
        print(f"\n  --- Cell {cell}: rho={rho}, alpha_b={alpha_b} ---")

        per_seed: List[Dict] = []
        for seed in HC_T2_SEEDS:
            r = hc_t2_calib_seed(cell, rho, alpha_b, seed, HC_T2_N)
            per_seed.append(r)
            print(f"    seed={seed}: tau_LATE={r['tau_LATE_true']:+.6f}  "
                  f"tau_IPSW={r['tau_IV_IPSW']:+.6f}  cov={r['tau_IPSW_covers_true']}")
        all_per_seed.extend(per_seed)

        # Size/power via M reps
        print(f"  Running M={HC_T2_M} reps for size/power...")
        sp = hc_t2_size_power(cell, rho, alpha_b, HC_T2_M, HC_T2_NREP)
        role = "SIZE" if "null" in cell else "POWER"
        print(f"  {cell} [{role}]: reject_IPSW={sp['reject_rate_IPSW']:.4f}  "
              f"SE={sp['SE_reject_IPSW']:.4f}  tau_mean={sp['tau_IV_IPSW_mean']:+.6f}")
        all_sp.append(sp)

    return all_per_seed, all_sp


# Section 2: HC_NI re-run

def run_hc_ni_rerun() -> Dict:
    """Re-run HC_NI (non-identification counterexample)."""
    print("\n" + "=" * 68)
    print("SECTION 2: HC_NI Re-run (non_identification_counterexample logic)")
    print(f"  Seeds: {NI_SEEDS}  N/seed: {NI_N:,}")
    print("=" * 68)

    t3i_results: List[Dict] = []
    t3ii_results: List[Dict] = []

    for seed in NI_SEEDS:
        r_i = run_t3i_one_seed(seed, NI_N)
        r_ii = run_t3ii_one_seed(seed, NI_N)
        t3i_results.append(r_i)
        t3ii_results.append(r_ii)
        print(f"  seed={seed}: T3-i NIE_true={r_i.get('NIE_true', float('nan')):.6f}  "
              f"T3-ii NIE_true={r_ii.get('NIE_true', float('nan')):.6f}")

    # Criterion checks
    def mean_key(results: List[Dict], key: str) -> float:
        vals = [r.get(key, float("nan")) for r in results]
        return float(np.nanmean(vals))

    nie_true_t3i  = mean_key(t3i_results, "NIE_true")
    nie_true_t3ii = mean_key(t3ii_results, "NIE_true")
    nie_naive_t3i  = mean_key(t3i_results, "NIE_naive")
    nie_naive_t3ii = mean_key(t3ii_results, "NIE_naive")

    # HC_NI criterion checks
    # (a): arm-wise S distribution match |diff| < 0.05
    diff_E_S_T0 = abs(mean_key(t3i_results, "E_S_T0") - mean_key(t3ii_results, "E_S_T0"))
    diff_E_S_T1 = abs(mean_key(t3i_results, "E_S_T1") - mean_key(t3ii_results, "E_S_T1"))
    diff_SD_T0  = abs(mean_key(t3i_results, "SD_S_T0") - mean_key(t3ii_results, "SD_S_T0"))
    diff_SD_T1  = abs(mean_key(t3i_results, "SD_S_T1") - mean_key(t3ii_results, "SD_S_T1"))
    crit_a = (diff_E_S_T0 < 0.05 and diff_E_S_T1 < 0.05
              and diff_SD_T0 < 0.05 and diff_SD_T1 < 0.05)

    # (b): naive estimator same |diff_naive_NIE| < 0.001
    diff_naive = abs(nie_naive_t3i - nie_naive_t3ii)
    crit_b = diff_naive < 0.001

    # (c): T3-i NIE_true > 0.01, T3-ii NIE_true < 0.001
    nie_t3i_vals = [r.get("NIE_true", float("nan")) for r in t3i_results]
    nie_t3ii_vals = [r.get("NIE_true", float("nan")) for r in t3ii_results]
    ci_t3i_L = float(np.nanmean(nie_t3i_vals) - 1.96 * np.nanstd(nie_t3i_vals, ddof=1) / np.sqrt(len(NI_SEEDS)))
    ci_t3i_U = float(np.nanmean(nie_t3i_vals) + 1.96 * np.nanstd(nie_t3i_vals, ddof=1) / np.sqrt(len(NI_SEEDS)))
    crit_c = (nie_true_t3i > 0.01 and nie_true_t3ii < 0.001
              and ci_t3i_L > 0.001)  # CI non-overlap with zero

    hc_ni_pass = crit_a and crit_b and crit_c

    print(f"\n  HC_NI crit_a (arm-match): {crit_a}  diff_E_S_T0={diff_E_S_T0:.4f}")
    print(f"  HC_NI crit_b (naive same): {crit_b}  diff_naive={diff_naive:.2e}")
    print(f"  HC_NI crit_c (NIE_true differ): {crit_c}  "
          f"T3i={nie_true_t3i:.6f}  T3ii={nie_true_t3ii:.6f}")
    print(f"  HC_NI overall: {'PASS' if hc_ni_pass else 'FAIL'}")

    return {
        "t3i_per_seed": t3i_results,
        "t3ii_per_seed": t3ii_results,
        "NIE_true_T3i_mean": nie_true_t3i,
        "NIE_true_T3i_std": float(np.nanstd(nie_t3i_vals, ddof=1)),
        "NIE_true_T3ii_mean": nie_true_t3ii,
        "NIE_naive_T3i_mean": nie_naive_t3i,
        "NIE_naive_T3ii_mean": nie_naive_t3ii,
        "diff_E_S_T0": diff_E_S_T0,
        "diff_E_S_T1": diff_E_S_T1,
        "diff_SD_T0": diff_SD_T0,
        "diff_SD_T1": diff_SD_T1,
        "diff_naive_NIE": diff_naive,
        "CI_T3i": [ci_t3i_L, ci_t3i_U],
        "crit_a_pass": crit_a,
        "crit_b_pass": crit_b,
        "crit_c_pass": crit_c,
        "HC_NI_supported": hc_ni_pass,
    }


# Section 3: HC3(G4) re-run

def run_hc3_g4_rerun() -> Dict:
    """Re-run HC3(G4) kappa recalibration."""
    print("\n" + "=" * 68)
    print("SECTION 3: HC3(G4) Re-run (kappa_grid_g4_recalibration logic)")
    print(f"  Seeds: {G4_SEEDS}  N/seed: {G4_N:,}")
    print("=" * 68)

    # Load previously calibrated alpha_b from pilot JSON
    pilot_g4 = json.load(open(PILOT_G4_JSON))
    alpha_b_new = pilot_g4["new_G4"]["alpha_b"]
    print(f"  Using frozen alpha_b from pilot: {alpha_b_new:.8f}")

    kappa_vals: List[float] = []
    per_seed_rows: List[Dict] = []
    for seed in G4_SEEDS:
        NIE, NDE, kappa = g4_compute_kappa(alpha_b_new, G4_N, seed)
        kappa_vals.append(kappa)
        per_seed_rows.append({
            "seed": seed,
            "alpha_b": alpha_b_new,
            "NIE": NIE,
            "NDE": NDE,
            "kappa": kappa,
        })
        print(f"  seed={seed}: kappa={kappa:.8f}")

    kappa_mean = float(np.mean(kappa_vals))
    kappa_std  = float(np.std(kappa_vals, ddof=1))
    hc3a_pass  = kappa_mean >= 0.999

    print(f"\n  HC3(G4): kappa_mean={kappa_mean:.8f} ± {kappa_std:.2e}")
    print(f"  HC3(G4) crit (kappa >= 0.999): {'PASS' if hc3a_pass else 'FAIL'}")

    return {
        "alpha_b_calibrated": alpha_b_new,
        "per_seed": per_seed_rows,
        "kappa_mean": kappa_mean,
        "kappa_std": kappa_std,
        "HC3a_kappa_gte_999": hc3a_pass,
        "HC3a_definition": "null_cell_kappa_true_ge_0999",
    }


# Section 4: Deterministic match verification

def verify_match_hc_t2(per_seed: List[Dict], tol: float = 1e-6) -> Dict:
    """Verify HC_T2 re-run matches pilot per_cell CSV."""
    print("\n  [Match check] HC_T2 vs pilot:")
    if not PILOT_HC_T2_CSV.exists():
        print(f"    WARNING: pilot CSV not found at {PILOT_HC_T2_CSV}")
        return {"status": "pilot_file_missing"}

    pilot_df = pd.read_csv(PILOT_HC_T2_CSV)
    rerun_df = pd.DataFrame(per_seed)

    # Compare key columns: tau_LATE_true, tau_IV_IPSW, D_L, D_U
    check_cols = ["tau_LATE_true", "tau_IV_IPSW", "D_L", "D_U"]
    mismatches = {}
    for col in check_cols:
        if col not in pilot_df.columns or col not in rerun_df.columns:
            mismatches[col] = "column_missing"
            continue
        max_diff = float(np.max(np.abs(pilot_df[col].values - rerun_df[col].values)))
        match = max_diff < tol
        mismatches[col] = {"max_diff": max_diff, "match": match}
        status = "OK" if match else "MISMATCH"
        print(f"    {col}: max_diff={max_diff:.2e}  [{status}]")

    all_match = all(
        v.get("match", False) for v in mismatches.values()
        if isinstance(v, dict)
    )
    print(f"    Overall match: {'PASS' if all_match else 'FAIL'}")
    return {"status": "PASS" if all_match else "FAIL", "details": mismatches}


def verify_match_hc_ni(hc_ni: Dict, tol: float = 1e-6) -> Dict:
    """Verify HC_NI re-run matches pilot JSON."""
    print("  [Match check] HC_NI vs pilot:")
    if not PILOT_HC_NI_JSON.exists():
        print(f"    WARNING: pilot JSON not found at {PILOT_HC_NI_JSON}")
        return {"status": "pilot_file_missing"}

    pilot_json = json.load(open(PILOT_HC_NI_JSON))
    pilot_hcni = pilot_json.get("HC_NI_results", {})

    checks = {
        "NIE_true_T3i_mean": (hc_ni["NIE_true_T3i_mean"],
                               pilot_hcni.get("T3i_NIE_true_mean", float("nan"))),
        "NIE_true_T3ii_mean": (hc_ni["NIE_true_T3ii_mean"],
                                pilot_hcni.get("T3ii_NIE_true_mean", float("nan"))),
    }
    results = {}
    for k, (rerun_val, pilot_val) in checks.items():
        diff = abs(rerun_val - pilot_val)
        match = diff < tol
        results[k] = {"rerun": rerun_val, "pilot": pilot_val, "diff": diff, "match": match}
        status = "OK" if match else "MISMATCH"
        print(f"    {k}: rerun={rerun_val:.6e}  pilot={pilot_val:.6e}  [{status}]")

    all_match = all(v.get("match", False) for v in results.values())
    print(f"    Overall match: {'PASS' if all_match else 'FAIL'}")
    return {"status": "PASS" if all_match else "FAIL", "details": results}


def verify_match_hc3_g4(hc3: Dict, tol: float = 1e-6) -> Dict:
    """Verify HC3(G4) re-run matches pilot JSON."""
    print("  [Match check] HC3(G4) vs pilot:")
    if not PILOT_G4_JSON.exists():
        print(f"    WARNING: pilot JSON not found at {PILOT_G4_JSON}")
        return {"status": "pilot_file_missing"}

    pilot_g4 = json.load(open(PILOT_G4_JSON))
    pilot_kappa = pilot_g4["new_G4"].get("kappa_mean", float("nan"))
    rerun_kappa = hc3["kappa_mean"]
    diff = abs(rerun_kappa - pilot_kappa)
    match = diff < tol
    print(f"    kappa_mean: rerun={rerun_kappa:.8f}  pilot={pilot_kappa:.8f}  diff={diff:.2e}  "
          f"[{'OK' if match else 'MISMATCH'}]")
    return {"status": "PASS" if match else "FAIL",
            "kappa_diff": diff, "rerun_kappa": rerun_kappa, "pilot_kappa": pilot_kappa}


# Section 5: Bonferroni-Holm K=5

def compute_p_values(per_seed: List[Dict], hc_ni: Dict, hc3: Dict) -> List[float]:
    """
    Compute p-values for H1-H5 (Bonferroni-Holm K=5 family).
    Returns list of 5 p-values, NaN where not available.

    H1: HC_NI  — one-sample t-test: NIE_true_T3i > 0 (vs H0: NIE_true_T3i <= 0)
    H2: HC_T2(b) — one-sided z-test: tau_IV_IPSW < 0 at T2-a (N=1M primary)
    H3: HC1    — NOT_YET_RUN → NaN
    H4: HC2    — NOT_YET_RUN → NaN
    H5: HC3(G4) — two-sided t-test: kappa_mean != 1.0 (H0: kappa = 1)
    """

    # H1: HC_NI — t-test on NIE_true_T3i across 5 seeds
    nie_t3i_vals = [r.get("NIE_true", float("nan")) for r in hc_ni["t3i_per_seed"]]
    t_stat, p_h1_twosided = stats.ttest_1samp(
        [v for v in nie_t3i_vals if not np.isnan(v)], 0.0
    )
    # One-tailed (H1: NIE_true > 0)
    p_h1 = float(p_h1_twosided / 2.0) if t_stat > 0 else 1.0

    # H2: HC_T2(b) — one-sided z-test tau < 0 at T2-a primary cell (N=1M per seed)
    t2a_rows = [r for r in per_seed if r.get("cell") == "T2-a"]
    if t2a_rows:
        # Use sandwich SE from each N=1M run; average tau and pool SE
        taus = [r["tau_IV_IPSW"] for r in t2a_rows]
        ses  = [r["SE_tau_IPSW"] for r in t2a_rows]
        tau_mean = float(np.mean(taus))
        # Combined SE (take mean SE divided by sqrt(n_seeds) for cross-seed mean)
        se_combined = float(np.mean(ses)) / np.sqrt(len(taus))
        z_h2 = tau_mean / se_combined
        p_h2 = float(stats.norm.cdf(z_h2))  # one-sided: P(Z < z) where z < 0 → small
    else:
        p_h2 = float("nan")

    # H3: computed in hc1_hc2_confirmatory.py (NaN placeholder here)
    p_h3 = float("nan")

    # H4: computed in hc1_hc2_confirmatory.py (NaN placeholder here)
    p_h4 = float("nan")

    # H5: HC3(G4) — two-sided t-test: kappa != 1.0
    kappa_vals = [r["kappa"] for r in hc3["per_seed"]]
    t_stat_h5, p_h5 = stats.ttest_1samp(kappa_vals, 1.0)
    p_h5 = float(p_h5)

    return [p_h1, p_h2, p_h3, p_h4, p_h5]


# Main

def main() -> None:
    t_start  = datetime.now()
    git_hash = get_git_commit()

    print("=" * 74)
    print("CONFIRMATORY RE-RUN POST-FREEZE")
    print("Provenance establishment: HC_T2(b)(c) + HC_NI + HC3(G4)")
    print(f"Date      : {t_start.isoformat()}")
    print(f"Python    : {sys.version.split()[0]}")
    print(f"NumPy     : {np.__version__}  SciPy: {getattr(__import__('scipy'), '__version__', '?')}")
    print(f"Host      : {socket.gethostname()}")
    print(f"git (pilot): {git_hash}")
    print(f"GPU       : none (CPU-only per prereg §4.3.3)")
    print("=" * 74)

    # ------------------------------------------------------------------
    # 1. HC_T2(b)(c)
    # ------------------------------------------------------------------
    per_seed_hct2, sp_hct2 = run_hc_t2_rerun()

    # ------------------------------------------------------------------
    # 2. HC_NI
    # ------------------------------------------------------------------
    hc_ni = run_hc_ni_rerun()

    # ------------------------------------------------------------------
    # 3. HC3(G4)
    # ------------------------------------------------------------------
    hc3 = run_hc3_g4_rerun()

    # ------------------------------------------------------------------
    # 4. Deterministic match verification
    # ------------------------------------------------------------------
    print("\n" + "=" * 68)
    print("SECTION 4: Deterministic Match Verification")
    print("=" * 68)
    match_hct2 = verify_match_hc_t2(per_seed_hct2)
    match_ni   = verify_match_hc_ni(hc_ni)
    match_g4   = verify_match_hc3_g4(hc3)
    all_match  = (match_hct2["status"] == "PASS" and
                  match_ni["status"]   == "PASS" and
                  match_g4["status"]   == "PASS")
    print(f"\n  DETERMINISTIC MATCH OVERALL: {'PASS' if all_match else 'FAIL'}")

    # ------------------------------------------------------------------
    # 5. Bonferroni-Holm K=5
    # ------------------------------------------------------------------
    print("\n" + "=" * 68)
    print("SECTION 5: Bonferroni-Holm K=5 (prereg §6.2 M5)")
    print("=" * 68)
    p_values = compute_p_values(per_seed_hct2, hc_ni, hc3)
    hypoth_names = [
        "H1:HC_NI (NIE_true_T3i>0)",
        "H2:HC_T2(b) (tau_IV_IPSW<0, T2-a)",
        "H3:HC1 (NIE recovery, Tier1) [NOT_YET_RUN]",
        "H4:HC2 (collider correction) [NOT_YET_RUN]",
        "H5:HC3(G4) (kappa=1 at G4 null)",
    ]
    print(f"\n  Raw p-values:")
    for name, p in zip(hypoth_names, p_values):
        ps = f"{p:.4e}" if not np.isnan(p) else "NaN (not yet run)"
        print(f"    {name}: p={ps}")

    bh_result = bonferroni_holm(p_values)
    print(f"\n  Bonferroni-Holm correction (alpha={bh_result['alpha']}):")
    print(f"    K_available = {bh_result['K_available']} / {bh_result['K_total_prereg']} total")
    for i, (name, p) in enumerate(zip(hypoth_names, p_values)):
        adj = bh_result["adjusted_p_values"].get(i, float("nan"))
        rej = bh_result["reject_H0"].get(i, None)
        if not np.isnan(p):
            print(f"    {name}: adj_p={adj:.4e}  reject_H0={rej}")
        else:
            print(f"    {name}: PENDING (not yet run)")

    # ------------------------------------------------------------------
    # 6. Save results
    # ------------------------------------------------------------------
    elapsed = (datetime.now() - t_start).total_seconds()

    # CSV: per-seed HC_T2 + size/power + HC_NI summary + HC3 per seed
    rows_csv: List[Dict] = []
    for r in per_seed_hct2:
        r2 = dict(r)
        r2["result_type"] = "HC_T2_calibration"
        rows_csv.append(r2)
    for sp in sp_hct2:
        sp2 = dict(sp)
        sp2["result_type"] = "HC_T2_size_power"
        rows_csv.append(sp2)
    for i, (r_i, r_ii) in enumerate(
            zip(hc_ni["t3i_per_seed"], hc_ni["t3ii_per_seed"])):
        ni_row_i = dict(r_i)
        ni_row_i["result_type"] = "HC_NI_T3i"
        ni_row_i["seed"] = NI_SEEDS[i]
        rows_csv.append(ni_row_i)
        ni_row_ii = dict(r_ii)
        ni_row_ii["result_type"] = "HC_NI_T3ii"
        ni_row_ii["seed"] = NI_SEEDS[i]
        rows_csv.append(ni_row_ii)
    for r in hc3["per_seed"]:
        r2 = dict(r)
        r2["result_type"] = "HC3_G4"
        rows_csv.append(r2)

    df_out = pd.DataFrame(rows_csv)
    df_out.to_csv(OUT_CSV, index=False)
    print(f"\nCSV saved: {OUT_CSV}")

    # JSON aggregate
    env_log = {
        "script": "confirmatory_rerun_postfreeze.py",
        "date": t_start.isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": sys.version,
        "git_commit": git_hash,
        "elapsed_seconds": elapsed,
        "purpose": "confirmatory re-run post-freeze for provenance establishment",
        "notes": (
            "HC1/HC2 not yet run (Tier 1 experiments pending). "
            "BH correction applied to K_available=3 out of K=5 prereg hypotheses."
        ),
    }

    output_json = {
        "env_log": env_log,
        "preregistration": "preregistration_FROZEN_2026-07-11.md §5.1 §6.2",
        "HC_T2_summary": {
            "cells_run": [s["cell"] for s in TIER2_CELLS],
            "seeds": HC_T2_SEEDS,
            "N_per_seed": HC_T2_N,
            "M_reps": HC_T2_M,
            "N_rep": HC_T2_NREP,
            "size_power": {sp["cell"]: {
                "role": "SIZE" if "null" in sp["cell"] else "POWER",
                "reject_rate_IPSW": sp["reject_rate_IPSW"],
                "SE_reject_IPSW": sp["SE_reject_IPSW"],
                "tau_IV_IPSW_mean": sp["tau_IV_IPSW_mean"],
            } for sp in sp_hct2},
        },
        "HC_NI_summary": {
            "crit_a_pass": hc_ni["crit_a_pass"],
            "crit_b_pass": hc_ni["crit_b_pass"],
            "crit_c_pass": hc_ni["crit_c_pass"],
            "HC_NI_supported": hc_ni["HC_NI_supported"],
            "NIE_true_T3i_mean": hc_ni["NIE_true_T3i_mean"],
            "NIE_true_T3ii_mean": hc_ni["NIE_true_T3ii_mean"],
            "CI_T3i": hc_ni["CI_T3i"],
        },
        "HC3_G4_summary": {
            "alpha_b_calibrated": hc3["alpha_b_calibrated"],
            "kappa_mean": hc3["kappa_mean"],
            "kappa_std": hc3["kappa_std"],
            "HC3a_kappa_gte_999": hc3["HC3a_kappa_gte_999"],
        },
        "deterministic_match": {
            "HC_T2": match_hct2,
            "HC_NI": match_ni,
            "HC3_G4": match_g4,
            "overall": "PASS" if all_match else "FAIL",
        },
        "bonferroni_holm": {
            "raw_p_values": {
                name: (float(p) if not np.isnan(p) else None)
                for name, p in zip(hypoth_names, p_values)
            },
            "result": bh_result,
            "note": "H3(HC1) and H4(HC2) are pending (Tier 1 experiments not yet run). "
                    "BH applied to H1+H2+H5 (K_available=3).",
        },
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output_json, f, indent=2, ensure_ascii=False, default=str)
    print(f"JSON saved: {OUT_JSON}")

    print(f"\n{'=' * 74}")
    print(f"CONFIRMATORY RE-RUN COMPLETE  ({elapsed:.1f}s)")
    print(f"  HC_T2 match: {match_hct2['status']}")
    print(f"  HC_NI match: {match_ni['status']}")
    print(f"  HC3(G4) match: {match_g4['status']}")
    print(f"  HC_NI: {'PASS' if hc_ni['HC_NI_supported'] else 'FAIL'}")
    print(f"  HC3(G4): {'PASS' if hc3['HC3a_kappa_gte_999'] else 'FAIL'}")
    print(f"  BH H1 adj_p: {bh_result['adjusted_p_values'].get(0, float('nan')):.4e}")
    print(f"  BH H2 adj_p: {bh_result['adjusted_p_values'].get(1, float('nan')):.4e}")
    print(f"  Output CSV : {OUT_CSV}")
    print(f"  Output JSON: {OUT_JSON}")
    print("=" * 74)


if __name__ == "__main__":
    main()
