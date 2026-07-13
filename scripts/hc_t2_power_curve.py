"""
HC_T2(b)(iii) supplementary power curve across N_rep ∈ {10k, 20k, 50k, 100k}.
Demonstrates power reachability (not a pass/fail criterion) for Tier-2 cells.
For T2-c/d where N_rep=100k does not reach 0.80, fits probit(power) = a + b*log(N_rep) to extrapolate.
Outputs: results/hc_t2_power_curve.csv, results/hc_t2_power_curve_aggregate.json
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
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy
import scipy.stats
import scipy.optimize

# Paths
BASE_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CSV_OUT  = RESULTS_DIR / "hc_t2_power_curve.csv"
JSON_OUT = RESULTS_DIR / "hc_t2_power_curve_aggregate.json"

# Frozen SCM parameters (identical to hc_t2_reduction.py)
LOGIT_P0: float   = float(np.log(0.55 / 0.45))   # logit(0.55) ≈ 0.20067
GAMMA_B: float    = 0.50
STD_EPS_B: float  = 0.30
STD_EPS_V: float  = 5.00
BETA_U: float     = 0.10
STD_EPS_Y: float  = 0.05
ESC_REDUCTION: float = 0.30
SPEED_MULT: float = 5.0
ETA_0: float      = -2.0
ETA_T: float      = -0.5
ETA_U: float      =  0.3
ETA_B: float      =  0.2
IV_STR: float     = 1.5

G4_ALPHA_B: float = 4.99720863   # null cell alpha_b (kappa ≈ 1, tau ≈ 0)

# Power curve sweep
N_REP_LIST: List[int] = [10_000, 20_000, 50_000, 100_000]
M_REPS_PC: int        = 1000    # repetitions per (cell, N_rep) pair
REP_BASE_SEED_PC: int = 11111   # distinct from original 77777

ALPHA_LEVEL: float = 0.05

# Tier-2 cells + null size cell
TIER2_CELLS: List[Dict] = [
    {"cell": "T2-a",    "rho": 0.5, "alpha_b": 0.3},
    {"cell": "T2-b",    "rho": 1.0, "alpha_b": 0.3},
    {"cell": "T2-c",    "rho": 0.5, "alpha_b": 3.972},
    {"cell": "T2-d",    "rho": 1.0, "alpha_b": 3.972},
    {"cell": "T2-null", "rho": 0.5, "alpha_b": G4_ALPHA_B},
]

# tau_LATE_true reference values from hc_t2_reduction (git 8da43de / N=10^6 x 5 seeds)
TAU_TRUE_REF: Dict[str, float] = {
    "T2-a":    -0.063038,   # mean ± 1.27e-04
    "T2-b":    -0.058665,   # mean ± 1.25e-04
    "T2-c":    -0.018299,   # mean ± 7.30e-05
    "T2-d":    -0.015322,   # mean ± 6.96e-05
    "T2-null": +0.001848,   # kappa ≈ 1, size cell
}


# Core helpers (copied from hc_t2_reduction.py for standalone execution)

def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )


def _ipsw_wald_reject(
    Y_c1: np.ndarray,
    T_c1: np.ndarray,
    Z_c1: np.ndarray,
    W_c1: np.ndarray,
) -> Tuple[bool, float, float]:
    """
    IPSW-weighted IV Wald test of H0: tau_LATE = 0.

    Returns
    -------
    (reject, tau_iv, p_value)
    reject : bool    -- True if two-sided p-value < ALPHA_LEVEL
    tau_iv : float   -- point estimate
    p_value: float
    """
    m1 = (Z_c1 == 1)
    m0 = (Z_c1 == 0)

    w1, w0 = W_c1[m1], W_c1[m0]
    sw1, sw0 = w1.sum(), w0.sum()

    if sw1 < 1e-8 or sw0 < 1e-8:
        return (False, float("nan"), float("nan"))

    EY_Z1 = float(np.dot(Y_c1[m1], w1) / sw1)
    EY_Z0 = float(np.dot(Y_c1[m0], w0) / sw0)
    ET_Z1 = float(np.dot(T_c1[m1], w1) / sw1)
    ET_Z0 = float(np.dot(T_c1[m0], w0) / sw0)

    RF = EY_Z1 - EY_Z0
    FS = ET_Z1 - ET_Z0

    if abs(FS) < 1e-8:
        return (False, float("nan"), float("nan"))

    tau = RF / FS

    # Hajek sandwich variance
    resY_Z1 = Y_c1[m1] - EY_Z1
    resY_Z0 = Y_c1[m0] - EY_Z0
    resT_Z1 = T_c1[m1] - ET_Z1
    resT_Z0 = T_c1[m0] - ET_Z0

    var_RF = (np.dot(w1 ** 2, resY_Z1 ** 2) / sw1 ** 2
            + np.dot(w0 ** 2, resY_Z0 ** 2) / sw0 ** 2)
    var_FS = (np.dot(w1 ** 2, resT_Z1 ** 2) / sw1 ** 2
            + np.dot(w0 ** 2, resT_Z0 ** 2) / sw0 ** 2)

    var_tau = max(1e-20, var_RF / FS ** 2 + RF ** 2 * var_FS / FS ** 4)
    SE_tau = float(np.sqrt(var_tau))

    z_stat = tau / SE_tau
    p_val  = float(2.0 * scipy.stats.norm.sf(abs(z_stat)))

    return (bool(p_val < ALPHA_LEVEL), float(tau), p_val)


# One simulation repetition for power curve

def run_one_rep_pc(
    rho: float,
    alpha_b: float,
    seed: int,
    N_rep: int,
) -> Tuple[bool, float]:
    """
    One repetition of H0: tau=0 test for power curve analysis.

    Parameters
    ----------
    rho      : confounding strength for P(T|U,Z)
    alpha_b  : compensation coefficient T --> b
    seed     : RNG seed for this repetition
    N_rep    : total generated observations

    Returns
    -------
    (reject_ipsw, tau_ipsw)
    reject_ipsw : bool   -- whether IPSW-Wald rejects H0
    tau_ipsw    : float  -- point estimate (nan if degenerate)
    """
    rng = np.random.default_rng(seed)

    U     = rng.standard_normal(N_rep)
    eps_b = rng.normal(0.0, STD_EPS_B, N_rep)
    eps_v = rng.normal(0.0, STD_EPS_V, N_rep)
    eps_y = rng.normal(0.0, STD_EPS_Y, N_rep)

    v_base = 50.0 + 10.0 * U + eps_v

    Z   = rng.binomial(1, 0.5, N_rep).astype(float)
    P_T = sigmoid(LOGIT_P0 + rho * U + IV_STR * (Z - 0.5))
    T   = rng.binomial(1, P_T).astype(float)

    b        = alpha_b * T + GAMMA_B * U + eps_b
    v_actual = v_base + SPEED_MULT * b
    dv       = v_actual * (1.0 - ESC_REDUCTION * T)
    Y        = (dv / 100.0) ** 4 + BETA_U * U + eps_y

    P_crash = sigmoid(ETA_0 + ETA_T * T + ETA_U * U + ETA_B * b)
    C       = rng.binomial(1, P_crash).astype(float)
    W_ipsw  = 1.0 / P_crash

    mask_C1 = (C == 1)
    n_C1    = int(mask_C1.sum())

    if n_C1 < 20:   # guard degenerate samples
        return (False, float("nan"))

    reject, tau_iv, _p = _ipsw_wald_reject(
        Y[mask_C1], T[mask_C1], Z[mask_C1], W_ipsw[mask_C1]
    )
    return (reject, tau_iv)


# Power curve runner for one (cell, N_rep) pair

def run_power_curve_one(
    cell: str,
    rho: float,
    alpha_b: float,
    N_rep: int,
    m_reps: int,
    rep_seeds: np.ndarray,
) -> Dict:
    """
    Run m_reps repetitions of H0 test for one (cell, N_rep) configuration.

    Parameters
    ----------
    cell      : cell label (e.g., "T2-a")
    rho       : confounding strength
    alpha_b   : compensation coefficient
    N_rep     : sample size per repetition
    m_reps    : number of Monte Carlo repetitions
    rep_seeds : 1D array of seeds of length >= m_reps

    Returns
    -------
    dict with reject_rate_IPSW, SE_reject_IPSW, n_valid, 95%_CI_L, 95%_CI_U
    """
    rejects: List[bool] = []
    taus:    List[float] = []

    t0 = time.perf_counter()
    for m in range(m_reps):
        rej, tau = run_one_rep_pc(rho, alpha_b, int(rep_seeds[m]), N_rep)
        rejects.append(rej)
        if not np.isnan(tau):
            taus.append(tau)
    elapsed = time.perf_counter() - t0

    valid_rej  = [r for r in rejects if isinstance(r, bool)]
    n_valid    = len(valid_rej)
    rej_rate   = float(np.mean(valid_rej)) if valid_rej else float("nan")

    se_rej = (float(np.sqrt(rej_rate * (1 - rej_rate) / n_valid))
              if n_valid > 0 and not np.isnan(rej_rate) else float("nan"))

    # 95% normal-approximation CI on rejection rate
    ci_l = float(max(0.0, rej_rate - 1.96 * se_rej)) if not np.isnan(se_rej) else float("nan")
    ci_u = float(min(1.0, rej_rate + 1.96 * se_rej)) if not np.isnan(se_rej) else float("nan")

    tau_mean = float(np.mean(taus))  if taus else float("nan")
    tau_std  = float(np.std(taus, ddof=1)) if len(taus) > 1 else float("nan")

    role = "SIZE" if "null" in cell else "POWER"

    return {
        "cell"             : cell,
        "rho"              : rho,
        "alpha_b"          : alpha_b,
        "N_rep"            : N_rep,
        "M_reps"           : m_reps,
        "role"             : role,
        "n_valid"          : n_valid,
        "reject_rate_IPSW" : rej_rate,
        "SE_reject_IPSW"   : se_rej,
        "CI_95_L"          : ci_l,
        "CI_95_U"          : ci_u,
        "tau_IV_IPSW_mean" : tau_mean,
        "tau_IV_IPSW_std"  : tau_std,
        "elapsed_sec"      : round(elapsed, 2),
    }


# Power curve extrapolation (probit model: inv_Phi(power) = a + b*log(N))

def fit_power_curve_and_extrapolate(
    N_rep_list: List[int],
    powers: List[float],
    target_power: float = 0.80,
) -> Dict:
    """
    Fit probit(power) = a + b * log10(N_rep) and solve for N_rep at target_power.

    Parameters
    ----------
    N_rep_list    : list of N_rep values used
    powers        : list of observed rejection rates (same order as N_rep_list)
    target_power  : target power (default 0.80)

    Returns
    -------
    dict with fit parameters, R^2, N_rep_extrapolated_for_target_power
    """
    valid = [(n, p) for n, p in zip(N_rep_list, powers)
             if not np.isnan(p) and 0.0 < p < 1.0]
    if len(valid) < 2:
        return {
            "fit_valid"          : False,
            "reason"             : "insufficient valid power values (need >=2 with 0<p<1)",
            "N_rep_for_power_80" : None,
            "note"               : "extrapolation not possible",
        }

    x_vals = np.array([np.log10(n) for n, _ in valid])
    y_vals = np.array([scipy.stats.norm.ppf(p) for _, p in valid])  # probit transform

    # Fit: probit(power) = a + b * log10(N_rep)
    try:
        popt, pcov = scipy.optimize.curve_fit(
            lambda x, a, b: a + b * x,
            x_vals, y_vals,
            p0=[-4.0, 1.0],
        )
    except Exception as e:
        return {
            "fit_valid" : False,
            "reason"    : str(e),
            "N_rep_for_power_80": None,
            "note"      : "curve_fit failed",
        }

    a, b = float(popt[0]), float(popt[1])
    y_pred = a + b * x_vals
    ss_res = float(np.sum((y_vals - y_pred) ** 2))
    ss_tot = float(np.sum((y_vals - y_vals.mean()) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else float("nan")

    # Solve for N_rep when probit(power) = probit(target_power)
    target_probit = float(scipy.stats.norm.ppf(target_power))   # = 0.842 for 0.80
    # a + b * log10(N) = target_probit  -->  N = 10^((target_probit - a) / b)
    if abs(b) < 1e-8:
        N_req = float("inf")
        note = "slope ≈ 0; power does not increase with N"
    elif (target_probit - a) / b < 0:
        N_req = float("nan")
        note = "slope direction incompatible with target power"
    else:
        log10_N_req = (target_probit - a) / b
        N_req = float(10 ** log10_N_req)
        note = f"probit model: probit(power) = {a:.3f} + {b:.3f} * log10(N)"

    # Also report power at specific N_rep values via model
    power_at_nrep = {
        str(n): float(scipy.stats.norm.cdf(a + b * np.log10(n)))
        for n in [10_000, 20_000, 50_000, 100_000, 200_000, 500_000]
    }

    return {
        "fit_valid"               : True,
        "a_intercept"             : a,
        "b_slope_log10N"          : b,
        "R2"                      : r2,
        "N_fitted_points"         : len(valid),
        "fitted_N_rep_list"       : [int(n) for n, _ in valid],
        "fitted_power_list"       : [float(p) for _, p in valid],
        "target_power"            : target_power,
        "N_rep_for_power_80"      : float(N_req),
        "N_rep_for_power_80_label": (
            f"≈{round(N_req / 1000)}k" if not np.isinf(N_req) and not np.isnan(N_req)
            else "inf/nan"
        ),
        "power_at_nrep_model"     : power_at_nrep,
        "note"                    : note,
    }


# Git / env helpers

def get_git_commit() -> str:
    try:
        r = subprocess.run(
            ["git", "-C", str(BASE_DIR), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        return r.stdout.strip() if r.returncode == 0 else "unavailable"
    except Exception:
        return "unavailable"


def make_env_log(git_hash: str, t_start: datetime) -> Dict:
    return {
        "experiment_name"  : "hc_t2_power_curve",
        "description"      : (
            "Supplementary power-curve experiment for HC_T2(b)(iii). "
            "Sweeps N_rep to show T2-a/b reach power>=0.80 at realistic sample sizes. "
            "Ref: DRAFT §5.1 HC_T2(b)(iii). git 8da43de / 8117390 = prior reduction run."
        ),
        "date"             : t_start.isoformat(),
        "python_version"   : sys.version,
        "numpy_version"    : np.__version__,
        "scipy_version"    : scipy.__version__,
        "pandas_version"   : pd.__version__,
        "os"               : platform.platform(),
        "hostname"         : socket.gethostname(),
        "gpu"              : "none (CPU-only per DRAFT §4.3.3 rule 5)",
        "git_commit"       : git_hash,
        "N_rep_list"       : N_REP_LIST,
        "M_reps_per_cell"  : M_REPS_PC,
        "alpha_level"      : ALPHA_LEVEL,
        "REP_BASE_SEED_PC" : REP_BASE_SEED_PC,
        "IV_STR"           : IV_STR,
        "ESC_REDUCTION"    : ESC_REDUCTION,
        "prior_run_N_rep10k_power": {
            "T2-a":    0.5555,
            "T2-b":    0.4405,
            "T2-c":    0.0925,
            "T2-d":    0.0635,
            "T2-null": 0.0500,
        },
        "prior_tau_LATE_true": TAU_TRUE_REF,
        "note_M_reps": (
            "M=1000 (vs. original 2000) is sufficient for power curve purposes. "
            "SE of rejection rate at p=0.80: sqrt(0.16/1000) ≈ 0.013. "
            "This is a reporting instrument, not a confirmatory criterion."
        ),
    }


# Main

def main() -> None:
    t_start  = datetime.now()
    git_hash = get_git_commit()

    print("=" * 74)
    print("HC_T2(b)(iii) Power Curve Supplementary Experiment")
    print("preregistration_DRAFT_2026-07-10.md §5.1 HC_T2(b)(iii)")
    print(f"Date      : {t_start.isoformat()}")
    print(f"Python    : {sys.version.split()[0]}  NumPy: {np.__version__}  "
          f"SciPy: {scipy.__version__}")
    print(f"Host      : {socket.gethostname()}  OS: {platform.system()}")
    print(f"git       : {git_hash}")
    print(f"GPU       : none (CPU-only)")
    print(f"N_rep sweep: {N_REP_LIST}")
    print(f"M_reps    : {M_REPS_PC} per (cell, N_rep)")
    print(f"Cells     : {[c['cell'] for c in TIER2_CELLS]}")
    print("=" * 74)

    # Pre-generate all seeds from the dedicated base seed
    rng_meta = np.random.default_rng(REP_BASE_SEED_PC)
    all_seeds = rng_meta.integers(0, 2**31 - 1, M_REPS_PC)

    # -----------------------------------------------------------------------
    # Run power curve
    # -----------------------------------------------------------------------
    all_rows: List[Dict] = []

    for spec in TIER2_CELLS:
        cell, rho, alpha_b = spec["cell"], spec["rho"], spec["alpha_b"]
        role = "SIZE" if "null" in cell else "POWER"
        tau_ref = TAU_TRUE_REF.get(cell, float("nan"))

        print(f"\n{'='*74}")
        print(f"Cell: {cell}  rho={rho}  alpha_b={alpha_b}  "
              f"role={role}  tau_true={tau_ref:+.6f}")
        print(f"{'='*74}")

        cell_rows: List[Dict] = []

        for N_rep in N_REP_LIST:
            print(f"  N_rep={N_rep:>7,}  M={M_REPS_PC}  ...", end="", flush=True)
            row = run_power_curve_one(
                cell, rho, alpha_b, N_rep, M_REPS_PC, all_seeds
            )
            all_rows.append(row)
            cell_rows.append(row)

            p    = row["reject_rate_IPSW"]
            se   = row["SE_reject_IPSW"]
            mark = ">=0.80 *" if (not np.isnan(p) and p >= 0.80) else ""
            print(
                f"  reject_IPSW = {p:.4f} ± {se:.4f}  "
                f"95%CI=[{row['CI_95_L']:.4f},{row['CI_95_U']:.4f}]  "
                f"{row['elapsed_sec']:.1f}s  {mark}"
            )

        # Extrapolation
        powers_cell = [r["reject_rate_IPSW"] for r in cell_rows]
        extrap = fit_power_curve_and_extrapolate(N_REP_LIST, powers_cell, 0.80)
        print(f"\n  [Extrapolation fit for {cell}]")
        if extrap["fit_valid"]:
            print(f"    probit(power) = {extrap['a_intercept']:.4f} + "
                  f"{extrap['b_slope_log10N']:.4f} * log10(N_rep)  "
                  f"R^2={extrap['R2']:.4f}")
            print(f"    Estimated N_rep for power=0.80: {extrap['N_rep_for_power_80_label']} "
                  f"(= {extrap['N_rep_for_power_80']:.0f})")
            for nk, pw in extrap["power_at_nrep_model"].items():
                print(f"      model power @ N_rep={nk}: {pw:.4f}")
        else:
            print(f"    Extrapolation not valid: {extrap.get('reason', '')}")

    # -----------------------------------------------------------------------
    # Summary table
    # -----------------------------------------------------------------------
    print("\n" + "=" * 74)
    print("POWER CURVE SUMMARY TABLE")
    print(f"{'Cell':<8} {'N_rep':>7} {'role':>5} {'power':>7} {'±SE':>7} "
          f"{'CI_L':>7} {'CI_U':>7} {'>=0.80':>7}")
    print("-" * 74)
    for row in all_rows:
        p  = row["reject_rate_IPSW"]
        se = row["SE_reject_IPSW"]
        cl = row["CI_95_L"]
        cu = row["CI_95_U"]
        mark = "YES" if (not np.isnan(p) and p >= 0.80) else "---"
        print(
            f"{row['cell']:<8} {row['N_rep']:>7,} {row['role']:>5} "
            f"{p:>7.4f} {se:>7.4f} {cl:>7.4f} {cu:>7.4f} {mark:>7}"
        )

    # -----------------------------------------------------------------------
    # Dedicated analysis: T2-a and T2-b threshold detection
    # -----------------------------------------------------------------------
    print("\n" + "=" * 74)
    print("T2-a / T2-b : FIRST N_rep WHERE power >= 0.80")
    print("=" * 74)
    for cell_name in ["T2-a", "T2-b"]:
        rows_c = [r for r in all_rows if r["cell"] == cell_name]
        reached = [(r["N_rep"], r["reject_rate_IPSW"]) for r in rows_c
                   if not np.isnan(r["reject_rate_IPSW"]) and r["reject_rate_IPSW"] >= 0.80]
        if reached:
            n_first, p_first = reached[0]
            print(f"  {cell_name}: power >= 0.80 FIRST REACHED at N_rep = {n_first:,} "
                  f"(power = {p_first:.4f})")
        else:
            rows_c_sorted = sorted(rows_c, key=lambda r: r["N_rep"])
            p_max = max((r["reject_rate_IPSW"] for r in rows_c_sorted
                         if not np.isnan(r["reject_rate_IPSW"])), default=float("nan"))
            print(f"  {cell_name}: power < 0.80 in all N_rep tested (max = {p_max:.4f}). "
                  f"Check extrapolation.")

    print("\nT2-null (SIZE CONTROL CHECK):")
    null_rows = [r for r in all_rows if r["cell"] == "T2-null"]
    for row in null_rows:
        p = row["reject_rate_IPSW"]
        size_ok = "OK (<=0.10)" if (not np.isnan(p) and p <= 2 * ALPHA_LEVEL) else "FAIL"
        print(f"  N_rep={row['N_rep']:>7,}: size = {p:.4f} ± {row['SE_reject_IPSW']:.4f}  {size_ok}")

    # -----------------------------------------------------------------------
    # Save CSV
    # -----------------------------------------------------------------------
    df = pd.DataFrame(all_rows)
    df.to_csv(CSV_OUT, index=False)
    print(f"\nCSV saved: {CSV_OUT}")

    # -----------------------------------------------------------------------
    # Build aggregate JSON with extrapolation per cell
    # -----------------------------------------------------------------------
    # Re-run extrapolation per cell for JSON
    per_cell_summary: Dict[str, Dict] = {}
    for spec in TIER2_CELLS:
        cell = spec["cell"]
        rows_c = [r for r in all_rows if r["cell"] == cell]
        rows_c_sorted = sorted(rows_c, key=lambda r: r["N_rep"])
        powers_c = [r["reject_rate_IPSW"] for r in rows_c_sorted]
        n_list_c = [r["N_rep"] for r in rows_c_sorted]
        extrap_c = fit_power_curve_and_extrapolate(n_list_c, powers_c, 0.80)

        # Find first N_rep with power >= 0.80 (empirical)
        empirical_80 = next(
            (r["N_rep"] for r in rows_c_sorted
             if not np.isnan(r["reject_rate_IPSW"]) and r["reject_rate_IPSW"] >= 0.80),
            None,
        )

        per_cell_summary[cell] = {
            "tau_LATE_true_ref"     : TAU_TRUE_REF.get(cell, float("nan")),
            "role"                  : "SIZE" if "null" in cell else "POWER",
            "power_by_N_rep"        : {str(r["N_rep"]): {
                "reject_rate_IPSW": r["reject_rate_IPSW"],
                "SE_reject_IPSW"  : r["SE_reject_IPSW"],
                "CI_95_L"         : r["CI_95_L"],
                "CI_95_U"         : r["CI_95_U"],
            } for r in rows_c_sorted},
            "first_N_rep_power_80_empirical": empirical_80,
            "extrapolation"         : extrap_c,
        }

    # Final verdicts
    t2a_80 = per_cell_summary["T2-a"]["first_N_rep_power_80_empirical"]
    t2b_80 = per_cell_summary["T2-b"]["first_N_rep_power_80_empirical"]
    t2a_80_ext = per_cell_summary["T2-a"]["extrapolation"].get("N_rep_for_power_80")
    t2b_80_ext = per_cell_summary["T2-b"]["extrapolation"].get("N_rep_for_power_80")

    draft_update_text = (
        "Section 5.1 HC_T2(b)(iii) power curve update:\n"
        f"T2-a (|tau|≈0.063): power(N_rep) by {M_REPS_PC} reps IPSW-Wald\n"
    )
    for spec in TIER2_CELLS:
        c = spec["cell"]
        if "null" in c:
            continue
        rows_c = sorted([r for r in all_rows if r["cell"] == c], key=lambda r: r["N_rep"])
        line = f"  {c}: " + "  ".join(
            f"N={r['N_rep']//1000}k → {r['reject_rate_IPSW']:.3f}" for r in rows_c
        )
        draft_update_text += line + "\n"

    aggregate = {
        "computation_type"  : "hc_t2_power_curve",
        "description"       : (
            "Supplementary power curve for HC_T2(b)(iii). "
            "H0:ATE_LATE=0 IPSW-Wald rejection rate as function of N_rep. "
            "M=1000 reps per (cell, N_rep). All 5 cells × 4 N_rep values."
        ),
        "preregistration"   : str(BASE_DIR / "preregistration_DRAFT_2026-07-10.md"),
        "prior_reduction_run": str(RESULTS_DIR / "hc_t2_reduction_aggregate.json"),
        "env_log"           : make_env_log(git_hash, t_start),
        "per_cell_summary"  : per_cell_summary,
        "verdict_T2_a_b_reachability": {
            "T2-a_first_N_rep_power_80_empirical" : t2a_80,
            "T2-b_first_N_rep_power_80_empirical" : t2b_80,
            "T2-a_N_rep_power_80_extrapolated"    : t2a_80_ext,
            "T2-b_N_rep_power_80_extrapolated"    : t2b_80_ext,
            "interpretation": (
                "T2-a/b have |tau|≈0.06 (meaningful effect). "
                "Power scales with sqrt(N_rep) via IPSW Wald. "
                "At N_rep sufficient for 0.80, the test is feasible at realistic sample sizes. "
                "T2-c/d have |tau|≈0.015-0.018 (small effect) and require much larger N_rep."
            ),
        },
        "draft_update_text" : draft_update_text,
        "size_stability"    : {
            str(r["N_rep"]): {
                "reject_rate_IPSW": r["reject_rate_IPSW"],
                "within_2alpha"   : bool(
                    not np.isnan(r["reject_rate_IPSW"])
                    and r["reject_rate_IPSW"] <= 2 * ALPHA_LEVEL
                ),
            }
            for r in all_rows if r["cell"] == "T2-null"
        },
        "all_rows"          : all_rows,
    }

    with open(JSON_OUT, "w", encoding="utf-8") as f:
        json.dump(aggregate, f, ensure_ascii=False, indent=2)
    print(f"JSON saved: {JSON_OUT}")

    t_end = datetime.now()
    elapsed_total = (t_end - t_start).total_seconds()
    print(f"\nTotal elapsed: {elapsed_total:.1f} seconds ({elapsed_total/60:.1f} minutes)")
    print(f"git commit at run time: {git_hash}")
    print("\nDone. Verify file existence before reporting.")
    print(f"  {CSV_OUT}")
    print(f"  {JSON_OUT}")


if __name__ == "__main__":
    main()
