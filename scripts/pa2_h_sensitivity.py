"""
PA2: H(c) quality sensitivity analysis for IPSW-corrected IV Wald ATE_LATE estimator (exploratory).
Three degradation series on oracle crash probability H(c)=P(C=1|X):
  noise (log-normal perturbation), spatial (stratum-level aggregation),
  temporal (drop eta_B*b from H model, simulating within-year behavioral variation).
Outputs: results/pa2_h_sensitivity.csv, results/pa2_h_sensitivity_aggregate.json
"""

from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy
import scipy.optimize
import scipy.stats

# Paths
BASE_DIR   = Path(__file__).resolve().parents[1]
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

FROZEN_PREREG = BASE_DIR / "preregistration_FROZEN_2026-07-11.md"

# Frozen SCM parameters (identical to hc_t2_reduction.py)
LOGIT_P0   : float = float(np.log(0.55 / 0.45))   # logit(0.55) ≈ 0.2007
GAMMA_B    : float = 0.50
STD_EPS_B  : float = 0.30
STD_EPS_V  : float = 5.00
BETA_U     : float = 0.10
STD_EPS_Y  : float = 0.05
ESC_REDUC  : float = 0.30    # illustrative modeling choice
SPEED_MULT : float = 5.0
ETA_0      : float = -2.0
ETA_T      : float = -0.5
ETA_U      : float =  0.3
ETA_B      : float =  0.2
IV_STR     : float = 1.5     # logit-scale IV shift

G4_ALPHA_B : float = 4.99720863   # null-cell alpha_b (kappa ≈ 1.0)

SEEDS      : List[int] = [42, 123, 456, 789, 2026]
N_FINAL    : int = 1_000_000
M_REPS     : int = 2000
N_REP      : int = 10_000
REP_BASE_SEED : int = 88888          # different from hc_t2_reduction.py (77777)
ALPHA_LEVEL : float = 0.05

# Tier-2 cells (same ordering as hc_t2_reduction.py)
TIER2_CELLS : List[Dict] = [
    {"cell": "T2-a",    "rho": 0.5, "alpha_b": 0.3},
    {"cell": "T2-b",    "rho": 1.0, "alpha_b": 0.3},
    {"cell": "T2-c",    "rho": 0.5, "alpha_b": 3.972},
    {"cell": "T2-d",    "rho": 1.0, "alpha_b": 3.972},
    {"cell": "T2-null", "rho": 0.5, "alpha_b": G4_ALPHA_B},
]

# PA2 degradation conditions
# Series 1: multiplicative log-normal noise on oracle H
NOISE_SIGMAS : List[float] = [0.0, 0.1, 0.2, 0.4]

# Series 2: spatial coarsening – K_STATE equal-probability bins of U × 2 T values
K_STATE_BINS : int = 10
# Quantile breakpoints for K_STATE_BINS equal-probability bins of U ~ N(0,1)
_U_BKPTS = scipy.stats.norm.ppf(np.arange(1, K_STATE_BINS) / K_STATE_BINS)

# Flat list of all (series, condition_label, condition_tag) for CSV
# condition_tag is a short string used in CSV
CONDITIONS : List[Tuple[str, str, float]] = (
    [("noise", f"sigma={s}", s) for s in NOISE_SIGMAS]
    + [("spatial", "state_K10", 0.0)]
    + [("temporal", "no_b_term", 0.0)]
)
# TAG = (series, condition_label) pair for unique keys
ALL_COND_KEYS = [(s, lbl) for s, lbl, _ in CONDITIONS]


# ===========================================================================
# Utility functions
# ===========================================================================

def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )


def imbens_manski_cval(width: float, sigma: float) -> float:
    """Imbens-Manski (2004) critical value (retained for compatibility)."""
    if sigma < 1e-12 or width / sigma > 200:
        return 1.96
    def eq(c: float) -> float:
        return scipy.stats.norm.cdf(c + width / sigma) - scipy.stats.norm.cdf(-c) - 0.95
    try:
        return float(scipy.optimize.brentq(eq, 0.001, 10.0))
    except ValueError:
        return 1.96


# ===========================================================================
# H degradation functions
# ===========================================================================

def apply_noise_degradation(
    P_crash: np.ndarray,
    sigma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Series 1: Multiplicative log-normal noise.
    H_noisy(i) = P_crash(i) * exp(eps),  eps ~ N(0, sigma^2).
    sigma=0 returns oracle H unchanged.
    Result clamped to [1e-4, 1-1e-4].
    """
    if sigma == 0.0:
        return P_crash.copy()
    eps = rng.normal(0.0, sigma, len(P_crash))
    H_noisy = P_crash * np.exp(eps)
    return np.clip(H_noisy, 1e-4, 1.0 - 1e-4)


def apply_spatial_degradation(
    P_crash: np.ndarray,
    U: np.ndarray,
    T_iv: np.ndarray,
) -> np.ndarray:
    """
    Series 2: State-level spatial coarsening.
    Each unit's H = within-stratum mean P_crash.
    Stratum = (T_val, U_quantile_bin), K_STATE_BINS × 2 = 20 strata.
    Simulates county-to-state aggregation of AADT.
    Result clamped to [1e-4, 1-1e-4].
    """
    U_bin = np.clip(np.digitize(U, _U_BKPTS), 0, K_STATE_BINS - 1)
    H_spatial = np.full_like(P_crash, fill_value=P_crash.mean())  # fallback
    for t_val in (0, 1):
        for u_b in range(K_STATE_BINS):
            mask = (T_iv == t_val) & (U_bin == u_b)
            n_mask = mask.sum()
            if n_mask > 0:
                H_spatial[mask] = P_crash[mask].mean()
            # else: fallback = global mean (already set)
    return np.clip(H_spatial, 1e-4, 1.0 - 1e-4)


def apply_temporal_degradation(
    T_iv: np.ndarray,
    U: np.ndarray,
) -> np.ndarray:
    """
    Series 3: Temporal coarsening (annual AADT average only).
    H_temporal(i) = sigmoid(eta_0 + eta_T*T + eta_U*U)  [no eta_B*b term].
    Simulates ignoring within-year behavioral variation.
    Result clamped to [1e-4, 1-1e-4].
    """
    H_temporal = sigmoid(ETA_0 + ETA_T * T_iv + ETA_U * U)
    return np.clip(H_temporal, 1e-4, 1.0 - 1e-4)


def compute_H_degraded(
    series: str,
    sigma: float,
    P_crash_true: np.ndarray,
    U: np.ndarray,
    T_iv: np.ndarray,
    noise_rng: np.random.Generator,
) -> np.ndarray:
    """Dispatch to appropriate degradation function."""
    if series == "noise":
        return apply_noise_degradation(P_crash_true, sigma, noise_rng)
    elif series == "spatial":
        return apply_spatial_degradation(P_crash_true, U, T_iv)
    elif series == "temporal":
        return apply_temporal_degradation(T_iv, U)
    else:
        raise ValueError(f"Unknown series: {series!r}")


# ===========================================================================
# IPSW Wald estimator (identical to hc_t2_reduction.py)
# ===========================================================================

def _ipsw_wald(
    Y_c1: np.ndarray,
    T_c1: np.ndarray,
    Z_c1: np.ndarray,
    W_c1: np.ndarray,
) -> Dict:
    """
    Hajek-type IPSW IV Wald on C=1 data.
    W_c1 = 1/H_degraded for C=1 units.
    """
    m1 = (Z_c1 == 1)
    m0 = (Z_c1 == 0)
    sw1 = W_c1[m1].sum()
    sw0 = W_c1[m0].sum()

    if sw1 < 1e-12 or sw0 < 1e-12:
        nan = float("nan")
        return {"tau": nan, "SE_tau": nan, "CI_L": nan, "CI_U": nan,
                "reject": False, "FS": nan, "RF": nan}

    EY_Z1 = float(np.dot(Y_c1[m1], W_c1[m1]) / sw1)
    EY_Z0 = float(np.dot(Y_c1[m0], W_c1[m0]) / sw0)
    ET_Z1 = float(np.dot(T_c1[m1], W_c1[m1]) / sw1)
    ET_Z0 = float(np.dot(T_c1[m0], W_c1[m0]) / sw0)

    RF = EY_Z1 - EY_Z0
    FS = ET_Z1 - ET_Z0

    if abs(FS) < 1e-8:
        nan = float("nan")
        return {"tau": nan, "SE_tau": nan, "CI_L": nan, "CI_U": nan,
                "reject": False, "FS": float(FS), "RF": float(RF)}

    tau = RF / FS

    resY1 = Y_c1[m1] - EY_Z1
    resY0 = Y_c1[m0] - EY_Z0
    resT1 = T_c1[m1] - ET_Z1
    resT0 = T_c1[m0] - ET_Z0
    w1, w0 = W_c1[m1], W_c1[m0]

    var_RF = (np.dot(w1**2, resY1**2) / sw1**2
            + np.dot(w0**2, resY0**2) / sw0**2)
    var_FS = (np.dot(w1**2, resT1**2) / sw1**2
            + np.dot(w0**2, resT0**2) / sw0**2)

    SE_tau = float(np.sqrt(max(1e-20,
        var_RF / FS**2 + RF**2 * var_FS / FS**4)))
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
    }


# ===========================================================================
# Calibration: N=10^6 × 5 seeds, all conditions from the same base data
# ===========================================================================

def run_calibration_seed_allcond(
    cell: str,
    rho: float,
    alpha_b: float,
    seed: int,
    N: int = N_FINAL,
) -> List[Dict]:
    """
    Generate ONE set of SCM data (N=10^6) for (cell, seed).
    Compute tau_LATE_true (twin simulation, independent of H).
    Apply all 6 degradation conditions to the SAME base data.
    Returns: list of flat dicts (one per condition).
    """
    rng = np.random.default_rng(seed)

    # --- exogenous noise (shared twin sim) ---
    U      = rng.standard_normal(N)
    eps_b  = rng.normal(0.0, STD_EPS_B, N)
    eps_v  = rng.normal(0.0, STD_EPS_V, N)
    v_base = 50.0 + 10.0 * U + eps_v

    # --- twin simulation: T(Z=0), T(Z=1) ---
    P_T_Z0 = sigmoid(LOGIT_P0 + rho * U + IV_STR * (0.0 - 0.5))
    P_T_Z1 = sigmoid(LOGIT_P0 + rho * U + IV_STR * (1.0 - 0.5))
    U_T    = rng.uniform(0.0, 1.0, N)
    T_Z0   = (U_T <= P_T_Z0).astype(float)
    T_Z1   = (U_T <= P_T_Z1).astype(float)
    is_complier = (T_Z1 > T_Z0)
    pi_c        = float(is_complier.mean())

    # --- structural counterfactuals ---
    b_T0 = GAMMA_B * U + eps_b
    b_T1 = alpha_b + GAMMA_B * U + eps_b

    def Y_cf(T_out: float, b_in: np.ndarray) -> np.ndarray:
        esc = 1.0 - ESC_REDUC * T_out
        dv  = (v_base + SPEED_MULT * b_in) * esc
        return (dv / 100.0) ** 4 + BETA_U * U   # eps_y cancels in differences

    Y_11 = Y_cf(1.0, b_T1)
    Y_10 = Y_cf(1.0, b_T0)
    Y_00 = Y_cf(0.0, b_T0)

    # tau_LATE_true (ATE_LATE = total effect among compliers)
    NIE_LATE    = float(np.mean((Y_11 - Y_10)[is_complier]))
    NDE_LATE    = float(np.mean((Y_10 - Y_00)[is_complier]))
    tau_LATE    = float(NIE_LATE + NDE_LATE)   # ATE_LATE
    kappa_LATE  = NIE_LATE / abs(NDE_LATE) if abs(NDE_LATE) > 1e-20 else float("nan")

    # --- IV draw (fresh Z for Wald) ---
    Z      = rng.binomial(1, 0.5, N).astype(float)
    P_T_IV = sigmoid(LOGIT_P0 + rho * U + IV_STR * (Z - 0.5))
    T_IV   = rng.binomial(1, P_T_IV).astype(float)

    b_IV     = alpha_b * T_IV + GAMMA_B * U + eps_b
    v_act_IV = v_base + SPEED_MULT * b_IV
    dv_IV    = v_act_IV * (1.0 - ESC_REDUC * T_IV)
    eps_y_iv = rng.normal(0.0, STD_EPS_Y, N)
    Y_IV     = (dv_IV / 100.0) ** 4 + BETA_U * U + eps_y_iv

    # Oracle P_crash (used as basis for all H degradations)
    P_crash_oracle = sigmoid(ETA_0 + ETA_T * T_IV + ETA_U * U + ETA_B * b_IV)
    C_IV = rng.binomial(1, P_crash_oracle).astype(float)
    mask_C1 = (C_IV == 1)
    n_C1    = int(mask_C1.sum())
    crash_rate = float(n_C1 / N)

    # IPSW weight statistics for oracle (useful baseline)
    W_oracle     = 1.0 / P_crash_oracle
    W_mean_ora   = float(np.mean(W_oracle))
    W_cv_ora     = float(np.std(W_oracle) / W_mean_ora) if W_mean_ora > 0 else float("nan")

    # Pre-generate noise for the noise series (deterministic given seed)
    # Use separate internal rng for noise to not shift base rng state
    noise_rng = np.random.default_rng(
        int(seed * 1_000_007 + 123456789) % (2**31)
    )

    results: List[Dict] = []

    for series, cond_label, sigma in CONDITIONS:
        # --- Compute degraded H ---
        H_deg = compute_H_degraded(
            series, sigma, P_crash_oracle, U, T_IV, noise_rng
        )
        W_deg = 1.0 / H_deg

        # IPSW weight stats (on all N units)
        W_mean_deg = float(np.mean(W_deg))
        W_std_deg  = float(np.std(W_deg))
        W_cv_deg   = float(W_std_deg / W_mean_deg) if W_mean_deg > 0 else float("nan")

        # IPSW Wald on C=1 data
        Y_c1 = Y_IV[mask_C1]
        T_c1 = T_IV[mask_C1]
        Z_c1 = Z[mask_C1]
        W_c1 = W_deg[mask_C1]

        ipsw = _ipsw_wald(Y_c1, T_c1, Z_c1, W_c1)

        tau_iv   = ipsw["tau"]
        bias     = (float(tau_iv) - tau_LATE) if not np.isnan(tau_iv) else float("nan")
        ci_l     = ipsw["CI_L"]
        ci_u     = ipsw["CI_U"]
        ci_cov   = bool(
            not np.isnan(ci_l) and not np.isnan(ci_u)
            and ci_l <= tau_LATE <= ci_u
        )

        results.append({
            "series"         : series,
            "condition"      : cond_label,
            "sigma"          : sigma,
            "cell"           : cell,
            "rho"            : rho,
            "alpha_b"        : alpha_b,
            "seed"           : seed,
            "N"              : N,
            "pi_c"           : pi_c,
            "tau_LATE_true"  : tau_LATE,
            "kappa_LATE_true": kappa_LATE,
            "NIE_LATE"       : NIE_LATE,
            "NDE_LATE"       : NDE_LATE,
            "tau_IV_IPSW"    : float(tau_iv) if not np.isnan(tau_iv) else float("nan"),
            "bias"           : bias,
            "SE_tau"         : float(ipsw["SE_tau"]),
            "CI_L"           : float(ci_l) if not np.isnan(ci_l) else float("nan"),
            "CI_U"           : float(ci_u) if not np.isnan(ci_u) else float("nan"),
            "CI_covers_true" : int(ci_cov),
            "FS_IPSW"        : float(ipsw["FS"]),
            "RF_IPSW"        : float(ipsw["RF"]),
            "W_mean_oracle"  : W_mean_ora,
            "W_cv_oracle"    : W_cv_ora,
            "W_mean_degraded": W_mean_deg,
            "W_cv_degraded"  : W_cv_deg,
            "n_C1"           : n_C1,
            "crash_rate"     : crash_rate,
        })

    return results


# ===========================================================================
# Size/Power: M=2000 reps × N_rep=10,000
# ===========================================================================

def _run_one_rep_pa2(
    rho: float,
    alpha_b: float,
    series: str,
    sigma: float,
    rep_seed: int,
    N: int = N_REP,
) -> Tuple[bool, float]:
    """
    One repetition for size/power analysis under given H degradation.
    Returns: (reject_IPSW, tau_IPSW)
    """
    rng = np.random.default_rng(rep_seed)

    U      = rng.standard_normal(N)
    eps_b  = rng.normal(0.0, STD_EPS_B, N)
    eps_v  = rng.normal(0.0, STD_EPS_V, N)
    eps_y  = rng.normal(0.0, STD_EPS_Y, N)
    v_base = 50.0 + 10.0 * U + eps_v

    Z   = rng.binomial(1, 0.5, N).astype(float)
    P_T = sigmoid(LOGIT_P0 + rho * U + IV_STR * (Z - 0.5))
    T   = rng.binomial(1, P_T).astype(float)

    b        = alpha_b * T + GAMMA_B * U + eps_b
    v_actual = v_base + SPEED_MULT * b
    dv       = v_actual * (1.0 - ESC_REDUC * T)
    Y        = (dv / 100.0) ** 4 + BETA_U * U + eps_y

    P_crash_oracle = sigmoid(ETA_0 + ETA_T * T + ETA_U * U + ETA_B * b)

    # Noise RNG (separate from data RNG)
    noise_rng = np.random.default_rng(
        int(rep_seed * 999_983 + 314159) % (2**31)
    )

    H_deg = compute_H_degraded(series, sigma, P_crash_oracle, U, T, noise_rng)

    C    = rng.binomial(1, P_crash_oracle).astype(float)
    mask = (C == 1)
    n_c1 = int(mask.sum())

    if n_c1 < 20:
        return (False, float("nan"))

    W_deg = 1.0 / H_deg
    r = _ipsw_wald(Y[mask], T[mask], Z[mask], W_deg[mask])

    return (r["reject"], float(r["tau"]) if not np.isnan(r["tau"]) else float("nan"))


def run_size_power_pa2(
    cell: str,
    rho: float,
    alpha_b: float,
    series: str,
    sigma: float,
    m_reps: int = M_REPS,
    n_rep: int = N_REP,
) -> Dict:
    """
    Run M=2000 repetitions for one (cell, series, condition).
    Returns dict with rejection rate, SE, and tau distribution stats.
    """
    rng_meta  = np.random.default_rng(REP_BASE_SEED)
    rep_seeds = rng_meta.integers(0, 2**31 - 1, m_reps)

    reject_list: List[bool]  = []
    tau_list:    List[float] = []

    for m in range(m_reps):
        rej, tau_val = _run_one_rep_pa2(rho, alpha_b, series, sigma,
                                         int(rep_seeds[m]), n_rep)
        reject_list.append(rej)
        tau_list.append(tau_val)

    valid_rej = [r for r in reject_list if isinstance(r, bool)]
    valid_tau = [t for t in tau_list    if not np.isnan(t)]
    n_valid   = len(valid_rej)

    rej_rate = float(np.mean(valid_rej)) if valid_rej else float("nan")
    se_rej   = (float(np.sqrt(rej_rate * (1 - rej_rate) / n_valid))
                if n_valid > 0 and not np.isnan(rej_rate) else float("nan"))
    tau_mean = float(np.mean(valid_tau)) if valid_tau else float("nan")
    tau_std  = float(np.std(valid_tau, ddof=1)) if len(valid_tau) > 1 else float("nan")

    return {
        "cell"              : cell,
        "rho"               : rho,
        "alpha_b"           : alpha_b,
        "series"            : series,
        "condition"         : f"sigma={sigma}" if series == "noise" else (
                                "state_K10" if series == "spatial" else "no_b_term"),
        "sigma"             : sigma,
        "M_reps"            : m_reps,
        "N_rep"             : n_rep,
        "n_valid"           : n_valid,
        "reject_rate_IPSW"  : rej_rate,
        "SE_reject_IPSW"    : se_rej,
        "tau_IV_IPSW_mean"  : tau_mean,
        "tau_IV_IPSW_std"   : tau_std,
    }


# ===========================================================================
# Aggregate helpers
# ===========================================================================

def _ms(vals: List[float]) -> Tuple[float, float]:
    """Mean and std (ddof=1) of a list of floats, ignoring NaNs."""
    arr = np.array([v for v in vals if not np.isnan(v)], dtype=float)
    if len(arr) == 0:
        return float("nan"), float("nan")
    return float(arr.mean()), float(arr.std(ddof=1)) if len(arr) > 1 else (float(arr[0]), float("nan"))


def aggregate_calib_cell_cond(rows: List[Dict]) -> Dict:
    """Aggregate 5-seed calibration rows for one (cell, series, condition)."""
    if not rows:
        return {}
    r0 = rows[0]

    tau_LATE_vals    = [r["tau_LATE_true"]   for r in rows]
    tau_iv_vals      = [r["tau_IV_IPSW"]     for r in rows]
    bias_vals        = [r["bias"]            for r in rows]
    ci_cov_vals      = [bool(r["CI_covers_true"]) for r in rows]
    W_cv_deg_vals    = [r["W_cv_degraded"]   for r in rows]

    tau_LATE_mean, tau_LATE_std = _ms(tau_LATE_vals)
    tau_iv_mean,   tau_iv_std   = _ms(tau_iv_vals)
    bias_mean,     bias_std     = _ms(bias_vals)
    W_cv_mean,     W_cv_std     = _ms(W_cv_deg_vals)

    coverage = float(np.mean(ci_cov_vals))   # fraction of seeds covered

    return {
        "cell"                     : r0["cell"],
        "rho"                      : r0["rho"],
        "alpha_b"                  : r0["alpha_b"],
        "series"                   : r0["series"],
        "condition"                : r0["condition"],
        "sigma"                    : r0["sigma"],
        "N_per_seed"               : r0["N"],
        "tau_LATE_true_mean"       : tau_LATE_mean,
        "tau_LATE_true_std"        : tau_LATE_std,
        "tau_IV_IPSW_mean"         : tau_iv_mean,
        "tau_IV_IPSW_std"          : tau_iv_std,
        "bias_mean"                : bias_mean,
        "bias_std"                 : bias_std,
        "CI_coverage"              : coverage,
        "W_cv_degraded_mean"       : W_cv_mean,
        "W_cv_degraded_std"        : W_cv_std,
        "per_seed"                 : rows,
    }


# ===========================================================================
# Breakdown point identification
# ===========================================================================

def identify_breakdown_noise(
    sp_results_by_key: Dict[Tuple[str, str], Dict],
    null_cell: str = "T2-null",
) -> Dict:
    """
    For the noise series, identify the smallest sigma at which SIZE > 2*alpha.
    Also report the sigma at which coverage (across seeds, T2-a) drops below 0.90.
    sp_results_by_key: {(series, condition_label, cell): sp_result}
    """
    size_threshold = 2.0 * ALPHA_LEVEL   # = 0.10
    cov_threshold  = 0.90

    # SIZE breakdown: T2-null, noise series
    size_breakdown = None
    for sigma in NOISE_SIGMAS:
        cond_label = f"sigma={sigma}"
        key = ("noise", cond_label, null_cell)
        if key in sp_results_by_key:
            rate = sp_results_by_key[key]["reject_rate_IPSW"]
            if not np.isnan(rate) and rate > size_threshold:
                size_breakdown = sigma
                break

    # Coverage breakdown: T2-a, noise series, across-seed coverage
    cov_breakdown = None
    for sigma in NOISE_SIGMAS:
        cond_label = f"sigma={sigma}"
        # This requires calib aggregates, not sp_results; we'll check it separately
        pass

    return {
        "size_breakdown_sigma"   : size_breakdown,
        "size_threshold"         : size_threshold,
        "note_coverage_breakdown": "see calib_aggregates[cell=T2-a][series=noise]",
    }


# ===========================================================================
# Git / env helpers
# ===========================================================================

def get_git_commit() -> str:
    try:
        r = subprocess.run(
            ["git", "-C", str(BASE_DIR / "pilot"), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        return r.stdout.strip() if r.returncode == 0 else "unavailable"
    except Exception:
        return "unavailable"


def make_env_log(git_hash: str, t_start: datetime) -> Dict:
    return {
        "experiment_name"     : "pa2_h_sensitivity",
        "date"                : t_start.isoformat(),
        "python_version"      : sys.version,
        "numpy_version"       : np.__version__,
        "scipy_version"       : scipy.__version__,
        "pandas_version"      : pd.__version__,
        "os"                  : platform.platform(),
        "hostname"            : socket.gethostname(),
        "gpu"                 : "none (CPU-only per DRAFT §4.3.3 rule 5)",
        "git_commit"          : git_hash,
        "seeds_main"          : SEEDS,
        "N_main"              : N_FINAL,
        "M_reps"              : M_REPS,
        "N_rep"               : N_REP,
        "alpha_level"         : ALPHA_LEVEL,
        "IV_STR"              : IV_STR,
        "ESC_REDUCTION"       : ESC_REDUC,
        "frozen_prereg"       : str(FROZEN_PREREG),
        "frozen_prereg_chmod" : "444 (read-only)",
        "noise_sigmas"        : NOISE_SIGMAS,
        "K_state_bins"        : K_STATE_BINS,
        "conditions"          : [(s, lbl) for s, lbl, _ in CONDITIONS],
        "rep_base_seed"       : REP_BASE_SEED,
    }


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    # Safety check: frozen preregistration must exist and be read-only
    if not FROZEN_PREREG.exists():
        raise FileNotFoundError(f"Frozen preregistration not found: {FROZEN_PREREG}")
    mode = oct(os.stat(FROZEN_PREREG).st_mode)[-3:]
    if mode != "444":
        print(f"WARNING: frozen prereg permissions = {mode} (expected 444)")

    t_start  = datetime.now()
    git_hash = get_git_commit()

    print("=" * 76)
    print("PA2: H(c) Quality Sensitivity Analysis")
    print("Ref: preregistration_FROZEN_2026-07-11.md §5.3 (READ-ONLY)")
    print(f"Date   : {t_start.isoformat()}")
    print(f"Python : {sys.version.split()[0]}  NumPy: {np.__version__}  "
          f"SciPy: {scipy.__version__}")
    print(f"Host   : {socket.gethostname()}  OS: {platform.system()}")
    print(f"git    : {git_hash}")
    print(f"GPU    : none (CPU-only)")
    print(f"Seeds (main)  : {SEEDS}  N/seed: {N_FINAL:,}")
    print(f"Size/Power    : M={M_REPS} reps × N_rep={N_REP:,} each")
    print(f"Conditions    : {len(CONDITIONS)}  "
          f"({len(NOISE_SIGMAS)} noise + 1 spatial + 1 temporal)")
    print(f"Cells         : {len(TIER2_CELLS)}")
    print("=" * 76)

    # -----------------------------------------------------------------------
    # Part A: Calibration (N=10^6 × 5 seeds, all conditions per base draw)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 76)
    print("PART A: Calibration  (N=10^6 × 5 seeds, generate base data ONCE per seed)")
    print("=" * 76)

    all_calib_rows: List[Dict] = []

    for spec in TIER2_CELLS:
        cell, rho, alpha_b = spec["cell"], spec["rho"], spec["alpha_b"]
        print(f"\n--- Cell {cell}: rho={rho}, alpha_b={alpha_b} ---")

        for seed in SEEDS:
            print(f"  seed={seed} ...", end="", flush=True)
            rows = run_calibration_seed_allcond(cell, rho, alpha_b, seed)
            all_calib_rows.extend(rows)

            # Quick print: bias per condition
            for r in rows:
                tag = f"{r['series']}:{r['condition']}"
                print(f"\n    [{tag:30s}] "
                      f"bias={r['bias']:+.5f}  "
                      f"CI_covers={bool(r['CI_covers_true'])}  "
                      f"W_cv_deg={r['W_cv_degraded']:.3f}", end="")
            print()

    # -----------------------------------------------------------------------
    # Aggregate calibration across 5 seeds
    # -----------------------------------------------------------------------
    df_calib = pd.DataFrame(all_calib_rows)
    calib_aggregates: Dict[Tuple[str, str, str], Dict] = {}

    for series, cond_label, sigma in CONDITIONS:
        for spec in TIER2_CELLS:
            cell = spec["cell"]
            mask = (
                (df_calib["series"]    == series) &
                (df_calib["condition"] == cond_label) &
                (df_calib["cell"]      == cell)
            )
            rows_sub = df_calib[mask].to_dict("records")
            agg = aggregate_calib_cell_cond(rows_sub)
            calib_aggregates[(series, cond_label, cell)] = agg

    # -----------------------------------------------------------------------
    # Part B: Size/Power (M=2000 reps × N_rep=10,000)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 76)
    print(f"PART B: Size/Power  (M={M_REPS} reps × N_rep={N_REP:,} each)")
    print("=" * 76)

    sp_results_by_key: Dict[Tuple[str, str, str], Dict] = {}

    for series, cond_label, sigma in CONDITIONS:
        for spec in TIER2_CELLS:
            cell, rho, alpha_b = spec["cell"], spec["rho"], spec["alpha_b"]
            role  = "SIZE" if "null" in cell else "POWER"
            print(f"  [{role}] {cell} | {series}:{cond_label} ...", end="", flush=True)

            sp = run_size_power_pa2(cell, rho, alpha_b, series, sigma, M_REPS, N_REP)
            sp_results_by_key[(series, cond_label, cell)] = sp

            print(
                f"  reject_IPSW={sp['reject_rate_IPSW']:.4f} "
                f"± {sp['SE_reject_IPSW']:.4f}  "
                f"tau_mean={sp['tau_IV_IPSW_mean']:+.5f}"
            )

    # -----------------------------------------------------------------------
    # Breakdown point identification (noise series, SIZE cell = T2-null)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 76)
    print("BREAKDOWN POINT ANALYSIS")
    print("=" * 76)

    size_threshold = 2.0 * ALPHA_LEVEL   # 0.10
    cov_threshold  = 0.90

    breakdown_summary: Dict = {
        "size_breakdown_sigma"    : None,
        "coverage_breakdown_sigma": None,
        "size_threshold"          : size_threshold,
        "coverage_threshold"      : cov_threshold,
        "noise_series_SIZE"       : {},
        "noise_series_COVERAGE_T2a": {},
        "spatial_SIZE"            : None,
        "temporal_SIZE"           : None,
    }

    print(f"\nNoise series — SIZE (T2-null) and COVERAGE (T2-a, all 5 seeds):")
    print(f"  Threshold SIZE > {size_threshold:.2f} or COVERAGE < {cov_threshold:.2f}")

    for sigma in NOISE_SIGMAS:
        cond_label = f"sigma={sigma}"

        # SIZE (T2-null)
        sp_null = sp_results_by_key.get(("noise", cond_label, "T2-null"), {})
        rate_null = sp_null.get("reject_rate_IPSW", float("nan"))
        breakdown_summary["noise_series_SIZE"][cond_label] = rate_null

        # COVERAGE (T2-a, across 5 seeds)
        agg_t2a = calib_aggregates.get(("noise", cond_label, "T2-a"), {})
        cov_t2a = agg_t2a.get("CI_coverage", float("nan"))
        breakdown_summary["noise_series_COVERAGE_T2a"][cond_label] = cov_t2a

        print(f"  sigma={sigma:.1f}: SIZE={rate_null:.4f}  COVERAGE(T2-a)={cov_t2a:.2f}", end="")

        # Breakdown detection
        size_broken = (not np.isnan(rate_null)) and (rate_null > size_threshold)
        cov_broken  = (not np.isnan(cov_t2a))  and (cov_t2a  < cov_threshold)

        if size_broken and breakdown_summary["size_breakdown_sigma"] is None:
            breakdown_summary["size_breakdown_sigma"] = sigma
            print(f"  << SIZE BREAKDOWN (sigma={sigma})", end="")
        if cov_broken  and breakdown_summary["coverage_breakdown_sigma"] is None:
            breakdown_summary["coverage_breakdown_sigma"] = sigma
            print(f"  << COVERAGE BREAKDOWN (sigma={sigma})", end="")
        print()

    # Spatial / temporal SIZE
    sp_spatial_null  = sp_results_by_key.get(("spatial",  "state_K10", "T2-null"), {})
    sp_temporal_null = sp_results_by_key.get(("temporal", "no_b_term", "T2-null"), {})
    breakdown_summary["spatial_SIZE"]  = sp_spatial_null.get("reject_rate_IPSW",  float("nan"))
    breakdown_summary["temporal_SIZE"] = sp_temporal_null.get("reject_rate_IPSW", float("nan"))

    print(f"\nSpatial  coarsening SIZE (T2-null): "
          f"{breakdown_summary['spatial_SIZE']:.4f}"
          f"  (threshold={size_threshold:.2f})")
    print(f"Temporal coarsening SIZE (T2-null): "
          f"{breakdown_summary['temporal_SIZE']:.4f}"
          f"  (threshold={size_threshold:.2f})")

    if not np.isnan(breakdown_summary["spatial_SIZE"]) and breakdown_summary["spatial_SIZE"] > size_threshold:
        print("  << SPATIAL COARSENING BREAKS SIZE CONTROL")
    if not np.isnan(breakdown_summary["temporal_SIZE"]) and breakdown_summary["temporal_SIZE"] > size_threshold:
        print("  << TEMPORAL COARSENING BREAKS SIZE CONTROL")

    # -----------------------------------------------------------------------
    # Summary tables for console
    # -----------------------------------------------------------------------
    print("\n" + "=" * 76)
    print("TABLE A: Calibration bias and coverage across seeds")
    print(f"{'series':10s} {'condition':15s} {'cell':8s} "
          f"{'tau_true':10s} {'bias_mean':10s} {'bias_std':8s} {'coverage':8s} {'W_cv':6s}")
    print("-" * 78)
    for series, cond_label, _ in CONDITIONS:
        for spec in TIER2_CELLS:
            cell = spec["cell"]
            agg  = calib_aggregates.get((series, cond_label, cell), {})
            if not agg:
                continue
            print(
                f"{series:10s} {cond_label:15s} {cell:8s} "
                f"{agg['tau_LATE_true_mean']:+10.5f} "
                f"{agg['bias_mean']:+10.5f} "
                f"{agg['bias_std']:8.2e} "
                f"{agg['CI_coverage']:8.2f} "
                f"{agg['W_cv_degraded_mean']:6.3f}"
            )

    print("\n" + "=" * 76)
    print("TABLE B: Size/Power (rejection rates, M=2000 reps)")
    print(f"{'series':10s} {'condition':15s} {'cell':8s} {'role':5s} "
          f"{'rej_IPSW':9s} {'±SE':7s} {'tau_mean':9s}")
    print("-" * 70)
    for series, cond_label, _ in CONDITIONS:
        for spec in TIER2_CELLS:
            cell = spec["cell"]
            sp   = sp_results_by_key.get((series, cond_label, cell), {})
            if not sp:
                continue
            role = "SIZE" if "null" in cell else "POWER"
            print(
                f"{series:10s} {cond_label:15s} {cell:8s} {role:5s} "
                f"{sp['reject_rate_IPSW']:9.4f} "
                f"{sp['SE_reject_IPSW']:7.4f} "
                f"{sp['tau_IV_IPSW_mean']:+9.5f}"
            )

    # -----------------------------------------------------------------------
    # Save per-seed CSV
    # -----------------------------------------------------------------------
    bool_cols = [c for c in df_calib.columns if df_calib[c].dtype == bool]
    for col in bool_cols:
        df_calib[col] = df_calib[col].astype(int)

    csv_calib_path = RESULTS_DIR / "pa2_h_sensitivity.csv"
    df_calib.to_csv(csv_calib_path, index=False)
    print(f"\nCalibration CSV saved: {csv_calib_path}")

    # -----------------------------------------------------------------------
    # Save aggregate JSON
    # -----------------------------------------------------------------------
    t_end   = datetime.now()
    elapsed = (t_end - t_start).total_seconds()

    env_log = make_env_log(git_hash, t_start)

    # Serialisable calib aggregates (drop embedded per_seed for JSON top-level)
    calib_agg_serialisable: Dict[str, Dict] = {}
    for (series, cond_label, cell), agg in calib_aggregates.items():
        k = f"{series}__{cond_label}__{cell}"
        agg_copy = {kk: vv for kk, vv in agg.items() if kk != "per_seed"}
        calib_agg_serialisable[k] = agg_copy

    # Serialisable sp results
    sp_serialisable: Dict[str, Dict] = {}
    for (series, cond_label, cell), sp in sp_results_by_key.items():
        k = f"{series}__{cond_label}__{cell}"
        sp_serialisable[k] = sp

    output_json = {
        "computation_type"       : "pa2_h_sensitivity",
        "description"            : (
            "PA2: H(c) quality sensitivity analysis. "
            "3 degradation series (noise/spatial/temporal) × 5 Tier-2 cells. "
            "Calibration N=10^6 × 5 seeds; size/power M=2000 × N_rep=10,000. "
            "Ref: preregistration_FROZEN_2026-07-11.md §5.3 PA2. CPU only."
        ),
        "frozen_prereg"          : str(FROZEN_PREREG),
        "conditions"             : [(s, lbl) for s, lbl, _ in CONDITIONS],
        "cells"                  : [c["cell"] for c in TIER2_CELLS],
        "breakdown_summary"      : breakdown_summary,
        "calib_aggregates"       : calib_agg_serialisable,
        "size_power"             : sp_serialisable,
        "elapsed_seconds"        : elapsed,
        "env_log"                : env_log,
    }

    json_path = RESULTS_DIR / "pa2_h_sensitivity_aggregate.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output_json, f, indent=2, ensure_ascii=False, default=str)
    print(f"Aggregate JSON saved: {json_path}")

    # -----------------------------------------------------------------------
    # Save human-readable summary MD (new file — NOT the frozen prereg)
    # -----------------------------------------------------------------------
    md_path = RESULTS_DIR / "pa2_results_2026-07-11.md"
    _write_summary_md(
        md_path, git_hash, t_start, elapsed,
        calib_aggregates, sp_results_by_key, breakdown_summary,
    )
    print(f"Summary MD saved:     {md_path}")

    # -----------------------------------------------------------------------
    # Sanity checks on output files
    # -----------------------------------------------------------------------
    for p in [csv_calib_path, json_path, md_path]:
        assert p.exists(), f"Output file missing: {p}"
        assert p.stat().st_size > 0, f"Output file empty: {p}"

    print(f"\nElapsed : {elapsed:.1f} s")
    print(f"git     : {git_hash}")
    print("\nOutput files (VERIFY ALL EXIST):")
    print(f"  {csv_calib_path}")
    print(f"  {json_path}")
    print(f"  {md_path}")

    print("\nBreakdown point summary:")
    print(f"  Noise SIZE breakdown sigma  : {breakdown_summary['size_breakdown_sigma']}")
    print(f"  Noise COV  breakdown sigma  : {breakdown_summary['coverage_breakdown_sigma']}")
    print(f"  Spatial SIZE (T2-null)      : {breakdown_summary['spatial_SIZE']:.4f}")
    print(f"  Temporal SIZE (T2-null)     : {breakdown_summary['temporal_SIZE']:.4f}")

    print("=" * 76)
    print("PA2 COMPLETE")
    print("=" * 76)


# ===========================================================================
# Summary MD writer
# ===========================================================================

def _write_summary_md(
    md_path: Path,
    git_hash: str,
    t_start: datetime,
    elapsed: float,
    calib_aggregates: Dict,
    sp_results: Dict,
    breakdown: Dict,
) -> None:
    """Write human-readable summary to pa2_results_2026-07-11.md (NEW FILE)."""

    lines: List[str] = []
    A = lines.append

    A("# PA2: H(c) Quality Sensitivity Analysis — Results")
    A("")
    A("**Date**: 2026-07-11")
    A(f"**git commit**: `{git_hash}`")
    A(f"**Elapsed**: {elapsed:.1f} s (CPU only, no GPU)")
    A(f"**Ref**: preregistration_FROZEN_2026-07-11.md §5.3 PA2 (chmod 444, read-only)")
    A("")
    A("## 1. Design Summary")
    A("")
    A("Three H degradation series applied to the oracle H = P(C=1|T,U,b):")
    A("")
    A("| Series | Conditions | Description |")
    A("|--------|-----------|-------------|")
    A("| Noise  | sigma in {0.0, 0.1, 0.2, 0.4} | Multiplicative log-normal: H_noisy = H_true * exp(N(0,sigma^2)) |")
    A("| Spatial | state_K10 | County-to-state: within-stratum mean H (10 U-bins x 2 T strata) |")
    A("| Temporal | no_b_term | Annual average: H_temporal = sigmoid(eta_0 + eta_T*T + eta_U*U) (no b term) |")
    A("")
    A("Cells: T2-a (rho=0.5, alpha_b=0.3), T2-b (rho=1.0, alpha_b=0.3), "
      "T2-c (rho=0.5, alpha_b=3.972), T2-d (rho=1.0, alpha_b=3.972), "
      "T2-null (rho=0.5, alpha_b=4.997)")
    A("")
    A(f"Calibration: N=10^6 x 5 seeds.  Size/Power: M={M_REPS} reps x N_rep={N_REP}.")
    A("")
    A("## 2. Calibration Results (bias and CI coverage, mean across 5 seeds)")
    A("")
    A("| Series | Condition | Cell | tau_true | bias_mean | bias_std | CI_coverage | W_cv |")
    A("|--------|-----------|------|----------|-----------|----------|-------------|------|")

    for series, cond_label, _ in CONDITIONS:
        for spec in TIER2_CELLS:
            cell = spec["cell"]
            agg  = calib_aggregates.get((series, cond_label, cell), {})
            if not agg:
                continue
            A(
                f"| {series} | {cond_label} | {cell} "
                f"| {agg['tau_LATE_true_mean']:+.5f} "
                f"| {agg['bias_mean']:+.5f} "
                f"| {agg['bias_std']:.2e} "
                f"| {agg['CI_coverage']:.2f} "
                f"| {agg['W_cv_degraded_mean']:.3f} |"
            )

    A("")
    A("## 3. Size / Power Results (M=2000 reps x N_rep=10,000)")
    A("")
    A("| Series | Condition | Cell | Role | rej_IPSW | SE | tau_mean |")
    A("|--------|-----------|------|------|----------|-----|---------|")

    for series, cond_label, _ in CONDITIONS:
        for spec in TIER2_CELLS:
            cell = spec["cell"]
            sp   = sp_results.get((series, cond_label, cell), {})
            if not sp:
                continue
            role = "SIZE" if "null" in cell else "POWER"
            A(
                f"| {series} | {cond_label} | {cell} | {role} "
                f"| {sp['reject_rate_IPSW']:.4f} "
                f"| {sp['SE_reject_IPSW']:.4f} "
                f"| {sp['tau_IV_IPSW_mean']:+.5f} |"
            )

    A("")
    A("## 4. Breakdown Point")
    A("")
    A("Criteria: SIZE (T2-null) > 2*alpha=0.10 OR Coverage (T2-a) < 0.90")
    A("")
    A("### 4.1 Noise Series")
    A("")
    A("| sigma | SIZE (T2-null) | COVERAGE (T2-a) | SIZE_broken | COV_broken |")
    A("|-------|---------------|----------------|-------------|-----------|")

    for sigma in NOISE_SIGMAS:
        cond_label = f"sigma={sigma}"
        size_val = breakdown["noise_series_SIZE"].get(cond_label, float("nan"))
        cov_val  = breakdown["noise_series_COVERAGE_T2a"].get(cond_label, float("nan"))
        size_broken = (not np.isnan(size_val)) and (size_val > 2 * ALPHA_LEVEL)
        cov_broken  = (not np.isnan(cov_val))  and (cov_val  < 0.90)
        A(
            f"| {sigma} "
            f"| {size_val:.4f} "
            f"| {cov_val:.2f} "
            f"| {'YES' if size_broken else 'no'} "
            f"| {'YES' if cov_broken  else 'no'} |"
        )

    A("")
    A(f"**SIZE breakdown sigma**: {breakdown['size_breakdown_sigma']}")
    A(f"**Coverage breakdown sigma**: {breakdown['coverage_breakdown_sigma']}")
    A("")
    A("### 4.2 Spatial / Temporal Coarsening")
    A("")
    A("| Series | Condition | SIZE (T2-null) | Broken? |")
    A("|--------|-----------|----------------|---------|")

    sp_s = breakdown.get("spatial_SIZE",  float("nan"))
    sp_t = breakdown.get("temporal_SIZE", float("nan"))
    A(f"| spatial  | state_K10 | {sp_s:.4f} "
      f"| {'YES' if not np.isnan(sp_s) and sp_s > 2*ALPHA_LEVEL else 'no'} |")
    A(f"| temporal | no_b_term | {sp_t:.4f} "
      f"| {'YES' if not np.isnan(sp_t) and sp_t > 2*ALPHA_LEVEL else 'no'} |")

    A("")
    A("## 5. Interpretation")
    A("")
    A("- **Noise series (sigma=0)**: oracle baseline. SIZE should match hc_t2_reduction.py (0.0500).")
    A("- **Noise sigma > 0**: multiplicative log-normal noise on H inflates IPSW weight variance.")
    A("  Coverage may decrease as sigma increases due to widened weight distribution.")
    A("- **Spatial coarsening**: within-stratum mean H loses individual-level variation.")
    A("  Bias is expected if stratum boundaries do not align with true H contours.")
    A("- **Temporal coarsening (no b term)**: H misspecification biases IPSW weights.")
    A("  Particularly severe for high-alpha_b cells where b strongly predicts crash.")
    A("")
    A("## 6. Files")
    A("")
    A(f"- Calibration CSV: `pilot/results/pa2_h_sensitivity.csv`")
    A(f"- Aggregate JSON:  `pilot/results/pa2_h_sensitivity_aggregate.json`")
    A(f"- This summary:    `pilot/results/pa2_results_2026-07-11.md`")
    A(f"- git commit:      `{git_hash}`")
    A("")
    A("*Generated by scripts/pa2_h_sensitivity.py*")

    md_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
