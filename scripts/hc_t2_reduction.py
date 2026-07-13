"""
HC_T2(b)(c): IV Wald IPSW point test and kappa_LATE identification via Horowitz-Manski bounds.
Based on kappa_LATE = 1 + tau/D, where tau = ATE_LATE is point-identified by IV Wald
and D = |NDE_LATE| is partially identified via Horowitz-Manski trimming of E[S(0)^4|complier].
Outputs: results/hc_t2_reduction_per_cell.csv, results/hc_t2_reduction_size_power.csv,
         results/hc_t2_reduction_aggregate.json
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
import scipy.stats
import scipy.optimize

# Paths
BASE_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Frozen SCM parameters
LOGIT_P0: float = float(np.log(0.55 / 0.45))   # logit(0.55) ≈ 0.20067
GAMMA_B:  float = 0.50
STD_EPS_B: float = 0.30
STD_EPS_V: float = 5.00
BETA_U:   float = 0.10
STD_EPS_Y: float = 0.05
ESC_REDUCTION: float = 0.30   # illustrative modeling choice
SPEED_MULT: float = 5.0
ETA_0: float = -2.0
ETA_T: float = -0.5
ETA_U: float =  0.3
ETA_B: float =  0.2
IV_STR: float = 1.5   # logit-scale IV shift

# Physics constant for HM bounds (D = PHI_C * E[S(0)^4 | complier])
PHI_C: float = (1.0 - (1.0 - ESC_REDUCTION) ** 4) / (100.0 ** 4)  # (1 - 0.7^4)/100^4

# G4 alpha_b (null cell for size testing)
G4_ALPHA_B: float = 4.99720863

# Main calibration
SEEDS:   List[int] = [42, 123, 456, 789, 2026]
N_FINAL: int = 1_000_000

# Size / power repetitions
M_REPS:   int = 2000
N_REP:    int = 10_000
REP_BASE_SEED: int = 77777   # base seed for generating M repetition seeds

# Tier-2 cells
TIER2_CELLS: List[Dict] = [
    {"cell": "T2-a",    "rho": 0.5, "alpha_b": 0.3},
    {"cell": "T2-b",    "rho": 1.0, "alpha_b": 0.3},
    {"cell": "T2-c",    "rho": 0.5, "alpha_b": 3.972},
    {"cell": "T2-d",    "rho": 1.0, "alpha_b": 3.972},
    {"cell": "T2-null", "rho": 0.5, "alpha_b": G4_ALPHA_B},  # size check
]

ALPHA_LEVEL: float = 0.05


# Helpers

def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )


def imbens_manski_cval(width: float, sigma: float) -> float:
    """
    Compute the Imbens-Manski (2004) critical value C_n such that
    Phi(C_n + width/sigma) - Phi(-C_n) = 0.95.

    For large width/sigma this converges to 1.96 (standard normal critical
    value). For narrow identification regions relative to sigma, C_n > 1.96.

    Parameters
    ----------
    width : float
        Width of the identification region (theta_U - theta_L).
    sigma : float
        Sampling standard deviation of the endpoint estimators.
    """
    if sigma < 1e-12 or width / sigma > 200:
        return 1.96
    def eq(c: float) -> float:
        return scipy.stats.norm.cdf(c + width / sigma) - scipy.stats.norm.cdf(-c) - 0.95
    try:
        return float(scipy.optimize.brentq(eq, 0.001, 10.0))
    except ValueError:
        return 1.96


# Task 1 + 3 (main calibration): one (cell, seed) run — N = N_FINAL

def run_calibration_seed(
    cell: str,
    rho: float,
    alpha_b: float,
    seed: int,
    N: int = N_FINAL,
) -> Dict:
    """
    Twin simulation for tau_LATE_true (Task 1) and HM bounds (Task 3),
    plus IPSW-corrected Wald point estimate + 95% CI (Task 2 main estimate).

    Returns a flat dict of all scalar outputs.
    """
    rng = np.random.default_rng(seed)

    # --- exogenous noise (shared across counterfactuals for twin sim) ---
    U     = rng.standard_normal(N)
    eps_b = rng.normal(0.0, STD_EPS_B, N)
    eps_v = rng.normal(0.0, STD_EPS_V, N)
    # eps_y cancels in NIE/NDE cross-world differences

    v_base = 50.0 + 10.0 * U + eps_v

    # --- twin simulation: T(Z=0), T(Z=1) via shared U_T ---
    P_T_Z0 = sigmoid(LOGIT_P0 + rho * U + IV_STR * (0.0 - 0.5))
    P_T_Z1 = sigmoid(LOGIT_P0 + rho * U + IV_STR * (1.0 - 0.5))
    U_T    = rng.uniform(0.0, 1.0, N)
    T_Z0   = (U_T <= P_T_Z0).astype(float)
    T_Z1   = (U_T <= P_T_Z1).astype(float)
    is_complier = (T_Z1 > T_Z0)
    pi_c        = float(is_complier.mean())
    n_complier  = int(is_complier.sum())

    # --- structural counterfactuals ---
    b_T0 = GAMMA_B * U + eps_b               # b when T set to 0
    b_T1 = alpha_b + GAMMA_B * U + eps_b     # b when T set to 1

    def Y_cf(T_out: float, b_in: np.ndarray) -> np.ndarray:
        esc = 1.0 - ESC_REDUCTION * T_out
        dv  = (v_base + SPEED_MULT * b_in) * esc
        return (dv / 100.0) ** 4 + BETA_U * U   # eps_y cancels

    Y_11 = Y_cf(1.0, b_T1)   # Y(T=1, b(1))
    Y_10 = Y_cf(1.0, b_T0)   # Y(T=1, b(0))
    Y_00 = Y_cf(0.0, b_T0)   # Y(T=0, b(0))

    # Population NIE / NDE / ATE
    NIE_pop = float(np.mean(Y_11 - Y_10))
    NDE_pop = float(np.mean(Y_10 - Y_00))
    ATE_pop = float(np.mean(Y_11 - Y_00))
    kappa_pop = NIE_pop / abs(NDE_pop)

    # Complier LATE (twin simulation — exact structural truth)
    NIE_LATE = float(np.mean((Y_11 - Y_10)[is_complier]))
    NDE_LATE = float(np.mean((Y_10 - Y_00)[is_complier]))
    tau_LATE = float(NIE_LATE + NDE_LATE)           # ATE_LATE = complier total effect
    D_true   = abs(NDE_LATE)
    kappa_LATE_true = NIE_LATE / abs(NDE_LATE)

    # Algebraic identity check: kappa_LATE = 1 + tau/D (CLAIM 1, C2)
    kappa_recon = float(1.0 + tau_LATE / D_true)
    claim1_match = bool(abs(kappa_LATE_true - kappa_recon) < 1e-6)

    # E[U|complier] (source of LATE vs ATE divergence)
    EU_complier = float(np.mean(U[is_complier]))
    EU_pop      = float(np.mean(U))   # ≈ 0 by construction

    # --- HM trimming bounds (Task 3) ---
    # S(0) = v_base + 5*b(T=0) = physical speed under no-treatment
    S0  = v_base + SPEED_MULT * b_T0
    W   = S0 ** 4                             # W = S(0)^4
    Ws  = np.sort(W)
    k   = max(1, int(round(pi_c * N)))
    D_L = float(PHI_C * Ws[:k].mean())       # smallest pi_c mass of W
    D_U = float(PHI_C * Ws[-k:].mean())      # largest  pi_c mass of W

    # Sanity: D_L <= D_true <= D_U
    D_contains = bool(D_L <= D_true <= D_U)

    # kappa identification region (from tau + D bounds)
    kap_L_ident = float(1.0 + tau_LATE / D_L)   # most negative (D_L smallest)
    kap_U_ident = float(1.0 + tau_LATE / D_U)   # closest to 1 (D_U largest)

    # Coverage: does identification region contain kappa_LATE_true?
    ident_covers_LATE = bool(kap_L_ident <= kappa_LATE_true <= kap_U_ident)
    ident_all_lt1     = bool(kap_U_ident < 1.0)  # tau<0 => all region < 1

    # --- IV Wald (fresh Z draw, independent) ---
    Z = rng.binomial(1, 0.5, N).astype(float)
    P_T_IV = sigmoid(LOGIT_P0 + rho * U + IV_STR * (Z - 0.5))
    T_IV   = rng.binomial(1, P_T_IV).astype(float)

    b_IV     = alpha_b * T_IV + GAMMA_B * U + eps_b
    v_act_IV = v_base + SPEED_MULT * b_IV
    dv_IV    = v_act_IV * (1.0 - ESC_REDUCTION * T_IV)
    eps_y_iv = rng.normal(0.0, STD_EPS_Y, N)
    Y_IV     = (dv_IV / 100.0) ** 4 + BETA_U * U + eps_y_iv

    # Oracle IPSW weights (B4)
    P_crash_IV = sigmoid(ETA_0 + ETA_T * T_IV + ETA_U * U + ETA_B * b_IV)
    C_IV       = rng.binomial(1, P_crash_IV).astype(float)
    W_ipsw     = 1.0 / P_crash_IV     # oracle weight for all N units

    # Mask to C=1 records
    mask_C1 = (C_IV == 1)
    n_C1    = int(mask_C1.sum())

    Y_c1 = Y_IV[mask_C1]
    T_c1 = T_IV[mask_C1]
    Z_c1 = Z[mask_C1]
    W_c1 = W_ipsw[mask_C1]      # oracle IPSW weights for C=1 units

    # --- IPSW-corrected Wald (Task 2a) ---
    ipsw_res = _ipsw_wald(Y_c1, T_c1, Z_c1, W_c1)

    # --- Uncorrected C=1 Wald (Task 2b, comparison) ---
    raw_res  = _unweighted_wald(Y_c1, T_c1, Z_c1)

    # Coverage: IPSW CI covers tau_LATE_true?
    tau_iv_ipsw_ci_covers = bool(
        ipsw_res["CI_L"] <= tau_LATE <= ipsw_res["CI_U"]
    )

    # Imbens-Manski two-layer CI (Task 3)
    # Endpoint SEs: SE(kap_L) = SE_tau/D_L,  SE(kap_U) = SE_tau/D_U
    se_tau_ipsw = ipsw_res["SE_tau"]
    se_kap_L = se_tau_ipsw / D_L if D_L > 1e-12 else 1e6
    se_kap_U = se_tau_ipsw / D_U if D_U > 1e-12 else 1e6

    # Imbens-Manski critical value (one shared C_n for both endpoints)
    width_ident = kap_U_ident - kap_L_ident
    sigma_max   = max(se_kap_L, se_kap_U)  # conservative: use larger SE
    c_im        = imbens_manski_cval(width_ident, sigma_max)

    # Using tau_IV_IPSW as the estimator for tau in the bounds
    tau_iv = ipsw_res["tau"]
    kap_L_from_iv = float(1.0 + tau_iv / D_L)
    kap_U_from_iv = float(1.0 + tau_iv / D_U)

    im_ci_L = float(kap_L_from_iv - c_im * se_kap_L)
    im_ci_U = float(kap_U_from_iv + c_im * se_kap_U)

    # IM CI coverage and < 1 check
    im_ci_covers_LATE = bool(im_ci_L <= kappa_LATE_true <= im_ci_U)
    im_ci_all_lt1     = bool(im_ci_U < 1.0)

    return {
        "cell"                    : cell,
        "rho"                     : rho,
        "alpha_b"                 : alpha_b,
        "seed"                    : seed,
        "N"                       : N,
        # Complier stats
        "pi_c"                    : pi_c,
        "n_complier"              : n_complier,
        "EU_complier"             : EU_complier,
        "EU_pop"                  : EU_pop,
        # Population structural truths
        "NIE_pop"                 : NIE_pop,
        "NDE_pop"                 : NDE_pop,
        "ATE_pop"                 : ATE_pop,
        "kappa_pop"               : kappa_pop,
        # Complier structural truths (Task 1)
        "NIE_LATE"                : NIE_LATE,
        "NDE_LATE"                : NDE_LATE,
        "tau_LATE_true"           : tau_LATE,
        "D_true"                  : D_true,
        "kappa_LATE_true"         : kappa_LATE_true,
        "kappa_recon_from_tau_D"  : kappa_recon,
        "claim1_kappa_eq_1_tau_D" : claim1_match,
        "tau_LATE_lt0"            : bool(tau_LATE < 0),
        # HM bounds (Task 3)
        "D_L"                     : D_L,
        "D_U"                     : D_U,
        "D_contains_D_true"       : D_contains,
        "kap_L_ident"             : kap_L_ident,
        "kap_U_ident"             : kap_U_ident,
        "ident_covers_LATE"       : ident_covers_LATE,
        "ident_all_lt1"           : ident_all_lt1,
        # IPSW Wald (Task 2 — main estimate)
        "tau_IV_IPSW"             : ipsw_res["tau"],
        "SE_tau_IPSW"             : ipsw_res["SE_tau"],
        "CI_L_IPSW"               : ipsw_res["CI_L"],
        "CI_U_IPSW"               : ipsw_res["CI_U"],
        "FS_IPSW"                 : ipsw_res["FS"],
        "RF_IPSW"                 : ipsw_res["RF"],
        "tau_IPSW_covers_true"    : tau_iv_ipsw_ci_covers,
        # C=1 raw (uncorrected, Task 2 comparison)
        "tau_IV_C1_raw"           : raw_res["tau"],
        "SE_tau_C1_raw"           : raw_res["SE_tau"],
        "CI_L_C1_raw"             : raw_res["CI_L"],
        "CI_U_C1_raw"             : raw_res["CI_U"],
        "FS_C1_raw"               : raw_res["FS"],
        # Imbens-Manski two-layer CI (Task 3)
        "tau_IV_used_for_bounds"  : tau_iv,
        "c_IM"                    : float(c_im),
        "kap_L_from_IV"           : kap_L_from_iv,
        "kap_U_from_IV"           : kap_U_from_iv,
        "im_ci_L"                 : im_ci_L,
        "im_ci_U"                 : im_ci_U,
        "im_ci_covers_LATE"       : im_ci_covers_LATE,
        "im_ci_all_lt1"           : im_ci_all_lt1,
        # Sample info
        "n_C1"                    : n_C1,
        "crash_rate"              : float(n_C1 / N),
    }


def _ipsw_wald(
    Y_c1: np.ndarray,
    T_c1: np.ndarray,
    Z_c1: np.ndarray,
    W_c1: np.ndarray,
) -> Dict:
    """
    IPSW-weighted IV Wald estimator (Hajek) on C=1 data with oracle weights.

    Returns tau_IV_IPSW, SE, 95% CI, z-stat, p-value.
    """
    m1 = (Z_c1 == 1)
    m0 = (Z_c1 == 0)

    w1, w0 = W_c1[m1], W_c1[m0]
    sw1,  sw0  = w1.sum(), w0.sum()

    EY_Z1 = float(np.dot(Y_c1[m1], w1) / sw1)
    EY_Z0 = float(np.dot(Y_c1[m0], w0) / sw0)
    ET_Z1 = float(np.dot(T_c1[m1], w1) / sw1)
    ET_Z0 = float(np.dot(T_c1[m0], w0) / sw0)

    RF = EY_Z1 - EY_Z0
    FS = ET_Z1 - ET_Z0

    if abs(FS) < 1e-8:
        return {"tau": float("nan"), "SE_tau": float("nan"),
                "CI_L": float("nan"), "CI_U": float("nan"),
                "z_stat": float("nan"), "p_value": float("nan"),
                "reject": False, "FS": float(FS), "RF": float(RF)}

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

    SE_tau = float(np.sqrt(max(1e-20, var_RF / FS ** 2 + RF ** 2 * var_FS / FS ** 4)))
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


def _unweighted_wald(
    Y_c1: np.ndarray,
    T_c1: np.ndarray,
    Z_c1: np.ndarray,
) -> Dict:
    """
    Unweighted IV Wald on C=1 data only (no IPSW = biased for collider).
    Used as comparison to demonstrate collider bias.
    """
    m1 = (Z_c1 == 1)
    m0 = (Z_c1 == 0)
    n1, n0 = int(m1.sum()), int(m0.sum())

    if n1 < 5 or n0 < 5:
        return {"tau": float("nan"), "SE_tau": float("nan"),
                "CI_L": float("nan"), "CI_U": float("nan"),
                "z_stat": float("nan"), "p_value": float("nan"),
                "reject": False, "FS": float("nan"), "RF": float("nan")}

    EY_Z1 = float(Y_c1[m1].mean())
    EY_Z0 = float(Y_c1[m0].mean())
    ET_Z1 = float(T_c1[m1].mean())
    ET_Z0 = float(T_c1[m0].mean())

    RF = EY_Z1 - EY_Z0
    FS = ET_Z1 - ET_Z0

    if abs(FS) < 1e-8:
        return {"tau": float("nan"), "SE_tau": float("nan"),
                "CI_L": float("nan"), "CI_U": float("nan"),
                "z_stat": float("nan"), "p_value": float("nan"),
                "reject": False, "FS": float(FS), "RF": float(RF)}

    tau = RF / FS

    # Delta method SE (standard unweighted)
    var_RF = (Y_c1[m1].var(ddof=1) / n1 + Y_c1[m0].var(ddof=1) / n0)
    var_FS = (ET_Z1 * (1 - ET_Z1) / n1 + ET_Z0 * (1 - ET_Z0) / n0)
    SE_tau = float(np.sqrt(max(1e-20, var_RF / FS ** 2 + RF ** 2 * var_FS / FS ** 4)))
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


# Task 2: Size / Power via M=2000 repetitions

def run_one_rep(
    rho: float,
    alpha_b: float,
    seed: int,
    N: int = N_REP,
) -> Tuple[bool, bool, float, float]:
    """
    One repetition of the H0: tau=0 test for size/power analysis.

    Returns
    -------
    (reject_ipsw, reject_raw, tau_ipsw, tau_raw)
    """
    rng = np.random.default_rng(seed)

    U     = rng.standard_normal(N)
    eps_b = rng.normal(0.0, STD_EPS_B, N)
    eps_v = rng.normal(0.0, STD_EPS_V, N)
    eps_y = rng.normal(0.0, STD_EPS_Y, N)

    v_base = 50.0 + 10.0 * U + eps_v

    Z   = rng.binomial(1, 0.5, N).astype(float)
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

    if n_C1 < 20:   # guard against degenerate samples
        return (False, False, float("nan"), float("nan"))

    Y_c1 = Y[mask_C1]
    T_c1 = T[mask_C1]
    Z_c1 = Z[mask_C1]
    W_c1 = W_ipsw[mask_C1]

    r_ipsw = _ipsw_wald(Y_c1, T_c1, Z_c1, W_c1)
    r_raw  = _unweighted_wald(Y_c1, T_c1, Z_c1)

    return (
        r_ipsw["reject"],
        r_raw["reject"],
        float(r_ipsw["tau"]) if not np.isnan(r_ipsw["tau"]) else float("nan"),
        float(r_raw["tau"])  if not np.isnan(r_raw["tau"])  else float("nan"),
    )


def run_size_power(
    cell: str,
    rho: float,
    alpha_b: float,
    m_reps: int = M_REPS,
    n_rep: int = N_REP,
) -> Dict:
    """
    Run M=2000 repetitions of the H0: tau=0 test.
    Returns rejection rates for IPSW-corrected and uncorrected C=1 Wald.
    """
    # Generate M distinct seeds from base
    rng_meta = np.random.default_rng(REP_BASE_SEED)
    rep_seeds = rng_meta.integers(0, 2**31 - 1, m_reps)

    reject_ipsw_list: List[bool] = []
    reject_raw_list:  List[bool] = []
    tau_ipsw_list:    List[float] = []
    tau_raw_list:     List[float] = []

    for m in range(m_reps):
        rej_i, rej_r, tau_i, tau_r = run_one_rep(rho, alpha_b, int(rep_seeds[m]), n_rep)
        reject_ipsw_list.append(rej_i)
        reject_raw_list.append(rej_r)
        tau_ipsw_list.append(tau_i)
        tau_raw_list.append(tau_r)

    # Rejection rates (excluding NaN)
    valid_ipsw = [r for r in reject_ipsw_list if not np.isnan(r) and isinstance(r, bool)]
    valid_raw  = [r for r in reject_raw_list  if not np.isnan(r) and isinstance(r, bool)]
    valid_ti   = [t for t in tau_ipsw_list if not np.isnan(t)]
    valid_tr   = [t for t in tau_raw_list  if not np.isnan(t)]

    n_valid_ipsw = len(valid_ipsw)
    n_valid_raw  = len(valid_raw)

    rej_rate_ipsw = float(np.mean(valid_ipsw)) if valid_ipsw else float("nan")
    rej_rate_raw  = float(np.mean(valid_raw))  if valid_raw  else float("nan")

    # SE of rejection rate (binomial SE)
    se_rej_ipsw = float(np.sqrt(rej_rate_ipsw * (1 - rej_rate_ipsw) / n_valid_ipsw)) \
                  if n_valid_ipsw > 0 and not np.isnan(rej_rate_ipsw) else float("nan")
    se_rej_raw  = float(np.sqrt(rej_rate_raw  * (1 - rej_rate_raw)  / n_valid_raw))  \
                  if n_valid_raw  > 0 and not np.isnan(rej_rate_raw)  else float("nan")

    # Mean / SD of tau estimates across reps
    tau_ipsw_mean = float(np.mean(valid_ti)) if valid_ti else float("nan")
    tau_ipsw_std  = float(np.std(valid_ti, ddof=1)) if len(valid_ti) > 1 else float("nan")
    tau_raw_mean  = float(np.mean(valid_tr)) if valid_tr else float("nan")
    tau_raw_std   = float(np.std(valid_tr, ddof=1)) if len(valid_tr) > 1 else float("nan")

    # Sign reversal diagnosis:
    # Among valid reps, fraction where tau_raw and tau_ipsw have opposite signs
    sign_rev_frac = float("nan")
    if valid_ti and valid_tr:
        both = [(ti, tr) for ti, tr in zip(valid_ti, valid_tr)
                if not np.isnan(ti) and not np.isnan(tr)]
        if both:
            n_sign_rev = sum(1 for ti, tr in both
                             if (ti * tr < 0))  # opposite signs
            sign_rev_frac = float(n_sign_rev / len(both))

    return {
        "cell"            : cell,
        "rho"             : rho,
        "alpha_b"         : alpha_b,
        "M_reps"          : m_reps,
        "N_rep"           : n_rep,
        "n_valid_ipsw"    : n_valid_ipsw,
        "n_valid_raw"     : n_valid_raw,
        "reject_rate_IPSW": rej_rate_ipsw,
        "SE_reject_IPSW"  : se_rej_ipsw,
        "reject_rate_C1_raw" : rej_rate_raw,
        "SE_reject_C1_raw"   : se_rej_raw,
        "tau_IV_IPSW_mean": tau_ipsw_mean,
        "tau_IV_IPSW_std" : tau_ipsw_std,
        "tau_IV_raw_mean" : tau_raw_mean,
        "tau_IV_raw_std"  : tau_raw_std,
        "sign_reversal_frac_IPSW_vs_raw" : sign_rev_frac,
    }


# Aggregate across seeds

def aggregate_cell(per_seed: List[Dict]) -> Dict:
    """Mean ± std (ddof=1) across 5 seeds."""

    def ms(key: str) -> Tuple[float, float]:
        vals = [r[key] for r in per_seed if not isinstance(r[key], bool)]
        return float(np.mean(vals)), float(np.std(vals, ddof=1))

    def fall(key: str) -> bool:
        return all(bool(r[key]) for r in per_seed)

    def fany(key: str) -> bool:
        return any(bool(r[key]) for r in per_seed)

    return {
        "cell"                         : per_seed[0]["cell"],
        "rho"                          : per_seed[0]["rho"],
        "alpha_b"                      : per_seed[0]["alpha_b"],
        "N_per_seed"                   : per_seed[0]["N"],
        # tau_LATE_true summary (Task 1)
        "tau_LATE_true_mean"           : ms("tau_LATE_true")[0],
        "tau_LATE_true_std"            : ms("tau_LATE_true")[1],
        "tau_LATE_all_lt0"             : fall("tau_LATE_lt0"),
        "kappa_LATE_true_mean"         : ms("kappa_LATE_true")[0],
        "kappa_LATE_true_std"          : ms("kappa_LATE_true")[1],
        "NIE_LATE_mean"                : ms("NIE_LATE")[0],
        "NIE_LATE_std"                 : ms("NIE_LATE")[1],
        "NDE_LATE_mean"                : ms("NDE_LATE")[0],
        "NDE_LATE_std"                 : ms("NDE_LATE")[1],
        "EU_complier_mean"             : ms("EU_complier")[0],
        "EU_pop_mean"                  : ms("EU_pop")[0],
        "pi_c_mean"                    : ms("pi_c")[0],
        "pi_c_std"                     : ms("pi_c")[1],
        "kappa_pop_mean"               : ms("kappa_pop")[0],
        "tau_pop_mean"                 : ms("ATE_pop")[0],
        "claim1_all_match"             : fall("claim1_kappa_eq_1_tau_D"),
        # HM bounds summary (Task 3)
        "D_L_mean"                     : ms("D_L")[0],
        "D_L_std"                      : ms("D_L")[1],
        "D_U_mean"                     : ms("D_U")[0],
        "D_U_std"                      : ms("D_U")[1],
        "D_true_mean"                  : ms("D_true")[0],
        "D_contains_all"               : fall("D_contains_D_true"),
        "kap_L_ident_mean"             : ms("kap_L_ident")[0],
        "kap_U_ident_mean"             : ms("kap_U_ident")[0],
        "ident_covers_LATE_all"        : fall("ident_covers_LATE"),
        "ident_all_lt1_all"            : fall("ident_all_lt1"),
        # IPSW Wald summary (Task 2)
        "tau_IV_IPSW_mean"             : ms("tau_IV_IPSW")[0],
        "tau_IV_IPSW_std"              : ms("tau_IV_IPSW")[1],
        "SE_tau_IPSW_mean"             : ms("SE_tau_IPSW")[0],
        "CI_L_IPSW_mean"               : ms("CI_L_IPSW")[0],
        "CI_U_IPSW_mean"               : ms("CI_U_IPSW")[0],
        "tau_IPSW_covers_true_all"     : fall("tau_IPSW_covers_true"),
        # C=1 raw (comparison)
        "tau_IV_C1_raw_mean"           : ms("tau_IV_C1_raw")[0],
        "tau_IV_C1_raw_std"            : ms("tau_IV_C1_raw")[1],
        # IM CI summary (Task 3)
        "c_IM_mean"                    : ms("c_IM")[0],
        "im_ci_L_mean"                 : ms("im_ci_L")[0],
        "im_ci_U_mean"                 : ms("im_ci_U")[0],
        "im_ci_covers_LATE_all"        : fall("im_ci_covers_LATE"),
        "im_ci_all_lt1_all"            : fall("im_ci_all_lt1"),
        # Per-seed records (embedded)
        "per_seed"                     : per_seed,
    }


# Git / env helpers

def get_git_commit() -> str:
    try:
        r = subprocess.run(
            ["git", "-C", str(BASE_DIR), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10
        )
        return r.stdout.strip() if r.returncode == 0 else "unavailable"
    except Exception:
        return "unavailable"


def make_env_log(git_hash: str, t_start: datetime) -> Dict:
    return {
        "experiment_name"  : "hc_t2_reduction",
        "date"             : t_start.isoformat(),
        "python_version"   : sys.version,
        "numpy_version"    : np.__version__,
        "scipy_version"    : scipy.__version__,
        "pandas_version"   : pd.__version__,
        "os"               : platform.platform(),
        "hostname"         : socket.gethostname(),
        "gpu"              : "none (CPU-only per DRAFT §4.3.3 rule 5)",
        "git_commit"       : git_hash,
        "seeds_main"       : SEEDS,
        "N_main"           : N_FINAL,
        "M_reps"           : M_REPS,
        "N_rep"            : N_REP,
        "alpha_level"      : ALPHA_LEVEL,
        "IV_STR"           : IV_STR,
        "ESC_REDUCTION"    : ESC_REDUCTION,
        "PHI_C"            : PHI_C,
    }


# Main

def main() -> None:
    t_start  = datetime.now()
    git_hash = get_git_commit()

    print("=" * 74)
    print("HC_T2(b)(c) Reduction Re-implementation")
    print("Algebraic identity: kappa_LATE = 1 + tau/D")
    print(f"Date      : {t_start.isoformat()}")
    print(f"Python    : {sys.version.split()[0]}")
    print(f"NumPy     : {np.__version__}  SciPy: {scipy.__version__}")
    print(f"Host      : {socket.gethostname()}  OS: {platform.system()}")
    print(f"git       : {git_hash}")
    print(f"GPU       : none (CPU-only)")
    print(f"Seeds (main): {SEEDS}  N/seed: {N_FINAL:,}")
    print(f"Size/Power: M={M_REPS} reps × N_rep={N_REP:,} each")
    print("=" * 74)

    # Tasks 1 + 3 + 2 (main estimate): twin sim + HM bounds + IPSW Wald
    all_per_seed:    List[Dict] = []
    cell_aggregates: List[Dict] = []

    print("\n" + "=" * 74)
    print("PART A: tau_LATE_true (Task 1) + HM bounds (Task 3) + IPSW Wald (Task 2)")
    print("=" * 74)

    for spec in TIER2_CELLS:
        cell, rho, alpha_b = spec["cell"], spec["rho"], spec["alpha_b"]
        print(f"\n--- Cell {cell}: rho={rho}, alpha_b={alpha_b} ---")

        per_seed: List[Dict] = []
        for seed in SEEDS:
            r = run_calibration_seed(cell, rho, alpha_b, seed)
            per_seed.append(r)
            print(
                f"  seed={seed:4d}  tau_LATE={r['tau_LATE_true']:+.6f}  "
                f"tau<0={'Y' if r['tau_LATE_lt0'] else 'N'}  "
                f"kappa_LATE={r['kappa_LATE_true']:.5f}  "
                f"claim1={'Y' if r['claim1_kappa_eq_1_tau_D'] else 'N'}"
            )
            print(
                f"          D_L={r['D_L']:.5f}  D_true={r['D_true']:.5f}  D_U={r['D_U']:.5f}  "
                f"D_bracket={'Y' if r['D_contains_D_true'] else 'N'}  "
                f"ident=[{r['kap_L_ident']:.3f},{r['kap_U_ident']:.3f}]  "
                f"all<1={'Y' if r['ident_all_lt1'] else 'N'}  "
                f"cov={'Y' if r['ident_covers_LATE'] else 'N'}"
            )
            print(
                f"          tau_IPSW={r['tau_IV_IPSW']:+.6f}±{r['SE_tau_IPSW']:.2e}  "
                f"CI=[{r['CI_L_IPSW']:+.5f},{r['CI_U_IPSW']:+.5f}]  "
                f"covers_true={'Y' if r['tau_IPSW_covers_true'] else 'N'}  "
                f"tau_C1_raw={r['tau_IV_C1_raw']:+.6f}"
            )
            print(
                f"          IM_CI=[{r['im_ci_L']:.3f},{r['im_ci_U']:.3f}]  "
                f"im_cov={'Y' if r['im_ci_covers_LATE'] else 'N'}  "
                f"im_all<1={'Y' if r['im_ci_all_lt1'] else 'N'}  "
                f"c_IM={r['c_IM']:.4f}"
            )

        all_per_seed.extend(per_seed)
        agg = aggregate_cell(per_seed)
        cell_aggregates.append(agg)

        print(f"\n  [Aggregate {cell}]")
        print(f"    tau_LATE_true = {agg['tau_LATE_true_mean']:+.6f} ± {agg['tau_LATE_true_std']:.2e}"
              f"  all_lt0={agg['tau_LATE_all_lt0']}")
        print(f"    kappa_LATE    = {agg['kappa_LATE_true_mean']:.5f} ± {agg['kappa_LATE_true_std']:.2e}")
        print(f"    D_L           = {agg['D_L_mean']:.6f} ± {agg['D_L_std']:.2e}")
        print(f"    D_true        = {agg['D_true_mean']:.6f}")
        print(f"    D_U           = {agg['D_U_mean']:.6f} ± {agg['D_U_std']:.2e}")
        print(f"    D bracket ALL = {agg['D_contains_all']}")
        print(f"    ident region  = [{agg['kap_L_ident_mean']:.4f}, {agg['kap_U_ident_mean']:.4f}]")
        print(f"    ident<1  ALL  = {agg['ident_all_lt1_all']}  "
              f"ident cov ALL = {agg['ident_covers_LATE_all']}")
        print(f"    tau_IV_IPSW   = {agg['tau_IV_IPSW_mean']:+.6f} ± {agg['tau_IV_IPSW_std']:.2e}")
        print(f"    tau_IV_C1_raw = {agg['tau_IV_C1_raw_mean']:+.6f} ± {agg['tau_IV_C1_raw_std']:.2e}")
        print(f"    IPSW cov ALL  = {agg['tau_IPSW_covers_true_all']}")
        print(f"    IM_CI = [{agg['im_ci_L_mean']:.4f}, {agg['im_ci_U_mean']:.4f}]  "
              f"covers ALL={agg['im_ci_covers_LATE_all']}  all<1 ALL={agg['im_ci_all_lt1_all']}")
        print(f"    claim1 ALL    = {agg['claim1_all_match']}")

    # Task 2: Size / Power via M=2000 repetitions
    print("\n" + "=" * 74)
    print(f"PART B: HC_T2(b) Size/Power (M={M_REPS} reps, N_rep={N_REP:,})")
    print("=" * 74)

    sp_results: List[Dict] = []
    for spec in TIER2_CELLS:
        cell, rho, alpha_b = spec["cell"], spec["rho"], spec["alpha_b"]
        label = "SIZE " if "null" in cell else "POWER"
        print(f"\n  Running {label} [{cell}]  rho={rho}  alpha_b={alpha_b}  ...", flush=True)
        sp = run_size_power(cell, rho, alpha_b, M_REPS, N_REP)
        sp_results.append(sp)
        print(
            f"    reject_IPSW = {sp['reject_rate_IPSW']:.4f} ± {sp['SE_reject_IPSW']:.4f}"
            f"  reject_raw = {sp['reject_rate_C1_raw']:.4f} ± {sp['SE_reject_C1_raw']:.4f}"
        )
        print(
            f"    tau_IV_IPSW_mean = {sp['tau_IV_IPSW_mean']:+.5f} ± {sp['tau_IV_IPSW_std']:.3e}"
            f"  tau_IV_raw_mean = {sp['tau_IV_raw_mean']:+.5f} ± {sp['tau_IV_raw_std']:.3e}"
        )
        if sp["sign_reversal_frac_IPSW_vs_raw"] is not None and not np.isnan(sp["sign_reversal_frac_IPSW_vs_raw"]):
            print(
                f"    sign reversal (tau_IPSW vs tau_raw opposite sign) = "
                f"{sp['sign_reversal_frac_IPSW_vs_raw']:.4f}"
            )

    # HC_T2(b)(c) PASS / FAIL summary
    print("\n" + "=" * 74)
    print("HC_T2 PASS/FAIL SUMMARY")
    print("=" * 74)

    # Extract relevant aggregates (T2-a/b/c/d, excluding null)
    tier2_aggs = [a for a in cell_aggregates if a["cell"] != "T2-null"]
    null_agg   = next(a for a in cell_aggregates if a["cell"] == "T2-null")
    null_sp    = next(s for s in sp_results      if s["cell"] == "T2-null")
    tier2_sps  = [s for s in sp_results          if s["cell"] != "T2-null"]

    # HC_T2(a) [already confirmed, just re-confirm via tau sign]
    hc_t2a = all(a["tau_LATE_all_lt0"] for a in tier2_aggs)

    # HC_T2(b) criteria:
    #   1. ALL 5 seeds: IPSW CI covers tau_LATE_true
    #   2. SIZE (null): reject_rate_IPSW <= 2*alpha (= 0.10)
    #   3. POWER (T2-a~d): reject_rate_IPSW >= 0.80 (at least for T2-a/b/c; T2-d also checked)
    hc_t2b_cov  = all(a["tau_IPSW_covers_true_all"] for a in tier2_aggs)
    hc_t2b_size = bool(
        not np.isnan(null_sp["reject_rate_IPSW"])
        and null_sp["reject_rate_IPSW"] <= 2 * ALPHA_LEVEL
    )
    hc_t2b_power = all(
        not np.isnan(s["reject_rate_IPSW"]) and s["reject_rate_IPSW"] >= 0.80
        for s in tier2_sps
    )
    hc_t2b = hc_t2b_cov and hc_t2b_size and hc_t2b_power

    # HC_T2(c) criteria:
    #   1. ALL 5 seeds: D bracket contains D_true
    #   2. ALL 5 seeds: ident_covers_LATE  (or IM_CI_covers_LATE)
    #   3. ALL 5 seeds: ident_all_lt1 (region fully < 1)
    hc_t2c_D    = all(a["D_contains_all"] for a in tier2_aggs)
    hc_t2c_cov  = all(a["im_ci_covers_LATE_all"] for a in tier2_aggs)
    hc_t2c_lt1  = all(a["im_ci_all_lt1_all"] for a in tier2_aggs)
    hc_t2c = hc_t2c_D and hc_t2c_cov and hc_t2c_lt1

    # Collider sign-reversal prevention
    # Show: IPSW size is controlled; raw C=1 has inflated type-I
    ipsw_size_ok = hc_t2b_size
    raw_inflated = bool(
        not np.isnan(null_sp["reject_rate_C1_raw"])
        and null_sp["reject_rate_C1_raw"] > 2 * ALPHA_LEVEL
    )

    for lbl, passed in [
        ("HC_T2(a) tau_LATE_true < 0 for ALL T2 cells", hc_t2a),
        ("HC_T2(b) [i] IPSW CI covers tau_LATE_true (all seeds)", hc_t2b_cov),
        ("HC_T2(b) [ii] SIZE ≤ 2*alpha (IPSW, null cell)", hc_t2b_size),
        ("HC_T2(b) [iii] POWER ≥ 0.80 (IPSW, T2-a~d)", hc_t2b_power),
        ("HC_T2(b) OVERALL", hc_t2b),
        ("HC_T2(c) [i] D bounds bracket D_true (all seeds)", hc_t2c_D),
        ("HC_T2(c) [ii] IM-CI covers kappa_LATE_true (all seeds)", hc_t2c_cov),
        ("HC_T2(c) [iii] IM-CI fully < 1 for tau<0 (all seeds)", hc_t2c_lt1),
        ("HC_T2(c) OVERALL", hc_t2c),
        ("Collider correction [IPSW size controlled]", ipsw_size_ok),
        ("Collider bias [C=1 raw inflates type-I]", raw_inflated),
    ]:
        flag = "PASS" if passed else "FAIL"
        print(f"  [{flag}]  {lbl}")

    # Detailed result tables
    print("\n" + "=" * 74)
    print("TABLE 1: tau_LATE_true (per cell, mean ± std across 5 seeds)")
    print(f"{'Cell':<8} {'rho':>4} {'alpha_b':>8} {'tau_LATE':>12} {'±std':>9} "
          f"{'all<0':>6} {'kappa_LATE':>11} {'EU_comp':>8}")
    print("-" * 74)
    for a in cell_aggregates:
        print(
            f"{a['cell']:<8} {a['rho']:>4.1f} {a['alpha_b']:>8.4f} "
            f"{a['tau_LATE_true_mean']:>+12.6f} {a['tau_LATE_true_std']:>9.2e} "
            f"{'YES' if a['tau_LATE_all_lt0'] else 'NO':>6} "
            f"{a['kappa_LATE_true_mean']:>11.5f} "
            f"{a['EU_complier_mean']:>+8.4f}"
        )

    print("\n" + "=" * 74)
    print("TABLE 2: HC_T2(b) tau point estimates (IPSW vs C=1 raw)")
    print(f"{'Cell':<8} {'tau_true':>9} {'tau_IPSW':>9} {'CI':>23} {'cov':>4} "
          f"{'tau_raw':>9} {'bias':>8}")
    print("-" * 74)
    for a in cell_aggregates:
        tau_t  = a["tau_LATE_true_mean"]
        tau_iv = a["tau_IV_IPSW_mean"]
        tau_r  = a["tau_IV_C1_raw_mean"]
        cl     = a["CI_L_IPSW_mean"]
        cu     = a["CI_U_IPSW_mean"]
        print(
            f"{a['cell']:<8} {tau_t:>+9.5f} {tau_iv:>+9.5f} "
            f"[{cl:>+9.5f},{cu:>+9.5f}] "
            f"{'Y' if a['tau_IPSW_covers_true_all'] else 'N':>4} "
            f"{tau_r:>+9.5f} {tau_r-tau_t:>+8.5f}"
        )

    print("\n" + "=" * 74)
    print(f"TABLE 3: HC_T2(b) Size/Power (M={M_REPS}, N_rep={N_REP})")
    print(f"{'Cell':<8} {'role':>6} {'rej_IPSW':>9} {'±SE':>7} {'rej_raw':>9} {'±SE':>7} "
          f"{'sign_rev':>9}")
    print("-" * 74)
    for s in sp_results:
        role = "SIZE" if "null" in s["cell"] else "POWER"
        sr   = s["sign_reversal_frac_IPSW_vs_raw"]
        print(
            f"{s['cell']:<8} {role:>6} "
            f"{s['reject_rate_IPSW']:>9.4f} {s['SE_reject_IPSW']:>7.4f} "
            f"{s['reject_rate_C1_raw']:>9.4f} {s['SE_reject_C1_raw']:>7.4f} "
            f"{sr:>9.4f}" if not np.isnan(sr) else
            f"{s['cell']:<8} {role:>6} "
            f"{s['reject_rate_IPSW']:>9.4f} {s['SE_reject_IPSW']:>7.4f} "
            f"{s['reject_rate_C1_raw']:>9.4f} {s['SE_reject_C1_raw']:>7.4f} "
            f"{'nan':>9}"
        )

    print("\n" + "=" * 74)
    print("TABLE 4: HC_T2(c) kappa_LATE identification region (mean across seeds)")
    print(f"{'Cell':<8} {'kappa_true':>10} {'D_L':>8} {'D_true':>8} {'D_U':>8} "
          f"{'kap_L':>7} {'kap_U':>7} {'IM_L':>7} {'IM_U':>7} {'<1':>4} {'cov':>4}")
    print("-" * 90)
    for a in cell_aggregates:
        print(
            f"{a['cell']:<8} {a['kappa_LATE_true_mean']:>10.5f} "
            f"{a['D_L_mean']:>8.5f} {a['D_true_mean']:>8.5f} {a['D_U_mean']:>8.5f} "
            f"{a['kap_L_ident_mean']:>7.3f} {a['kap_U_ident_mean']:>7.3f} "
            f"{a['im_ci_L_mean']:>7.3f} {a['im_ci_U_mean']:>7.3f} "
            f"{'Y' if a['im_ci_all_lt1_all'] else 'N':>4} "
            f"{'Y' if a['im_ci_covers_LATE_all'] else 'N':>4}"
        )

    # Save per-seed CSV
    df = pd.DataFrame(all_per_seed)
    for col in df.select_dtypes("bool").columns:
        df[col] = df[col].astype(int)
    csv_path = RESULTS_DIR / "hc_t2_reduction_per_cell.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nPer-seed CSV saved: {csv_path}")

    # Save size/power CSV
    df_sp = pd.DataFrame(sp_results)
    sp_csv_path = RESULTS_DIR / "hc_t2_reduction_size_power.csv"
    df_sp.to_csv(sp_csv_path, index=False)
    print(f"Size/Power CSV saved: {sp_csv_path}")

    # Save aggregate JSON
    t_end   = datetime.now()
    elapsed = (t_end - t_start).total_seconds()

    env_log = make_env_log(git_hash, t_start)

    # Build summary dict
    draft_summary = {}
    for a in cell_aggregates:
        c = a["cell"]
        draft_summary[c] = {
            "tau_LATE_true_mean"     : a["tau_LATE_true_mean"],
            "tau_LATE_true_std"      : a["tau_LATE_true_std"],
            "tau_LATE_all_lt0"       : a["tau_LATE_all_lt0"],
            "kappa_LATE_true_mean"   : a["kappa_LATE_true_mean"],
            "kappa_LATE_true_std"    : a["kappa_LATE_true_std"],
            "NIE_LATE_mean"          : a["NIE_LATE_mean"],
            "NDE_LATE_mean"          : a["NDE_LATE_mean"],
            "EU_complier_mean"       : a["EU_complier_mean"],
            "pi_c_mean"              : a["pi_c_mean"],
            "tau_IV_IPSW_mean"       : a["tau_IV_IPSW_mean"],
            "tau_IV_IPSW_std"        : a["tau_IV_IPSW_std"],
            "CI_L_IPSW_mean"         : a["CI_L_IPSW_mean"],
            "CI_U_IPSW_mean"         : a["CI_U_IPSW_mean"],
            "tau_IPSW_covers_all"    : a["tau_IPSW_covers_true_all"],
            "tau_IV_C1_raw_mean"     : a["tau_IV_C1_raw_mean"],
            "D_L_mean"               : a["D_L_mean"],
            "D_true_mean"            : a["D_true_mean"],
            "D_U_mean"               : a["D_U_mean"],
            "kap_L_ident_mean"       : a["kap_L_ident_mean"],
            "kap_U_ident_mean"       : a["kap_U_ident_mean"],
            "im_ci_L_mean"           : a["im_ci_L_mean"],
            "im_ci_U_mean"           : a["im_ci_U_mean"],
            "im_ci_covers_LATE_all"  : a["im_ci_covers_LATE_all"],
            "im_ci_all_lt1_all"      : a["im_ci_all_lt1_all"],
            "claim1_all_match"       : a["claim1_all_match"],
        }

    size_power_summary = {}
    for s in sp_results:
        c = s["cell"]
        size_power_summary[c] = {
            "reject_rate_IPSW"   : s["reject_rate_IPSW"],
            "SE_reject_IPSW"     : s["SE_reject_IPSW"],
            "reject_rate_C1_raw" : s["reject_rate_C1_raw"],
            "SE_reject_C1_raw"   : s["SE_reject_C1_raw"],
            "tau_IV_IPSW_mean"   : s["tau_IV_IPSW_mean"],
            "tau_IV_raw_mean"    : s["tau_IV_raw_mean"],
            "sign_rev_frac"      : s["sign_reversal_frac_IPSW_vs_raw"],
        }

    hc_results = {
        "HC_T2_a": {
            "description"       : "tau_LATE_true < 0 for all Tier-2 cells",
            "pass"              : bool(hc_t2a),
        },
        "HC_T2_b": {
            "description"       : "IPSW IV Wald covers tau_LATE_true AND size/power controlled",
            "pass_coverage"     : bool(hc_t2b_cov),
            "pass_size"         : bool(hc_t2b_size),
            "pass_power"        : bool(hc_t2b_power),
            "pass_overall"      : bool(hc_t2b),
            "size_IPSW"         : float(null_sp["reject_rate_IPSW"]),
            "size_C1_raw"       : float(null_sp["reject_rate_C1_raw"]),
            "ipsw_controls_size": bool(ipsw_size_ok),
            "raw_inflates_size" : bool(raw_inflated),
        },
        "HC_T2_c": {
            "description"       : "IM-CI covers kappa_LATE_true AND fully < 1",
            "pass_D_bracket"    : bool(hc_t2c_D),
            "pass_IM_coverage"  : bool(hc_t2c_cov),
            "pass_all_lt1"      : bool(hc_t2c_lt1),
            "pass_overall"      : bool(hc_t2c),
        },
    }

    output_json = {
        "computation_type"     : "hc_t2_reduction",
        "description"          : (
            "HC_T2(b)(c) re-implementation based on kappa_LATE=1+tau/D. "
            "Task 1: tau_LATE_true via twin sim. "
            "Task 2: IPSW-corrected Wald point test (H0:tau=0) + M=2000 size/power. "
            "Task 3: HM trimming bounds + IM two-layer CI for kappa_LATE. "
            "N=10^6 x 5 seeds per cell. CPU only. "
            "Ref: preregistration_DRAFT_2026-07-10.md §5.1 HC_T2(b)(c)."
        ),
        "preregistration"      : str(BASE_DIR / "preregistration_DRAFT_2026-07-10.md"),
        "theory_ref"           : "paper Section 4.3 (kappa_LATE reduction)",
        "frozen_scm_params"    : {
            "LOGIT_P0"          : LOGIT_P0,
            "gamma_b"           : GAMMA_B,
            "std_eps_b"         : STD_EPS_B,
            "std_eps_v"         : STD_EPS_V,
            "beta_u"            : BETA_U,
            "std_eps_y"         : STD_EPS_Y,
            "ESC_reduction"     : ESC_REDUCTION,
            "speed_mult"        : SPEED_MULT,
            "eta_0"             : ETA_0,
            "eta_T"             : ETA_T,
            "eta_U"             : ETA_U,
            "eta_B"             : ETA_B,
            "IV_STR"            : IV_STR,
            "PHI_C"             : PHI_C,
            "G4_alpha_b"        : G4_ALPHA_B,
        },
        "hc_results"           : hc_results,
        "per_cell_summary"     : draft_summary,
        "size_power_summary"   : size_power_summary,
        "cell_aggregates"      : cell_aggregates,
        "elapsed_seconds"      : elapsed,
        "env_log"              : env_log,
    }

    json_path = RESULTS_DIR / "hc_t2_reduction_aggregate.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output_json, f, indent=2, ensure_ascii=False, default=str)
    print(f"Aggregate JSON saved: {json_path}")

    print(f"\nElapsed: {elapsed:.1f} s")
    print(f"git    : {git_hash}")
    print(f"\nOutput files:")
    print(f"  {csv_path}")
    print(f"  {sp_csv_path}")
    print(f"  {json_path}")
    print("=" * 74)

    # Summary dict (copy-paste ready)
    print("\n" + "=" * 74)
    print("Section 5.1 numbers (HC_T2(b)(c) reduction results)")
    print("=" * 74)
    for cell in ["T2-a", "T2-b", "T2-c", "T2-d"]:
        d  = draft_summary[cell]
        sp = size_power_summary[cell]
        print(f"\n-- {cell} --")
        print(f"  tau_LATE_true        = {d['tau_LATE_true_mean']:+.6f} ± {d['tau_LATE_true_std']:.2e}  [all<0: {d['tau_LATE_all_lt0']}]")
        print(f"  kappa_LATE_true      = {d['kappa_LATE_true_mean']:.5f} ± {d['kappa_LATE_true_std']:.2e}")
        print(f"  tau_IV_IPSW          = {d['tau_IV_IPSW_mean']:+.6f} ± {d['tau_IV_IPSW_std']:.2e}")
        print(f"  95% CI tau_IV_IPSW   = [{d['CI_L_IPSW_mean']:+.6f}, {d['CI_U_IPSW_mean']:+.6f}]  covers={d['tau_IPSW_covers_all']}")
        print(f"  tau_IV_C1_raw        = {d['tau_IV_C1_raw_mean']:+.6f}  (bias={d['tau_IV_C1_raw_mean']-d['tau_LATE_true_mean']:+.6f})")
        print(f"  H0 reject_rate IPSW  = {sp['reject_rate_IPSW']:.4f} ± {sp['SE_reject_IPSW']:.4f}  (POWER)")
        print(f"  H0 reject_rate raw   = {sp['reject_rate_C1_raw']:.4f} ± {sp['SE_reject_C1_raw']:.4f}  (POWER)")
        print(f"  D_L                  = {d['D_L_mean']:.6f}")
        print(f"  D_true               = {d['D_true_mean']:.6f}")
        print(f"  D_U                  = {d['D_U_mean']:.6f}")
        print(f"  kappa ident region   = [{d['kap_L_ident_mean']:.4f}, {d['kap_U_ident_mean']:.4f}]")
        print(f"  IM CI kappa          = [{d['im_ci_L_mean']:.4f}, {d['im_ci_U_mean']:.4f}]  covers={d['im_ci_covers_LATE_all']}  all<1={d['im_ci_all_lt1_all']}")

    print(f"\n-- T2-null (SIZE test) --")
    d_null  = draft_summary["T2-null"]
    sp_null = size_power_summary["T2-null"]
    print(f"  tau_LATE_true (null)  = {d_null['tau_LATE_true_mean']:+.6f} ± {d_null['tau_LATE_true_std']:.2e}")
    print(f"  H0 reject_rate IPSW   = {sp_null['reject_rate_IPSW']:.4f} ± {sp_null['SE_reject_IPSW']:.4f}  (SIZE = should be ≤ 0.10)")
    print(f"  H0 reject_rate raw    = {sp_null['reject_rate_C1_raw']:.4f} ± {sp_null['SE_reject_C1_raw']:.4f}  (SIZE collider-biased)")
    print(f"  sign_rev_frac         = {sp_null['sign_rev_frac']:.4f}")
    print()
    print(f"HC_T2(b) OVERALL: {'PASS' if hc_t2b else 'FAIL'}")
    print(f"HC_T2(c) OVERALL: {'PASS' if hc_t2c else 'FAIL'}")
    print(f"git hash: {git_hash}")
    print("=" * 74)


if __name__ == "__main__":
    main()
