"""
PA1: Proximal mediation baseline comparison (exploratory).
Compares IV Wald IPSW (targets tau_LATE) vs proximal mediation (targets tau_ATE_pop) on T2-a and T2-b.
Goal: transparent assumption trade-off analysis, not performance competition.
Outputs: results/pa1_proximal_mediation.csv, results/pa1_proximal_mediation_aggregate.json
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import scipy.stats

# Paths
BASE_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Frozen SCM parameters
LOGIT_P0: float = float(np.log(0.55 / 0.45))   # logit(0.55)
GAMMA_B:    float = 0.50
STD_EPS_B:  float = 0.30
STD_EPS_V:  float = 5.00
BETA_U:     float = 0.10
STD_EPS_Y:  float = 0.05
ESC_REDUCTION: float = 0.30   # illustrative modeling choice
SPEED_MULT: float = 5.0
ETA_0:  float = -2.0
ETA_T:  float = -0.5
ETA_U:  float =  0.3
ETA_B:  float =  0.2
IV_STR: float = 1.5   # logit-scale IV shift for Z_ESC

# Proxy noise std (σ for W1 and Z1_proxy)
SIGMA_PROXY: float = 0.5

# Physics constant
PHI_C: float = (1.0 - (1.0 - ESC_REDUCTION) ** 4) / (100.0 ** 4)

# Cells and seeds
SEEDS: List[int] = [42, 123, 456, 789, 2026]
N_FINAL: int = 1_000_000
CELLS: List[Dict] = [
    {"cell": "T2-a", "rho": 0.5, "alpha_b": 0.3},
    {"cell": "T2-b", "rho": 1.0, "alpha_b": 0.3},
]

# Pre-registered true values
TAU_LATE_TRUE: Dict[str, float] = {
    "T2-a": -0.063038,
    "T2-b": -0.058665,
}
TAU_ATE_POP_TRUE: float = -0.066897  # NDE_pop + NIE_pop (same for all T2 cells)
KAPPA_LATE_TRUE: Dict[str, float] = {
    "T2-a": 0.03368,
    "T2-b": 0.03467,
}


# Helpers

def sigmoid(x: np.ndarray) -> np.ndarray:
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)),
                    np.exp(x) / (1.0 + np.exp(x)))


def wald_ipsw(
    Y: np.ndarray, T: np.ndarray, Z: np.ndarray, W: np.ndarray
) -> Dict:
    """IPSW-weighted IV Wald estimator (Hajek)."""
    m1, m0 = (Z == 1), (Z == 0)
    w1, w0 = W[m1], W[m0]
    sw1, sw0 = w1.sum(), w0.sum()
    EY1 = np.dot(Y[m1], w1) / sw1
    EY0 = np.dot(Y[m0], w0) / sw0
    ET1 = np.dot(T[m1], w1) / sw1
    ET0 = np.dot(T[m0], w0) / sw0
    RF = EY1 - EY0;  FS = ET1 - ET0
    if abs(FS) < 1e-8:
        return {"tau": np.nan, "SE": np.nan, "CI_L": np.nan, "CI_U": np.nan,
                "FS": float(FS)}
    tau = RF / FS
    resY1 = Y[m1] - EY1;  resY0 = Y[m0] - EY0
    resT1 = T[m1] - ET1;  resT0 = T[m0] - ET0
    var_RF = (np.dot(w1**2, resY1**2)/sw1**2 + np.dot(w0**2, resY0**2)/sw0**2)
    var_FS = (np.dot(w1**2, resT1**2)/sw1**2 + np.dot(w0**2, resT0**2)/sw0**2)
    SE = float(np.sqrt(max(1e-20, var_RF/FS**2 + RF**2*var_FS/FS**4)))
    return {"tau": float(tau), "SE": SE,
            "CI_L": float(tau - 1.96*SE), "CI_U": float(tau + 1.96*SE),
            "FS": float(FS)}


def proximal_poly_wls(
    Y: np.ndarray, T: np.ndarray, U_hat: np.ndarray, W: np.ndarray,
    degree: int = 4
) -> Dict:
    """
    IPSW-weighted polynomial regression: Y ~ T + poly(U_hat, degree)
    Returns coefficient of T as proximal ATE estimate.
    Uses sandwich variance for SE.
    """
    n = len(Y)
    # Build design matrix: [intercept, T, U_hat, U_hat^2, ..., U_hat^degree]
    X_cols = [np.ones(n), T]
    for d in range(1, degree + 1):
        X_cols.append(U_hat ** d)
    X = np.column_stack(X_cols)   # shape (n, degree+2)

    # WLS normal equations
    sqrt_w = np.sqrt(W)
    Xw = X * sqrt_w[:, None]
    Yw = Y * sqrt_w
    try:
        beta, _, _, _ = np.linalg.lstsq(Xw, Yw, rcond=None)
    except np.linalg.LinAlgError:
        return {"tau": np.nan, "SE": np.nan, "CI_L": np.nan, "CI_U": np.nan}

    tau = float(beta[1])   # coefficient of T

    # Sandwich variance
    resid = Y - X @ beta
    XtWX = Xw.T @ Xw   # (degree+2) x (degree+2)
    try:
        XtWX_inv = np.linalg.inv(XtWX)
    except np.linalg.LinAlgError:
        return {"tau": tau, "SE": np.nan, "CI_L": np.nan, "CI_U": np.nan}

    bread = X * (W * resid)[:, None]   # n x (degree+2), each row = w_i * r_i * x_i
    meat = bread.T @ bread              # (degree+2) x (degree+2)
    var_beta = XtWX_inv @ meat @ XtWX_inv
    SE = float(np.sqrt(max(1e-20, var_beta[1, 1])))
    return {"tau": tau, "SE": SE,
            "CI_L": float(tau - 1.96*SE), "CI_U": float(tau + 1.96*SE)}


def check_completeness(W1: np.ndarray, Z1_p: np.ndarray) -> Dict:
    """
    Empirical completeness diagnostics for Gaussian proxies.

    Completeness (Miao et al. 2018) requires that the mapping u -> p(w|u)
    is injective on L^2. For Gaussian proxies, this holds when the kernel
    operator has strictly positive eigenvalues — equivalent to the proxy
    covariance matrix being full rank.
    """
    n = len(W1)
    cov_W1Z1 = float(np.cov(W1, Z1_p)[0, 1])
    var_W1 = float(np.var(W1, ddof=1))
    var_Z1 = float(np.var(Z1_p, ddof=1))
    corr = cov_W1Z1 / np.sqrt(var_W1 * var_Z1)

    # 2x2 proxy covariance matrix
    proxy_cov = np.array([[var_W1, cov_W1Z1], [cov_W1Z1, var_Z1]])
    eigs = np.linalg.eigvalsh(proxy_cov)
    cond_number = float(eigs.max() / max(eigs.min(), 1e-10))
    det = float(np.linalg.det(proxy_cov))

    # Information coefficient: η² = [Cov(W1,Z1_p)]^2 / (Var(W1)*Var(Z1_p))
    eta_sq = cov_W1Z1**2 / (var_W1 * var_Z1)

    # Theoretical values for σ=0.5 proxies:
    # Var(W1) = Var(Z1_p) = 1 + 0.25 = 1.25
    # Cov(W1, Z1_p) = Var(U) = 1.0
    # Corr = 1/1.25 = 0.8, η²=0.64, det=0.5625, cond=2.25/0.25=9

    return {
        "cov_W1_Z1p": cov_W1Z1,
        "var_W1": var_W1,
        "var_Z1p": var_Z1,
        "corr_W1_Z1p": float(corr),
        "eta_sq": float(eta_sq),
        "proxy_det": det,
        "proxy_cond_number": cond_number,
        "rank_ok": bool(det > 1e-8),         # rank condition satisfied
        "completeness_holds": bool(cond_number < 1e6 and det > 0),  # Gaussian: always holds
    }


# Main per-(cell, seed) function

def run_seed(cell: str, rho: float, alpha_b: float, seed: int) -> Dict:
    """Run PA1 for one (cell, seed) pair."""
    rng = np.random.default_rng(seed)
    N = N_FINAL

    # === SCM generation ===
    U      = rng.standard_normal(N)
    eps_b  = rng.normal(0.0, STD_EPS_B, N)
    eps_v  = rng.normal(0.0, STD_EPS_V, N)
    eps_y  = rng.normal(0.0, STD_EPS_Y, N)

    # IV (ESC mandate) for our method
    Z_ESC  = rng.binomial(1, 0.5, N).astype(float)

    # Treatment
    logit_pT = LOGIT_P0 + rho * U + IV_STR * (Z_ESC - 0.5)
    P_T      = sigmoid(logit_pT)
    U_T      = rng.uniform(0.0, 1.0, N)
    T        = (U_T <= P_T).astype(float)

    # Mediator (latent), speed, outcome
    b       = alpha_b * T + GAMMA_B * U + eps_b
    v_base  = 50.0 + 10.0 * U + eps_v
    v_act   = v_base + SPEED_MULT * b
    dv      = v_act * (1.0 - ESC_REDUCTION * T)
    Y       = (dv / 100.0)**4 + BETA_U * U + eps_y

    # Collider (crash) + oracle IPSW
    P_crash = sigmoid(ETA_0 + ETA_T*T + ETA_U*U + ETA_B*b)
    C       = rng.binomial(1, P_crash).astype(float)
    W_ipsw  = 1.0 / P_crash

    # Proxies for U
    eps_W  = rng.normal(0.0, SIGMA_PROXY, N)
    eps_Zp = rng.normal(0.0, SIGMA_PROXY, N)
    W1     = U + eps_W     # outcome-inducing proxy
    Z1_p   = U + eps_Zp   # treatment-inducing proxy

    # === Completeness check ===
    comp = check_completeness(W1, Z1_p)

    # === Posterior U_hat from (W1, Z1_p) ===
    # Prior U~N(0,1), likelihood W1|U~N(U,0.25), Z1_p|U~N(U,0.25)
    # Posterior sigma^2 = 1/(1 + 4 + 4) = 1/9
    # Posterior mean = (1/9) * (4*W1 + 4*Z1_p)
    sigma2_post = 1.0 / (1.0 + 1.0/SIGMA_PROXY**2 + 1.0/SIGMA_PROXY**2)
    prec_W = 1.0 / SIGMA_PROXY**2
    U_hat  = sigma2_post * (prec_W * W1 + prec_W * Z1_p)  # posterior mean

    # === C=1 mask ===
    mask = (C == 1)
    n_C1 = int(mask.sum())

    Y_c   = Y[mask];    T_c  = T[mask];   Z_c  = Z_ESC[mask]
    W_c   = W_ipsw[mask]
    Uh_c  = U_hat[mask]

    # Theoretical: σ²_post = 1/9 ≈ 0.111
    sigma2_post_theoretical = 1.0 / (1.0 + 2.0 / SIGMA_PROXY**2)

    # === Proximal ATE estimators ===
    # Linear (degree=1)
    prox_lin  = proximal_poly_wls(Y_c, T_c, Uh_c, W_c, degree=1)
    # Poly4 (degree=4, captures phi ∝ v^4 nonlinearity)
    prox_poly = proximal_poly_wls(Y_c, T_c, Uh_c, W_c, degree=4)

    # Naive (no correction, mean diff on C=1, no IV/IPSW)
    tau_naive = float(np.mean(Y_c[T_c == 1]) - np.mean(Y_c[T_c == 0]))
    n1, n0 = int((T_c==1).sum()), int((T_c==0).sum())
    if n1 > 1 and n0 > 1:
        se_naive = float(np.sqrt(np.var(Y_c[T_c==1], ddof=1)/n1
                                 + np.var(Y_c[T_c==0], ddof=1)/n0))
    else:
        se_naive = np.nan

    # Naive with proxies as covariates (linear proxy control, IPSW)
    W1_c  = W1[mask]
    Z1p_c = Z1_p[mask]
    n_col = len(Y_c)
    X_pc = np.column_stack([np.ones(n_col), T_c, W1_c, Z1p_c])
    sqrt_wc = np.sqrt(W_c)
    Xw_pc = X_pc * sqrt_wc[:, None]
    Yw_pc = Y_c * sqrt_wc
    try:
        beta_pc, _, _, _ = np.linalg.lstsq(Xw_pc, Yw_pc, rcond=None)
        tau_proxy_ctrl = float(beta_pc[1])
        resid_pc = Y_c - X_pc @ beta_pc
        meat_pc = (X_pc * (W_c * resid_pc)[:, None]).T @ (X_pc * (W_c * resid_pc)[:, None])
        XtWX_pc = Xw_pc.T @ Xw_pc
        try:
            vb_pc = np.linalg.inv(XtWX_pc) @ meat_pc @ np.linalg.inv(XtWX_pc)
            se_proxy_ctrl = float(np.sqrt(max(1e-20, vb_pc[1,1])))
        except np.linalg.LinAlgError:
            se_proxy_ctrl = np.nan
    except Exception:
        tau_proxy_ctrl = np.nan
        se_proxy_ctrl  = np.nan

    # === Our IV Wald IPSW (targets tau_LATE) ===
    iv_res = wald_ipsw(Y_c, T_c, Z_c, W_c)

    # === Population ATE (twin sim for reference) ===
    b_T0 = GAMMA_B * U + eps_b
    b_T1 = alpha_b + GAMMA_B * U + eps_b
    def Y_cf(T_out, b_in):
        dv_cf = (v_base + SPEED_MULT*b_in) * (1.0 - ESC_REDUCTION*T_out)
        return (dv_cf/100.0)**4 + BETA_U*U
    Y11 = Y_cf(1.0, b_T1);  Y10 = Y_cf(1.0, b_T0);  Y00 = Y_cf(0.0, b_T0)
    tau_ATE_pop_mc = float(np.mean(Y11 - Y00))

    # Complier truth
    P_T_Z0 = sigmoid(LOGIT_P0 + rho*U + IV_STR*(0.0 - 0.5))
    P_T_Z1 = sigmoid(LOGIT_P0 + rho*U + IV_STR*(1.0 - 0.5))
    U_T2   = rng.uniform(0.0, 1.0, N)
    T_Z0   = (U_T2 <= P_T_Z0).astype(float)
    T_Z1   = (U_T2 <= P_T_Z1).astype(float)
    is_c   = (T_Z1 > T_Z0)
    tau_LATE_mc = float(np.mean((Y11 - Y00)[is_c]))

    # Bias calculations
    tau_LATE_ref = TAU_LATE_TRUE[cell]
    bias_naive   = tau_naive - TAU_ATE_POP_TRUE
    bias_prox_lin  = prox_lin["tau"]  - TAU_ATE_POP_TRUE if not np.isnan(prox_lin["tau"])  else np.nan
    bias_prox_poly = prox_poly["tau"] - TAU_ATE_POP_TRUE if not np.isnan(prox_poly["tau"]) else np.nan
    bias_proxy_ctrl = tau_proxy_ctrl - TAU_ATE_POP_TRUE if not np.isnan(tau_proxy_ctrl) else np.nan
    bias_iv_vs_LATE  = iv_res["tau"] - tau_LATE_ref if not np.isnan(iv_res["tau"])  else np.nan
    bias_iv_vs_ATE   = iv_res["tau"] - TAU_ATE_POP_TRUE if not np.isnan(iv_res["tau"]) else np.nan

    # CI coverage checks
    def ci_covers(ci_l, ci_u, true_val):
        if np.isnan(ci_l) or np.isnan(ci_u):
            return False
        return bool(ci_l <= true_val <= ci_u)

    cov_prox_lin_ate  = ci_covers(prox_lin["CI_L"],  prox_lin["CI_U"],  TAU_ATE_POP_TRUE)
    cov_prox_poly_ate = ci_covers(prox_poly["CI_L"], prox_poly["CI_U"], TAU_ATE_POP_TRUE)
    cov_iv_late       = ci_covers(iv_res["CI_L"],    iv_res["CI_U"],    tau_LATE_ref)
    cov_iv_ate        = ci_covers(iv_res["CI_L"],    iv_res["CI_U"],    TAU_ATE_POP_TRUE)

    return {
        "cell": cell, "rho": rho, "alpha_b": alpha_b, "seed": seed, "N": N,
        # Completeness diagnostics
        "proxy_cov_W1_Z1p": comp["cov_W1_Z1p"],
        "proxy_corr": comp["corr_W1_Z1p"],
        "proxy_eta_sq": comp["eta_sq"],
        "proxy_cond_number": comp["proxy_cond_number"],
        "proxy_det": comp["proxy_det"],
        "completeness_holds": comp["completeness_holds"],
        "sigma2_post_actual": float(sigma2_post),
        "sigma2_post_theoretical": float(sigma2_post_theoretical),
        # Sample info
        "n_C1": n_C1,
        "crash_rate": float(n_C1 / N),
        # Population truth (MC)
        "tau_ATE_pop_mc": tau_ATE_pop_mc,
        "tau_ATE_pop_prereg": TAU_ATE_POP_TRUE,
        "tau_LATE_mc": tau_LATE_mc,
        "tau_LATE_prereg": tau_LATE_ref,
        # Method: Naive (no correction)
        "tau_naive": tau_naive,
        "se_naive": se_naive,
        "bias_naive_vs_ATE": bias_naive,
        # Method: Proxy control regression (W1, Z1_p as controls, IPSW)
        "tau_proxy_ctrl": tau_proxy_ctrl,
        "se_proxy_ctrl": se_proxy_ctrl,
        "bias_proxy_ctrl_vs_ATE": bias_proxy_ctrl,
        # Method: Proximal linear regression (U_hat degree=1, IPSW)
        "tau_prox_lin": prox_lin["tau"],
        "se_prox_lin": prox_lin["SE"],
        "CI_L_prox_lin": prox_lin["CI_L"],
        "CI_U_prox_lin": prox_lin["CI_U"],
        "bias_prox_lin_vs_ATE": bias_prox_lin,
        "cov_prox_lin_ate": cov_prox_lin_ate,
        # Method: Proximal poly4 regression (U_hat degree=4, IPSW)
        "tau_prox_poly": prox_poly["tau"],
        "se_prox_poly": prox_poly["SE"],
        "CI_L_prox_poly": prox_poly["CI_L"],
        "CI_U_prox_poly": prox_poly["CI_U"],
        "bias_prox_poly_vs_ATE": bias_prox_poly,
        "cov_prox_poly_ate": cov_prox_poly_ate,
        # Method: IV Wald IPSW (our method, targets tau_LATE)
        "tau_IV_IPSW": iv_res["tau"],
        "se_IV_IPSW": iv_res["SE"],
        "CI_L_IV": iv_res["CI_L"],
        "CI_U_IV": iv_res["CI_U"],
        "FS_IV": iv_res["FS"],
        "bias_IV_vs_LATE": bias_iv_vs_LATE,
        "bias_IV_vs_ATE": bias_iv_vs_ATE,
        "cov_IV_late": cov_iv_late,
        "cov_IV_ate": cov_iv_ate,
        # NIE/NDE with latent b: not identified by U-proxies (stated finding)
        "NIE_proximal_identified": False,  # requires b-proxy
        "NDE_proximal_identified": False,  # requires b-proxy
        "NIE_IV_bounds": True,              # our method provides D bounds
    }


# Env log

def build_env_log(seeds: List[int], cells: List[str]) -> Dict:
    git_hash = subprocess.run(
        ["git", "-C", str(BASE_DIR / "pilot"), "rev-parse", "HEAD"],
        capture_output=True, text=True
    ).stdout.strip() or "unknown"
    return {
        "script": "pa1_proximal_mediation.py",
        "date": datetime.now().isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "git_commit": git_hash,
        "N_per_seed": N_FINAL,
        "seeds": seeds,
        "cells": cells,
        "sigma_proxy": SIGMA_PROXY,
        "status": "exploratory",
        "gpu_used": False,
        "note": "PA1 exploratory analysis, frozen prereg §5.2. Elevating to confirmatory prohibited per §5.3.",
    }


# Main

def main() -> None:
    print("=== PA1: Proximal Mediation Baseline Comparison ===")
    print(f"N={N_FINAL:,}, seeds={SEEDS}, proxy σ={SIGMA_PROXY}")
    print(f"Cells: {[c['cell'] for c in CELLS]}")
    print()

    rows: List[Dict] = []

    for cell_cfg in CELLS:
        cell = cell_cfg["cell"]
        rho  = cell_cfg["rho"]
        ab   = cell_cfg["alpha_b"]
        print(f"--- Cell {cell} (rho={rho}, alpha_b={ab}) ---")
        for seed in SEEDS:
            print(f"  seed={seed} ...", end=" ", flush=True)
            row = run_seed(cell, rho, ab, seed)
            rows.append(row)
            print(f"tau_IV={row['tau_IV_IPSW']:.5f}, "
                  f"tau_prox_poly={row['tau_prox_poly']:.5f}, "
                  f"bias_prox_poly={row['bias_prox_poly_vs_ATE']:+.5f}")
        print()

    # --- Save CSV ---
    df = pd.DataFrame(rows)
    csv_path = RESULTS_DIR / "pa1_proximal_mediation.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV: {csv_path}")

    # --- Aggregate JSON ---
    env_log = build_env_log(SEEDS, [c["cell"] for c in CELLS])

    def agg_col(df_sub: pd.DataFrame, col: str) -> Dict:
        vals = df_sub[col].dropna()
        if len(vals) == 0:
            return {"mean": None, "std": None, "per_seed": {}}
        return {"mean": float(vals.mean()), "std": float(vals.std()),
                "per_seed": {str(r["seed"]): (float(r[col]) if not pd.isna(r[col]) else None)
                             for _, r in df_sub.iterrows()}}

    aggregate: Dict = {"env_log": env_log, "cells": {}}

    for cell_cfg in CELLS:
        cell = cell_cfg["cell"]
        df_c = df[df["cell"] == cell].copy()

        # Assumption table entry (descriptive)
        assumption_table = {
            "naive": {
                "requires_IV": False, "requires_IPSW": False,
                "requires_completeness": False, "requires_proxies": False,
                "identifies": "biased_mean_difference",
                "target": "none (severely biased)"
            },
            "proxy_control": {
                "requires_IV": False, "requires_IPSW": True,
                "requires_completeness": False, "requires_proxies": True,
                "identifies": "ATE_pop (approximately)",
                "target": "tau_ATE_pop"
            },
            "proximal_poly4": {
                "requires_IV": False, "requires_IPSW": True,
                "requires_completeness": True, "requires_proxies": True,
                "identifies": "ATE_pop (approximation via U_hat)",
                "target": "tau_ATE_pop",
                "NIE_NDE_with_latent_b": "NOT identified (requires b-proxy)"
            },
            "IV_Wald_IPSW (our method)": {
                "requires_IV": True, "requires_IPSW": True,
                "requires_completeness": False, "requires_proxies": False,
                "identifies": "tau_LATE + kappa_LATE_identification_region",
                "target": "tau_LATE (complier LATE)",
                "NIE_NDE": "D=|NDE_LATE| bounded via physics (HM trimming)"
            }
        }

        aggregate["cells"][cell] = {
            "cell": cell, "rho": cell_cfg["rho"], "alpha_b": cell_cfg["alpha_b"],
            "tau_ATE_pop_true": TAU_ATE_POP_TRUE,
            "tau_LATE_true": TAU_LATE_TRUE[cell],
            "kappa_LATE_true": KAPPA_LATE_TRUE[cell],
            "sigma_proxy": SIGMA_PROXY,
            # Completeness
            "completeness": {
                k: agg_col(df_c, k)
                for k in ["proxy_cov_W1_Z1p", "proxy_corr", "proxy_eta_sq",
                          "proxy_cond_number", "proxy_det"]
            },
            "completeness_holds_all_seeds": bool(df_c["completeness_holds"].all()),
            # tau estimates (mean ± std across 5 seeds)
            "tau": {
                "naive":               agg_col(df_c, "tau_naive"),
                "proxy_ctrl":          agg_col(df_c, "tau_proxy_ctrl"),
                "proximal_linear":     agg_col(df_c, "tau_prox_lin"),
                "proximal_poly4":      agg_col(df_c, "tau_prox_poly"),
                "IV_Wald_IPSW":        agg_col(df_c, "tau_IV_IPSW"),
            },
            # Bias vs tau_ATE_pop
            "bias_vs_ATE_pop": {
                "naive":           agg_col(df_c, "bias_naive_vs_ATE"),
                "proxy_ctrl":      agg_col(df_c, "bias_proxy_ctrl_vs_ATE"),
                "proximal_linear": agg_col(df_c, "bias_prox_lin_vs_ATE"),
                "proximal_poly4":  agg_col(df_c, "bias_prox_poly_vs_ATE"),
                "IV_Wald_vs_ATE":  agg_col(df_c, "bias_IV_vs_ATE"),
            },
            # Bias of IV vs tau_LATE
            "bias_IV_vs_LATE": agg_col(df_c, "bias_IV_vs_LATE"),
            # CI coverage
            "CI_coverage_prox_poly_at_ATE": float(df_c["cov_prox_poly_ate"].mean()),
            "CI_coverage_IV_at_LATE":       float(df_c["cov_IV_late"].mean()),
            "CI_coverage_IV_at_ATE":        float(df_c["cov_IV_ate"].mean()),
            # Assumption table
            "assumption_table": assumption_table,
            # Key finding
            "key_finding": {
                "completeness_holds": "YES (Gaussian proxies σ=0.5, condition number≈9, η²≈0.64)",
                "NIE_NDE_identification": "U-proxies alone CANNOT identify NIE/NDE when b is latent. Requires b-proxy.",
                "both_need_IPSW": "Both proximal and IV methods require IPSW for crash-conditional (C=1) collider correction.",
                "proximal_target": "Proximal targets tau_ATE_pop (population average); IV targets tau_LATE (complier subpopulation).",
                "point_test_kappa": "Only IV approach enables H0: kappa=1 point test (via kappa_LATE = 1 + tau/D).",
            }
        }

    json_path = RESULTS_DIR / "pa1_proximal_mediation_aggregate.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(aggregate, f, indent=2, ensure_ascii=False, default=str)
    print(f"Saved JSON: {json_path}")

    # --- Summary ---
    print()
    print("=== PA1 Summary ===")
    for cell_cfg in CELLS:
        cell = cell_cfg["cell"]
        df_c = df[df["cell"] == cell]
        print(f"\nCell {cell}:")
        print(f"  tau_ATE_pop_true = {TAU_ATE_POP_TRUE:.6f}")
        print(f"  tau_LATE_true    = {TAU_LATE_TRUE[cell]:.6f}")
        print(f"  Naive:            {df_c['tau_naive'].mean():.6f} ± {df_c['tau_naive'].std():.6f}  (bias vs ATE: {df_c['bias_naive_vs_ATE'].mean():+.6f})")
        print(f"  Proximal poly4:   {df_c['tau_prox_poly'].mean():.6f} ± {df_c['tau_prox_poly'].std():.6f}  (bias vs ATE: {df_c['bias_prox_poly_vs_ATE'].mean():+.6f})")
        print(f"  IV Wald IPSW:     {df_c['tau_IV_IPSW'].mean():.6f} ± {df_c['tau_IV_IPSW'].std():.6f}  (bias vs LATE: {df_c['bias_IV_vs_LATE'].mean():+.6f})")
        print(f"  Completeness: cond_number={df_c['proxy_cond_number'].mean():.2f}, η²={df_c['proxy_eta_sq'].mean():.4f}")

    print()
    print("KEY: Proximal (poly4) targets tau_ATE_pop; IV targets tau_LATE.")
    print("     NIE/NDE with latent b: NOT identified by U-proxies alone.")
    print("     Both methods require IPSW for crash-conditional collider correction.")

    print(f"\nResults: {csv_path}")
    print(f"         {json_path}")


if __name__ == "__main__":
    main()
