"""
Exp2: Power comparison — IV Wald point test vs Imbens-Manski bounds for H0: kappa_LATE=1 (exploratory).
For wide identification regions, the IM critical value c_IM < 1.96, making the bounds test
more powerful than the two-sided Wald test for tau < 0 alternatives (at the cost of being one-sided).
Outputs: results/exp2_bounds_vs_pointtest.csv, results/exp2_bounds_vs_pointtest_aggregate.json
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
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import scipy.stats
import scipy.optimize

# Paths & constants
BASE_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Frozen SCM parameters
LOGIT_P0: float = float(np.log(0.55 / 0.45))
GAMMA_B:  float = 0.50
STD_EPS_B: float = 0.30
STD_EPS_V: float = 5.00
BETA_U:   float = 0.10
STD_EPS_Y: float = 0.05
ESC_REDUCTION: float = 0.30
SPEED_MULT: float = 5.0
ETA_0:  float = -2.0
ETA_T:  float = -0.5
ETA_U:  float =  0.3
ETA_B:  float =  0.2
IV_STR: float = 1.5
PHI_C: float = (1.0 - (1.0 - ESC_REDUCTION) ** 4) / (100.0 ** 4)
G4_ALPHA_B: float = 4.99720863

SEEDS: List[int] = [42, 123, 456, 789, 2026]
N_FINAL: int = 1_000_000

TIER2_CELLS: List[Dict] = [
    {"cell": "T2-a",    "rho": 0.5, "alpha_b": 0.3},
    {"cell": "T2-b",    "rho": 1.0, "alpha_b": 0.3},
    {"cell": "T2-c",    "rho": 0.5, "alpha_b": 3.972},
    {"cell": "T2-d",    "rho": 1.0, "alpha_b": 3.972},
    {"cell": "T2-null", "rho": 0.5, "alpha_b": G4_ALPHA_B},
]

# Pre-registered tau_LATE_true and D bounds (from hc_t2_reduction results)
PREREG_VALS: Dict[str, Dict] = {
    "T2-a":    {"tau_LATE_true": -0.063038, "D_L": 0.01445, "D_U": 0.14352, "pi_c": 0.338},
    "T2-b":    {"tau_LATE_true": -0.058665, "D_L": 0.01277, "D_U": 0.15210, "pi_c": 0.300},
    "T2-c":    {"tau_LATE_true": -0.018299, "D_L": 0.01445, "D_U": 0.14352, "pi_c": 0.338},
    "T2-d":    {"tau_LATE_true": -0.015322, "D_L": 0.01277, "D_U": 0.15210, "pi_c": 0.300},
    "T2-null": {"tau_LATE_true": +0.001848, "D_L": 0.01445, "D_U": 0.14352, "pi_c": 0.338},
}

N_REP_LIST: List[int] = [10_000, 20_000, 50_000, 100_000]
M_REPS: int = 500
BASE_SEED: int = 99999   # single base seed for M-rep simulation
ALPHA: float = 0.05


# Helpers

def sigmoid(x: np.ndarray) -> np.ndarray:
    return np.where(x >= 0, 1.0/(1.0+np.exp(-x)), np.exp(x)/(1.0+np.exp(x)))


def imbens_manski_cval(width: float, sigma: float) -> float:
    """Imbens-Manski (2004) critical value c_n s.t. Phi(c_n+w/σ)-Phi(-c_n)=0.95."""
    if sigma < 1e-12:
        return 1.96
    w_over_s = width / sigma
    if w_over_s > 200:
        return 1.645   # approaches one-sided 95% critical value
    def eq(c: float) -> float:
        return scipy.stats.norm.cdf(c + w_over_s) - scipy.stats.norm.cdf(-c) - 0.95
    try:
        return float(scipy.optimize.brentq(eq, 0.001, 10.0))
    except ValueError:
        return 1.96


def wald_ipsw_fast(
    Y: np.ndarray, T: np.ndarray, Z: np.ndarray, W: np.ndarray
) -> Tuple[float, float]:
    """Fast IV Wald IPSW. Returns (tau, SE)."""
    m1, m0 = (Z == 1), (Z == 0)
    w1, w0 = W[m1], W[m0]
    sw1, sw0 = w1.sum(), w0.sum()
    if sw1 < 1e-8 or sw0 < 1e-8:
        return np.nan, np.nan
    EY1 = np.dot(Y[m1], w1)/sw1;  EY0 = np.dot(Y[m0], w0)/sw0
    ET1 = np.dot(T[m1], w1)/sw1;  ET0 = np.dot(T[m0], w0)/sw0
    FS = ET1 - ET0
    if abs(FS) < 1e-8:
        return np.nan, np.nan
    RF = EY1 - EY0
    tau = RF / FS
    resY1 = Y[m1]-EY1; resY0 = Y[m0]-EY0
    resT1 = T[m1]-ET1; resT0 = T[m0]-ET0
    vRF = (np.dot(w1**2,resY1**2)/sw1**2 + np.dot(w0**2,resY0**2)/sw0**2)
    vFS = (np.dot(w1**2,resT1**2)/sw1**2 + np.dot(w0**2,resT0**2)/sw0**2)
    SE = float(np.sqrt(max(1e-20, vRF/FS**2 + RF**2*vFS/FS**4)))
    return float(tau), SE


def generate_single_rep(
    rng: np.random.Generator,
    N: int,
    rho: float,
    alpha_b: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate one SCM realization, return (Y_c1, T_c1, Z_c1, W_c1)."""
    U     = rng.standard_normal(N)
    eps_b = rng.normal(0.0, STD_EPS_B, N)
    eps_v = rng.normal(0.0, STD_EPS_V, N)
    eps_y = rng.normal(0.0, STD_EPS_Y, N)
    Z     = rng.binomial(1, 0.5, N).astype(float)
    logit_pT = LOGIT_P0 + rho*U + IV_STR*(Z-0.5)
    P_T  = sigmoid(logit_pT)
    T    = (rng.uniform(0.0,1.0,N) <= P_T).astype(float)
    b    = alpha_b*T + GAMMA_B*U + eps_b
    vb   = 50.0 + 10.0*U + eps_v
    dv   = (vb + SPEED_MULT*b) * (1.0 - ESC_REDUCTION*T)
    Y    = (dv/100.0)**4 + BETA_U*U + eps_y
    Pcr  = sigmoid(ETA_0 + ETA_T*T + ETA_U*U + ETA_B*b)
    C    = rng.binomial(1, Pcr).astype(float)
    W    = 1.0 / Pcr
    mask = (C == 1)
    return Y[mask], T[mask], Z[mask], W[mask]


# Population D bounds: compute from N=10^6 (stable estimate)

def compute_D_bounds_population(rho: float, alpha_b: float, pi_c: float, seed: int = 42) -> Tuple[float, float]:
    """Compute HM bounds D_L, D_U from population (N=1M)."""
    rng = np.random.default_rng(seed)
    N = N_FINAL
    U     = rng.standard_normal(N)
    eps_b = rng.normal(0.0, STD_EPS_B, N)
    eps_v = rng.normal(0.0, STD_EPS_V, N)
    v_base = 50.0 + 10.0*U + eps_v
    b_T0   = GAMMA_B*U + eps_b   # b when T=0
    S0     = v_base + SPEED_MULT*b_T0
    W      = S0**4
    Ws     = np.sort(W)
    k      = max(1, int(round(pi_c * N)))
    D_L    = float(PHI_C * Ws[:k].mean())
    D_U    = float(PHI_C * Ws[-k:].mean())
    return D_L, D_U


# Single cell M-rep simulation at given N_rep

def run_mreps_single(
    rho: float,
    alpha_b: float,
    D_L: float,
    D_U: float,
    N_rep: int,
    M: int,
    base_seed: int,
) -> Dict:
    """
    Run M repetitions of IV Wald test at N_rep.
    Returns rejection rates for Method A (point test) and Method B (bounds-only IM CI).
    """
    rng = np.random.default_rng(base_seed)
    rep_seeds = rng.integers(0, 2**31 - 1, size=M)

    reject_A: List[bool] = []
    reject_B: List[bool] = []
    c_IM_vals: List[float] = []
    tau_vals: List[float] = []

    for rseed in rep_seeds:
        rng_rep = np.random.default_rng(int(rseed))
        Yc, Tc, Zc, Wc = generate_single_rep(rng_rep, N_rep, rho, alpha_b)
        if len(Yc) < 10:
            continue
        tau, SE = wald_ipsw_fast(Yc, Tc, Zc, Wc)
        if np.isnan(tau) or np.isnan(SE) or SE < 1e-12:
            continue

        # Method A: two-sided point test H0: tau=0
        z_stat = tau / SE
        rej_A = bool(abs(z_stat) > scipy.stats.norm.ppf(1.0 - ALPHA/2))
        reject_A.append(rej_A)
        tau_vals.append(tau)

        # Method B: IM CI upper bound excludes 1
        # kap_U = 1 + tau/D_U  (closest to 1 when tau < 0)
        # SE_kap_U = SE / D_U
        if D_U > 1e-10:
            kap_U_ident = 1.0 + tau / D_U
            kap_L_ident = 1.0 + tau / D_L if D_L > 1e-10 else -1e6
            width = max(0.0, kap_U_ident - kap_L_ident)   # may be negative if tau>0
            SE_kap_U = SE / D_U
            c_IM = imbens_manski_cval(width, SE_kap_U)
            c_IM_vals.append(c_IM)
            # Reject if kap_U + c_IM*SE_kap_U < 1
            # i.e., tau + c_IM * SE < 0
            rej_B = bool(tau + c_IM * SE < 0)
        else:
            c_IM_vals.append(1.96)
            rej_B = False
        reject_B.append(rej_B)

    n_valid = len(reject_A)
    if n_valid == 0:
        return {"n_valid": 0, "reject_A": np.nan, "reject_B": np.nan,
                "c_IM_mean": np.nan, "tau_mean": np.nan}

    return {
        "n_valid": n_valid,
        "reject_A_rate": float(np.mean(reject_A)),
        "reject_B_rate": float(np.mean(reject_B)),
        "reject_A_se": float(np.std(reject_A) / np.sqrt(n_valid)),
        "reject_B_se": float(np.std(reject_B) / np.sqrt(n_valid)),
        "c_IM_mean": float(np.mean(c_IM_vals)),
        "c_IM_std":  float(np.std(c_IM_vals)),
        "tau_mean":  float(np.mean(tau_vals)),
        "tau_std":   float(np.std(tau_vals)),
    }


# Env log

def build_env_log() -> Dict:
    git_hash = subprocess.run(
        ["git", "-C", str(BASE_DIR/"pilot"), "rev-parse", "HEAD"],
        capture_output=True, text=True).stdout.strip() or "unknown"
    return {
        "script": "exp2_bounds_vs_pointtest.py",
        "date": datetime.now().isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "git_commit": git_hash,
        "N_final": N_FINAL,
        "N_rep_list": N_REP_LIST,
        "M_reps": M_REPS,
        "base_seed": BASE_SEED,
        "alpha": ALPHA,
        "status": "exploratory",
        "gpu_used": False,
    }


# Main

def main() -> None:
    print("=== Exp2: Bounds-only vs Point-test Power Comparison ===")
    print(f"Cells: {[c['cell'] for c in TIER2_CELLS]}")
    print(f"N_rep: {N_REP_LIST}, M={M_REPS}, base_seed={BASE_SEED}")
    print()

    rows: List[Dict] = []

    for cell_cfg in TIER2_CELLS:
        cell     = cell_cfg["cell"]
        rho      = cell_cfg["rho"]
        alpha_b  = cell_cfg["alpha_b"]
        vals     = PREREG_VALS[cell]
        tau_true = vals["tau_LATE_true"]
        pi_c     = vals["pi_c"]
        role     = "SIZE" if cell == "T2-null" else "POWER"

        print(f"--- Cell {cell} ({role}, rho={rho}, tau_LATE={tau_true:.5f}) ---")
        print("  Computing population D bounds ...", end=" ")
        D_L, D_U = compute_D_bounds_population(rho, alpha_b, pi_c, seed=42)
        print(f"D_L={D_L:.5f}, D_U={D_U:.5f}")
        print(f"  Prereg D_L={vals['D_L']:.5f}, D_U={vals['D_U']:.5f}")

        for N_rep in N_REP_LIST:
            print(f"  N_rep={N_rep:>7,}: running M={M_REPS} reps ...", end=" ", flush=True)
            res = run_mreps_single(rho, alpha_b, D_L, D_U, N_rep, M_REPS, BASE_SEED)
            print(f"rejA={res.get('reject_A_rate', float('nan')):.3f}, "
                  f"rejB={res.get('reject_B_rate', float('nan')):.3f}, "
                  f"c_IM={res.get('c_IM_mean', float('nan')):.3f}")

            rows.append({
                "cell": cell, "rho": rho, "alpha_b": alpha_b,
                "tau_LATE_true": tau_true, "role": role,
                "D_L_computed": D_L, "D_U_computed": D_U,
                "D_L_prereg": vals["D_L"], "D_U_prereg": vals["D_U"],
                "N_rep": N_rep, "M": M_REPS,
                **res
            })
        print()

    # Save CSV
    df = pd.DataFrame(rows)
    csv_path = RESULTS_DIR / "exp2_bounds_vs_pointtest.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV: {csv_path}")

    # Aggregate JSON
    env_log = build_env_log()
    agg: Dict = {"env_log": env_log, "cells": {}}

    for cell_cfg in TIER2_CELLS:
        cell = cell_cfg["cell"]
        df_c = df[df["cell"] == cell].copy()
        cell_agg: Dict = {
            "tau_LATE_true": PREREG_VALS[cell]["tau_LATE_true"],
            "role": "SIZE" if cell == "T2-null" else "POWER",
            "power_curves": {},
        }
        for N_rep in N_REP_LIST:
            row = df_c[df_c["N_rep"] == N_rep].iloc[0]
            cell_agg["power_curves"][str(N_rep)] = {
                "method_A_point_test":   {"reject_rate": float(row["reject_A_rate"]), "SE": float(row["reject_A_se"])},
                "method_B_bounds_only":  {"reject_rate": float(row["reject_B_rate"]), "SE": float(row["reject_B_se"])},
                "c_IM_mean": float(row["c_IM_mean"]),
            }
        agg["cells"][cell] = cell_agg

    # Interpretive summary
    agg["interpretation"] = {
        "c_IM_convergence": "c_IM approaches 1.645 (one-sided critical value) as width/sigma increases (wide identification region).",
        "method_B_effective_alpha": "Method B effectively one-sided (rejects only when tau < 0). Size at null cell checks this.",
        "method_A_alpha": "Method A two-sided at 5%; lower power for directional alternatives tau < 0.",
        "key_finding": (
            "For wide identification regions (D_U/D_L >> 1), c_IM < 1.96, so Method B "
            "is less conservative than two-sided Method A. Both have correct 5% size "
            "for their respective alternatives. For fair comparison: Method A one-sided "
            "(z < -1.645) would match Method B asymptotically."
        ),
    }

    json_path = RESULTS_DIR / "exp2_bounds_vs_pointtest_aggregate.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2, ensure_ascii=False, default=str)
    print(f"Saved JSON: {json_path}")

    # Print power table
    print()
    print("=== Exp2 Power Table ===")
    print(f"{'Cell':<10} {'N_rep':>8} {'Role':<6} {'Method_A':>9} {'Method_B':>9} {'c_IM':>6}")
    print("-" * 55)
    for _, row in df.iterrows():
        print(f"{row['cell']:<10} {int(row['N_rep']):>8,} {row['role']:<6} "
              f"{row['reject_A_rate']:>9.3f} {row['reject_B_rate']:>9.3f} "
              f"{row['c_IM_mean']:>6.3f}")
    print(f"\nResults: {csv_path}")
    print(f"         {json_path}")


if __name__ == "__main__":
    main()
