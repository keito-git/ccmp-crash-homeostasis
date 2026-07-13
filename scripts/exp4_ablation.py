"""
Exp4: Component ablation study toggling physics (P), IV (I), and IPSW (W) independently (exploratory).
Tests all 8 (P,I,W) combinations on T2-a, T2-b, and T2-null for tau bias and kappa identification width.
P affects only kappa bounds (not tau), I removes confounding bias, W controls collider-induced size inflation.
Outputs: results/exp4_ablation.csv, results/exp4_ablation_aggregate.json
"""

from __future__ import annotations

import itertools
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
import scipy.stats

# Paths & constants
BASE_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

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
N_REP:   int = 10_000
M_REPS:  int = 500
BASE_SEED: int = 11111
ALPHA:   float = 0.05

CELLS_POWER: List[Dict] = [
    {"cell": "T2-a", "rho": 0.5, "alpha_b": 0.3},
    {"cell": "T2-b", "rho": 1.0, "alpha_b": 0.3},
]
CELL_NULL: Dict = {"cell": "T2-null", "rho": 0.5, "alpha_b": G4_ALPHA_B}

# Pre-registered truth
TAU_LATE_TRUE: Dict[str, float] = {
    "T2-a":    -0.063038,
    "T2-b":    -0.058665,
    "T2-null": +0.001848,
}
D_PREREG: Dict[str, Dict] = {
    "T2-a":    {"D_L": 0.01445, "D_U": 0.14352, "pi_c": 0.338},
    "T2-b":    {"D_L": 0.01277, "D_U": 0.15210, "pi_c": 0.300},
    "T2-null": {"D_L": 0.01445, "D_U": 0.14352, "pi_c": 0.338},
}

# All 8 ablation combos
COMBOS: List[Tuple[bool, bool, bool]] = list(itertools.product([False, True], repeat=3))
# (use_physics, use_iv, use_ipsw)


# Helpers

def sigmoid(x: np.ndarray) -> np.ndarray:
    return np.where(x >= 0, 1.0/(1.0+np.exp(-x)), np.exp(x)/(1.0+np.exp(x)))


def combo_label(use_physics: bool, use_iv: bool, use_ipsw: bool) -> str:
    p = "P" if use_physics else "p"
    i = "I" if use_iv else "i"
    w = "W" if use_ipsw else "w"
    return f"{p}{i}{w}"


def tau_estimate(
    Yc: np.ndarray, Tc: np.ndarray, Zc: np.ndarray, Wc: np.ndarray,
    use_iv: bool, use_ipsw: bool
) -> Tuple[float, float]:
    """
    Estimate tau from C=1 data given combo flags.
    Returns (tau_hat, SE).
    """
    W = Wc if use_ipsw else np.ones_like(Wc)

    if use_iv:
        # IV Wald (IPSW or unweighted)
        m1, m0 = (Zc == 1), (Zc == 0)
        w1, w0 = W[m1], W[m0]
        sw1, sw0 = w1.sum(), w0.sum()
        if sw1 < 1e-8 or sw0 < 1e-8:
            return np.nan, np.nan
        EY1 = np.dot(Yc[m1],w1)/sw1; EY0 = np.dot(Yc[m0],w0)/sw0
        ET1 = np.dot(Tc[m1],w1)/sw1; ET0 = np.dot(Tc[m0],w0)/sw0
        FS = ET1 - ET0
        if abs(FS) < 1e-8:
            return np.nan, np.nan
        RF = EY1 - EY0
        tau = RF / FS
        resY1=Yc[m1]-EY1; resY0=Yc[m0]-EY0
        resT1=Tc[m1]-ET1; resT0=Tc[m0]-ET0
        vRF=(np.dot(w1**2,resY1**2)/sw1**2+np.dot(w0**2,resY0**2)/sw0**2)
        vFS=(np.dot(w1**2,resT1**2)/sw1**2+np.dot(w0**2,resT0**2)/sw0**2)
        SE = float(np.sqrt(max(1e-20, vRF/FS**2 + RF**2*vFS/FS**4)))
        return float(tau), SE
    else:
        # Naive weighted mean difference (no IV)
        m1, m0 = (Tc == 1), (Tc == 0)
        w1, w0 = W[m1], W[m0]
        sw1, sw0 = w1.sum(), w0.sum()
        if sw1 < 1e-8 or sw0 < 1e-8:
            return np.nan, np.nan
        EY1 = np.dot(Yc[m1],w1)/sw1
        EY0 = np.dot(Yc[m0],w0)/sw0
        tau = float(EY1 - EY0)
        resY1=Yc[m1]-EY1; resY0=Yc[m0]-EY0
        vY = (np.dot(w1**2,resY1**2)/sw1**2 + np.dot(w0**2,resY0**2)/sw0**2)
        SE = float(np.sqrt(max(1e-20, vY)))
        return tau, SE


# Main calibration run (N=1M x 5 seeds): tau bias, CI coverage

def run_calibration_seed(
    cell: str, rho: float, alpha_b: float, seed: int,
    use_physics: bool, use_iv: bool, use_ipsw: bool
) -> Dict:
    rng = np.random.default_rng(seed)
    N   = N_FINAL
    U   = rng.standard_normal(N)
    eps_b=rng.normal(0.0,STD_EPS_B,N)
    eps_v=rng.normal(0.0,STD_EPS_V,N)
    eps_y=rng.normal(0.0,STD_EPS_Y,N)
    Z    = rng.binomial(1,0.5,N).astype(float)
    logit_pT = LOGIT_P0 + rho*U + IV_STR*(Z-0.5)
    P_T  = sigmoid(logit_pT)
    T    = (rng.uniform(0.0,1.0,N) <= P_T).astype(float)
    b    = alpha_b*T + GAMMA_B*U + eps_b
    vb   = 50.0 + 10.0*U + eps_v
    dv   = (vb + SPEED_MULT*b)*(1.0-ESC_REDUCTION*T)
    Y    = (dv/100.0)**4 + BETA_U*U + eps_y
    Pcr  = sigmoid(ETA_0+ETA_T*T+ETA_U*U+ETA_B*b)
    C    = rng.binomial(1,Pcr).astype(float)
    W    = 1.0/Pcr
    mask = (C==1)
    n_C1 = int(mask.sum())
    Yc,Tc,Zc,Wc = Y[mask],T[mask],Z[mask],W[mask]

    tau, SE = tau_estimate(Yc,Tc,Zc,Wc,use_iv,use_ipsw)
    tau_true = TAU_LATE_TRUE.get(cell, np.nan)
    bias = float(tau - tau_true) if not np.isnan(tau) else np.nan
    CI_L = float(tau - 1.96*SE) if not np.isnan(SE) else np.nan
    CI_U = float(tau + 1.96*SE) if not np.isnan(SE) else np.nan
    ci_covers = bool(CI_L <= tau_true <= CI_U) if not np.isnan(CI_L) else False

    # Physics: D bounds (independent of IV/IPSW choice)
    D_L_val = D_U_val = kappa_width = np.nan
    if use_physics:
        d = D_PREREG.get(cell, {})
        D_L_val = d.get("D_L", np.nan)
        D_U_val = d.get("D_U", np.nan)
        # Width of kappa identification region
        if not np.isnan(tau) and not np.isnan(D_L_val) and D_L_val > 1e-10:
            kap_L = 1.0 + tau/D_L_val
            kap_U = 1.0 + tau/D_U_val
            kappa_width = float(abs(kap_U - kap_L))
        else:
            kappa_width = np.nan

    return {
        "cell": cell, "rho": rho, "alpha_b": alpha_b, "seed": seed,
        "use_physics": use_physics, "use_iv": use_iv, "use_ipsw": use_ipsw,
        "combo": combo_label(use_physics,use_iv,use_ipsw),
        "n_C1": n_C1,
        "tau_hat": float(tau) if not np.isnan(tau) else np.nan,
        "SE": float(SE) if not np.isnan(SE) else np.nan,
        "CI_L": CI_L, "CI_U": CI_U,
        "tau_true": tau_true,
        "bias": bias,
        "ci_covers_true": ci_covers,
        "D_L": D_L_val, "D_U": D_U_val,
        "kappa_ident_width": kappa_width,
    }


# M-rep size/power at N_rep=10k

def run_mreps_ablation(
    rho: float, alpha_b: float,
    use_iv: bool, use_ipsw: bool,
    N_rep: int, M: int, base_seed: int
) -> Dict:
    """M reps for size/power (physics doesn't affect tau test)."""
    rng = np.random.default_rng(base_seed)
    rep_seeds = rng.integers(0, 2**31-1, size=M)

    reject: List[bool] = []
    tau_list: List[float] = []

    for rseed in rep_seeds:
        r = np.random.default_rng(int(rseed))
        N = N_rep
        U   = r.standard_normal(N)
        eps_b=r.normal(0.0,STD_EPS_B,N)
        eps_v=r.normal(0.0,STD_EPS_V,N)
        eps_y=r.normal(0.0,STD_EPS_Y,N)
        Z   = r.binomial(1,0.5,N).astype(float)
        logit_pT=LOGIT_P0+rho*U+IV_STR*(Z-0.5)
        P_T=sigmoid(logit_pT)
        T  =(r.uniform(0.0,1.0,N)<=P_T).astype(float)
        b  =alpha_b*T+GAMMA_B*U+eps_b
        vb =50.0+10.0*U+eps_v
        dv =(vb+SPEED_MULT*b)*(1.0-ESC_REDUCTION*T)
        Y  =(dv/100.0)**4+BETA_U*U+eps_y
        Pcr=sigmoid(ETA_0+ETA_T*T+ETA_U*U+ETA_B*b)
        C  =r.binomial(1,Pcr).astype(float)
        W  =1.0/Pcr
        mask=(C==1)
        if mask.sum() < 10:
            continue
        Yc,Tc,Zc,Wc=Y[mask],T[mask],Z[mask],W[mask]
        tau, SE = tau_estimate(Yc,Tc,Zc,Wc,use_iv,use_ipsw)
        if np.isnan(tau) or np.isnan(SE) or SE < 1e-12:
            continue
        z = tau/SE
        rej = bool(abs(z) > scipy.stats.norm.ppf(1.0-ALPHA/2))
        reject.append(rej)
        tau_list.append(float(tau))

    n_valid = len(reject)
    if n_valid == 0:
        return {"n_valid": 0, "reject_rate": np.nan, "tau_mean": np.nan}
    return {
        "n_valid": n_valid,
        "reject_rate": float(np.mean(reject)),
        "reject_SE": float(np.std(reject)/np.sqrt(n_valid)),
        "tau_mean": float(np.mean(tau_list)),
        "tau_std":  float(np.std(tau_list)),
    }


# Env log

def build_env_log() -> Dict:
    git_hash = subprocess.run(
        ["git","-C",str(BASE_DIR/"pilot"),"rev-parse","HEAD"],
        capture_output=True, text=True).stdout.strip() or "unknown"
    return {
        "script": "exp4_ablation.py",
        "date": datetime.now().isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "git_commit": git_hash,
        "N_final": N_FINAL, "N_rep": N_REP,
        "M_reps": M_REPS, "seeds": SEEDS,
        "combos": [combo_label(*c) for c in COMBOS],
        "status": "exploratory",
        "gpu_used": False,
    }


# Main

def main() -> None:
    print("=== Exp4: Component Ablation Study ===")
    print(f"8 combos (P=physics, I=IV, W=IPSW), uppercase=ON, lowercase=OFF")
    print(f"Cells: T2-a, T2-b (power) + T2-null (size)")
    print(f"N={N_FINAL:,} × 5 seeds for calibration; M={M_REPS} × N_rep={N_REP:,} for size/power")
    print()

    all_cells = CELLS_POWER + [CELL_NULL]

    # --- Part A: Calibration (N=1M x 5 seeds), tau bias ---
    print("--- Part A: Calibration (N=1M x 5 seeds per combo per cell) ---")
    calib_rows: List[Dict] = []
    for cell_cfg in all_cells:
        cell = cell_cfg["cell"]
        rho  = cell_cfg["rho"]
        ab   = cell_cfg["alpha_b"]
        for use_p, use_i, use_w in COMBOS:
            lb = combo_label(use_p, use_i, use_w)
            for seed in SEEDS:
                row = run_calibration_seed(cell, rho, ab, seed, use_p, use_i, use_w)
                calib_rows.append(row)
            # print progress
            df_tmp = pd.DataFrame([r for r in calib_rows
                                   if r["cell"]==cell and r["combo"]==lb])
            if len(df_tmp) == 5:
                print(f"  {cell} [{lb}]: "
                      f"tau={df_tmp['tau_hat'].mean():.5f}±{df_tmp['tau_hat'].std():.5f}, "
                      f"bias={df_tmp['bias'].mean():+.5f}, "
                      f"cov={df_tmp['ci_covers_true'].mean():.2f}")

    df_calib = pd.DataFrame(calib_rows)
    csv_a = RESULTS_DIR / "exp4_ablation_calibration.csv"
    df_calib.to_csv(csv_a, index=False)
    print(f"\nSaved calibration CSV: {csv_a}")

    # --- Part B: Size/Power (M-rep at N_rep=10k) ---
    print("\n--- Part B: Size/Power (M=500, N_rep=10k per combo) ---")
    mrep_rows: List[Dict] = []
    for cell_cfg in all_cells:
        cell = cell_cfg["cell"]
        rho  = cell_cfg["rho"]
        ab   = cell_cfg["alpha_b"]
        tau_true = TAU_LATE_TRUE.get(cell, np.nan)
        role = "SIZE" if cell == "T2-null" else "POWER"
        for use_p, use_i, use_w in COMBOS:
            lb = combo_label(use_p, use_i, use_w)
            res = run_mreps_ablation(rho, ab, use_i, use_w, N_REP, M_REPS, BASE_SEED)
            row = {
                "cell": cell, "rho": rho, "alpha_b": ab,
                "tau_LATE_true": tau_true, "role": role,
                "use_physics": use_p, "use_iv": use_i, "use_ipsw": use_w,
                "combo": lb,
                **res
            }
            mrep_rows.append(row)
            print(f"  {cell} [{lb}] ({role}): reject={res.get('reject_rate', float('nan')):.3f}")

    df_mrep = pd.DataFrame(mrep_rows)
    csv_b = RESULTS_DIR / "exp4_ablation_size_power.csv"
    df_mrep.to_csv(csv_b, index=False)
    print(f"\nSaved size/power CSV: {csv_b}")

    # --- Aggregate JSON ---
    env_log = build_env_log()
    agg: Dict = {
        "env_log": env_log,
        "combos": {},
        "summary_table": [],
    }

    for use_p, use_i, use_w in COMBOS:
        lb = combo_label(use_p, use_i, use_w)
        combo_entry: Dict = {
            "use_physics": use_p, "use_iv": use_i, "use_ipsw": use_w,
            "description": (
                f"Physics={'S-scale+D bounds' if use_p else 'none'}, "
                f"IV={'IV Wald' if use_i else 'naive mean diff'}, "
                f"IPSW={'oracle weights' if use_w else 'unweighted C=1'}"
            ),
            "cells": {}
        }
        for cell_cfg in all_cells:
            cell = cell_cfg["cell"]
            df_c = df_calib[(df_calib["cell"]==cell) & (df_calib["combo"]==lb)]
            df_m = df_mrep[(df_mrep["cell"]==cell) & (df_mrep["combo"]==lb)]
            tau_true = TAU_LATE_TRUE.get(cell, np.nan)
            combo_entry["cells"][cell] = {
                "tau_hat_mean": float(df_c["tau_hat"].mean()) if len(df_c)>0 else None,
                "tau_hat_std":  float(df_c["tau_hat"].std())  if len(df_c)>0 else None,
                "bias_mean":    float(df_c["bias"].mean())    if len(df_c)>0 else None,
                "bias_std":     float(df_c["bias"].std())     if len(df_c)>0 else None,
                "ci_coverage":  float(df_c["ci_covers_true"].mean()) if len(df_c)>0 else None,
                "kappa_width_mean": float(df_c["kappa_ident_width"].mean())
                                    if (use_p and len(df_c)>0) else None,
                "reject_rate": float(df_m["reject_rate"].values[0]) if len(df_m)>0 else None,
                "reject_SE":   float(df_m["reject_SE"].values[0])   if len(df_m)>0 else None,
            }
        agg["combos"][lb] = combo_entry

        # Summary table row
        row_T2a = agg["combos"][lb]["cells"].get("T2-a",{})
        row_T2b = agg["combos"][lb]["cells"].get("T2-b",{})
        row_null = agg["combos"][lb]["cells"].get("T2-null",{})
        agg["summary_table"].append({
            "combo": lb,
            "description": combo_entry["description"],
            "T2a_bias": row_T2a.get("bias_mean"),
            "T2a_ci_coverage": row_T2a.get("ci_coverage"),
            "T2a_power": row_T2a.get("reject_rate"),
            "T2b_bias": row_T2b.get("bias_mean"),
            "T2b_power": row_T2b.get("reject_rate"),
            "null_size": row_null.get("reject_rate"),
            "kappa_width_T2a": row_T2a.get("kappa_width_mean"),
        })

    json_path = RESULTS_DIR / "exp4_ablation_aggregate.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nSaved aggregate JSON: {json_path}")

    # Print ablation table
    print()
    print("=== Exp4 Ablation Table (summary) ===")
    print(f"{'Combo':>5} {'T2a_bias':>10} {'T2a_cov':>9} {'T2a_pow':>9} {'null_size':>9} {'κ_width':>9}")
    print("-" * 60)
    for row in agg["summary_table"]:
        def fmt(v): return f"{v:.4f}" if v is not None and not (isinstance(v, float) and np.isnan(v)) else "  N/A"
        print(f"{row['combo']:>5} {fmt(row['T2a_bias']):>10} {fmt(row['T2a_ci_coverage']):>9} "
              f"{fmt(row['T2a_power']):>9} {fmt(row['null_size']):>9} {fmt(row['kappa_width_T2a']):>9}")

    print()
    print("Legend: P=physics S-scale, I=IV Wald, W=IPSW. Upper=ON, lower=OFF.")
    print("T2a_bias: tau_hat - tau_LATE_true. T2a_cov: CI coverage of tau_LATE_true.")
    print("null_size: rejection rate at H0 (should be ≈0.05). κ_width: D-based kappa region width.")
    print(f"\nResults: {csv_a}")
    print(f"         {csv_b}")
    print(f"         {json_path}")


if __name__ == "__main__":
    main()
