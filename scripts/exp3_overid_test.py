"""
Exp3: Over-identification (Sargan/Hansen) test for exclusion restriction violations (exploratory).
Uses two IVs — Z1 (ESC mandate, IV_STR=1.5) and Z2 (AEB mandate, IV_STR=0.75) — and tests
H0: tau(Z1)=tau(Z2). Rejection at delta_ZY=0 reflects LATE heterogeneity between complier sets.
Outputs: results/exp3_overid_test.csv, results/exp3_overid_test_aggregate.json
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

# IV parameters
IV_STR_Z1: float = 1.5    # Z1: ESC mandate (original)
IV_STR_Z2: float = 0.75   # Z2: AEB mandate (second IV, weaker)

# Cell: T2-a
RHO:     float = 0.5
ALPHA_B: float = 0.3
CELL:    str   = "T2-a"

# Experiment parameters
DELTA_ZY_SWEEP: List[float] = [0.000, 0.005, 0.010, 0.020, 0.050]
SEEDS: List[int] = [42, 123, 456, 789, 2026]
N_FINAL: int = 1_000_000
N_REP:   int = 10_000
M_REPS:  int = 500
BASE_SEED: int = 88888
ALPHA:   float = 0.05


# Helpers

def sigmoid(x: np.ndarray) -> np.ndarray:
    return np.where(x >= 0, 1.0/(1.0+np.exp(-x)), np.exp(x)/(1.0+np.exp(x)))


def wald_ipsw_fast(
    Y: np.ndarray, T: np.ndarray, Z: np.ndarray, W: np.ndarray
) -> Tuple[float, float, float]:
    """IPSW Wald. Returns (tau, SE, FS)."""
    m1, m0 = (Z == 1), (Z == 0)
    w1, w0 = W[m1], W[m0]
    sw1, sw0 = w1.sum(), w0.sum()
    if sw1 < 1e-8 or sw0 < 1e-8:
        return np.nan, np.nan, np.nan
    EY1 = np.dot(Y[m1],w1)/sw1;  EY0 = np.dot(Y[m0],w0)/sw0
    ET1 = np.dot(T[m1],w1)/sw1;  ET0 = np.dot(T[m0],w0)/sw0
    FS = ET1 - ET0
    if abs(FS) < 1e-8:
        return np.nan, np.nan, float(FS)
    RF = EY1 - EY0
    tau = RF / FS
    resY1 = Y[m1]-EY1; resY0 = Y[m0]-EY0
    resT1 = T[m1]-ET1; resT0 = T[m0]-ET0
    vRF = (np.dot(w1**2,resY1**2)/sw1**2 + np.dot(w0**2,resY0**2)/sw0**2)
    vFS = (np.dot(w1**2,resT1**2)/sw1**2 + np.dot(w0**2,resT0**2)/sw0**2)
    SE = float(np.sqrt(max(1e-20, vRF/FS**2 + RF**2*vFS/FS**4)))
    return float(tau), SE, float(FS)


# tau_LATE calibration per IV (twin simulation, N=1M x 5 seeds)

def calibrate_tau_LATE(rho: float, alpha_b: float, iv_str: float, seed: int) -> Dict:
    """Twin simulation for tau_LATE for a given IV strength."""
    rng = np.random.default_rng(seed)
    N   = N_FINAL
    U   = rng.standard_normal(N)
    eps_b = rng.normal(0.0, STD_EPS_B, N)
    eps_v = rng.normal(0.0, STD_EPS_V, N)
    v_base = 50.0 + 10.0*U + eps_v

    # Counterfactual T under Z=0, Z=1 for this IV
    P_T_Z0 = sigmoid(LOGIT_P0 + rho*U + iv_str*(0.0-0.5))
    P_T_Z1 = sigmoid(LOGIT_P0 + rho*U + iv_str*(1.0-0.5))
    U_T    = rng.uniform(0.0, 1.0, N)
    T_Z0   = (U_T <= P_T_Z0).astype(float)
    T_Z1   = (U_T <= P_T_Z1).astype(float)
    is_c   = (T_Z1 > T_Z0)
    pi_c   = float(is_c.mean())

    b_T0 = GAMMA_B*U + eps_b
    b_T1 = alpha_b + GAMMA_B*U + eps_b

    def Y_cf(T_out, b_in):
        dv = (v_base + SPEED_MULT*b_in)*(1.0-ESC_REDUCTION*T_out)
        return (dv/100.0)**4 + BETA_U*U

    Y11 = Y_cf(1.0, b_T1); Y00 = Y_cf(0.0, b_T0)
    tau_LATE = float(np.mean((Y11-Y00)[is_c]))
    FS_true  = float(P_T_Z1.mean() - P_T_Z0.mean())

    return {
        "seed": seed, "iv_str": iv_str, "pi_c": pi_c,
        "tau_LATE": tau_LATE, "FS_true": FS_true,
        "EU_complier": float(U[is_c].mean()),
    }


# M-rep over-ID test at given delta_ZY

def run_mreps_overid(
    rho: float, alpha_b: float, delta_ZY: float,
    N_rep: int, M: int, base_seed: int,
) -> Dict:
    """
    M repetitions of over-identification test.
    Each rep: generate N_rep observations with Z1, Z2 both active.
    Y = physics + beta_u*U + delta_ZY*Z1 + eps_y  (violation on Z1 only)
    Compute tau_hat(Z1), tau_hat(Z2), Sargan statistic, rejection rate.
    """
    rng = np.random.default_rng(base_seed)
    rep_seeds = rng.integers(0, 2**31-1, size=M)

    sargan_stats: List[float] = []
    reject: List[bool] = []
    tau_Z1_list: List[float] = []
    tau_Z2_list: List[float] = []
    diff_list:   List[float] = []
    FS1_list:    List[float] = []
    FS2_list:    List[float] = []

    for rseed in rep_seeds:
        r = np.random.default_rng(int(rseed))
        N = N_rep
        U     = r.standard_normal(N)
        eps_b = r.normal(0.0, STD_EPS_B, N)
        eps_v = r.normal(0.0, STD_EPS_V, N)
        eps_y = r.normal(0.0, STD_EPS_Y, N)

        # Two independent IVs
        Z1 = r.binomial(1, 0.5, N).astype(float)
        Z2 = r.binomial(1, 0.5, N).astype(float)

        logit_pT = LOGIT_P0 + rho*U + IV_STR_Z1*(Z1-0.5) + IV_STR_Z2*(Z2-0.5)
        P_T  = sigmoid(logit_pT)
        T    = (r.uniform(0.0,1.0,N) <= P_T).astype(float)
        b    = alpha_b*T + GAMMA_B*U + eps_b
        vb   = 50.0 + 10.0*U + eps_v
        dv   = (vb + SPEED_MULT*b)*(1.0-ESC_REDUCTION*T)
        # Exclusion restriction violation: Z1 has direct effect delta_ZY on Y
        Y    = (dv/100.0)**4 + BETA_U*U + delta_ZY*Z1 + eps_y

        Pcr  = sigmoid(ETA_0 + ETA_T*T + ETA_U*U + ETA_B*b)
        C    = r.binomial(1, Pcr).astype(float)
        W    = 1.0 / Pcr
        mask = (C == 1)
        if mask.sum() < 20:
            continue
        Yc, Tc, Wc = Y[mask], T[mask], W[mask]
        Z1c, Z2c = Z1[mask], Z2[mask]

        # IV Wald IPSW for Z1 and Z2 separately
        t1, se1, fs1 = wald_ipsw_fast(Yc, Tc, Z1c, Wc)
        t2, se2, fs2 = wald_ipsw_fast(Yc, Tc, Z2c, Wc)

        if np.isnan(t1) or np.isnan(t2) or np.isnan(se1) or np.isnan(se2):
            continue
        if se1 < 1e-12 or se2 < 1e-12:
            continue

        # Sargan/Hansen test: H0: tau(Z1) = tau(Z2)
        diff = t1 - t2
        SE_diff = float(np.sqrt(se1**2 + se2**2))   # independence of Z1, Z2
        t_stat = diff / SE_diff
        rej = bool(abs(t_stat) > scipy.stats.norm.ppf(1.0 - ALPHA/2))

        sargan_stats.append(float(t_stat))
        reject.append(rej)
        tau_Z1_list.append(float(t1))
        tau_Z2_list.append(float(t2))
        diff_list.append(float(diff))
        FS1_list.append(float(fs1))
        FS2_list.append(float(fs2))

    n_valid = len(reject)
    if n_valid == 0:
        return {"n_valid": 0, "reject_rate": np.nan}

    return {
        "n_valid": n_valid,
        "reject_rate":   float(np.mean(reject)),
        "reject_SE":     float(np.std(reject) / np.sqrt(n_valid)),
        "tau_Z1_mean":   float(np.mean(tau_Z1_list)),
        "tau_Z1_std":    float(np.std(tau_Z1_list)),
        "tau_Z2_mean":   float(np.mean(tau_Z2_list)),
        "tau_Z2_std":    float(np.std(tau_Z2_list)),
        "diff_mean":     float(np.mean(diff_list)),
        "diff_std":      float(np.std(diff_list)),
        "sargan_t_mean": float(np.mean(sargan_stats)),
        "sargan_t_std":  float(np.std(sargan_stats)),
        "FS1_mean":      float(np.mean(FS1_list)),
        "FS2_mean":      float(np.mean(FS2_list)),
    }


# Env log

def build_env_log() -> Dict:
    git_hash = subprocess.run(
        ["git", "-C", str(BASE_DIR/"pilot"), "rev-parse", "HEAD"],
        capture_output=True, text=True).stdout.strip() or "unknown"
    return {
        "script": "exp3_overid_test.py",
        "date": datetime.now().isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "git_commit": git_hash,
        "cell": CELL,
        "rho": RHO, "alpha_b": ALPHA_B,
        "IV_STR_Z1": IV_STR_Z1, "IV_STR_Z2": IV_STR_Z2,
        "delta_ZY_sweep": DELTA_ZY_SWEEP,
        "N_rep": N_REP, "M_reps": M_REPS,
        "base_seed": BASE_SEED,
        "seeds_for_calibration": SEEDS,
        "status": "exploratory",
        "gpu_used": False,
    }


# Main

def main() -> None:
    print("=== Exp3: Over-identification Test ===")
    print(f"Cell: {CELL} (rho={RHO}, alpha_b={ALPHA_B})")
    print(f"Z1: IV_STR={IV_STR_Z1}, Z2: IV_STR={IV_STR_Z2}")
    print(f"delta_ZY sweep: {DELTA_ZY_SWEEP}")
    print(f"N_rep={N_REP:,}, M={M_REPS}, base_seed={BASE_SEED}")
    print()

    # Step 1: Calibrate tau_LATE for each IV separately (N=1M x 5 seeds)
    print("--- Calibrating tau_LATE_Z1 and tau_LATE_Z2 (5 seeds x N=1M) ---")
    calib_rows: List[Dict] = []
    for seed in SEEDS:
        r1 = calibrate_tau_LATE(RHO, ALPHA_B, IV_STR_Z1, seed)
        r2 = calibrate_tau_LATE(RHO, ALPHA_B, IV_STR_Z2, seed)
        calib_rows.append({"seed": seed, "Z": "Z1", **r1})
        calib_rows.append({"seed": seed, "Z": "Z2", **r2})
        print(f"  seed={seed}: tau_LATE_Z1={r1['tau_LATE']:.6f} "
              f"(FS={r1['FS_true']:.4f}), "
              f"tau_LATE_Z2={r2['tau_LATE']:.6f} "
              f"(FS={r2['FS_true']:.4f})")

    df_calib = pd.DataFrame(calib_rows)
    tau_LATE_Z1_mean = float(df_calib[df_calib["Z"]=="Z1"]["tau_LATE"].mean())
    tau_LATE_Z2_mean = float(df_calib[df_calib["Z"]=="Z2"]["tau_LATE"].mean())
    LATE_hetero = tau_LATE_Z1_mean - tau_LATE_Z2_mean
    FS1_mean = float(df_calib[df_calib["Z"]=="Z1"]["FS_true"].mean())
    FS2_mean = float(df_calib[df_calib["Z"]=="Z2"]["FS_true"].mean())

    print(f"\ntau_LATE_Z1 = {tau_LATE_Z1_mean:.6f} (FS1={FS1_mean:.4f})")
    print(f"tau_LATE_Z2 = {tau_LATE_Z2_mean:.6f} (FS2={FS2_mean:.4f})")
    print(f"LATE heterogeneity (Z1-Z2) = {LATE_hetero:.6f}")
    print(f"Expected bias at delta_ZY>0: +delta_ZY/FS1 = {1.0/FS1_mean:.2f} x delta_ZY")
    print()

    # Step 2: M-rep over-ID test sweep
    print("--- Over-ID rejection rates vs delta_ZY ---")
    rows: List[Dict] = []
    for dzY in DELTA_ZY_SWEEP:
        print(f"  delta_ZY={dzY:.4f}: running M={M_REPS} reps ...", end=" ", flush=True)
        res = run_mreps_overid(RHO, ALPHA_B, dzY, N_REP, M_REPS, BASE_SEED)
        print(f"reject={res.get('reject_rate', float('nan')):.3f} "
              f"(tau_Z1={res.get('tau_Z1_mean', float('nan')):.5f}, "
              f"tau_Z2={res.get('tau_Z2_mean', float('nan')):.5f})")
        rows.append({"delta_ZY": dzY,
                     "tau_LATE_Z1_calib": tau_LATE_Z1_mean,
                     "tau_LATE_Z2_calib": tau_LATE_Z2_mean,
                     "LATE_heterogeneity": LATE_hetero,
                     "FS1_calib": FS1_mean,
                     "FS2_calib": FS2_mean,
                     "expected_bias_Z1": dzY / FS1_mean if FS1_mean > 1e-8 else np.nan,
                     **res})
    print()

    # Save CSV
    df = pd.DataFrame(rows)
    csv_path = RESULTS_DIR / "exp3_overid_test.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV: {csv_path}")

    # Aggregate JSON
    env_log = build_env_log()
    agg: Dict = {
        "env_log": env_log,
        "calibration": {
            "tau_LATE_Z1_mean": tau_LATE_Z1_mean,
            "tau_LATE_Z2_mean": tau_LATE_Z2_mean,
            "LATE_heterogeneity": LATE_hetero,
            "FS1_mean": FS1_mean,
            "FS2_mean": FS2_mean,
            "note": (
                "LATE heterogeneity between Z1 and Z2 complier sets is expected "
                "because complier membership depends on IV strength, leading to "
                "different E[U|complier]. Baseline rejection rate at delta_ZY=0 "
                "tests whether LATE heterogeneity causes spurious rejection."
            ),
            "per_seed": df_calib.to_dict(orient="records"),
        },
        "sweep": []
    }
    for _, row in df.iterrows():
        agg["sweep"].append(row.to_dict())

    # Interpretive summary
    base_row = df[df["delta_ZY"] == 0.0].iloc[0] if len(df[df["delta_ZY"]==0.0]) > 0 else None
    if base_row is not None:
        agg["interpretation"] = {
            "size_at_delta0": float(base_row["reject_rate"]),
            "note_on_size": (
                "Rejection rate at delta_ZY=0 reflects LATE heterogeneity between "
                "Z1 (FS≈{:.3f}) and Z2 (FS≈{:.3f}) complier sets. "
                "If > 0.05, test detects LATE heterogeneity (not just excl. restriction violation).".format(
                    FS1_mean, FS2_mean)
            ),
            "power_at_delta005": float(df[df["delta_ZY"]==0.005]["reject_rate"].values[0]) if len(df[df["delta_ZY"]==0.005]) > 0 else None,
            "power_at_delta010": float(df[df["delta_ZY"]==0.010]["reject_rate"].values[0]) if len(df[df["delta_ZY"]==0.010]) > 0 else None,
            "power_at_delta050": float(df[df["delta_ZY"]==0.050]["reject_rate"].values[0]) if len(df[df["delta_ZY"]==0.050]) > 0 else None,
        }

    json_path = RESULTS_DIR / "exp3_overid_test_aggregate.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2, ensure_ascii=False, default=str)
    print(f"Saved JSON: {json_path}")

    # Print table
    print()
    print("=== Exp3 Detection Power Table ===")
    print(f"{'delta_ZY':>10} {'reject_rate':>12} {'SE':>8} {'tau_Z1':>9} {'tau_Z2':>9} {'diff':>9}")
    print("-" * 65)
    for _, row in df.iterrows():
        print(f"{row['delta_ZY']:>10.4f} {row['reject_rate']:>12.3f} "
              f"{row['reject_SE']:>8.3f} {row['tau_Z1_mean']:>9.5f} "
              f"{row['tau_Z2_mean']:>9.5f} {row['diff_mean']:>9.5f}")

    print(f"\nLATE heterogeneity (Z1-Z2): {LATE_hetero:.6f}")
    print(f"Results: {csv_path}")
    print(f"         {json_path}")


if __name__ == "__main__":
    main()
