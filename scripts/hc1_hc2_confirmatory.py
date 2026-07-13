"""
HC1 and HC2 confirmatory experiments completing the Bonferroni-Holm K=5 family.
HC1 verifies that the physics-constrained DR estimator recovers NIE more accurately than naive DR.
HC2 verifies that IPSW correction removes Berkson collider bias in both Tier-1 and Tier-2 cells.
Outputs: results/hc1_hc2_confirmatory_results.csv, results/hc1_hc2_confirmatory_aggregate.json
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
from scipy import stats

# Paths
BASE_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR  = BASE_DIR / "results"
SCRIPTS_DIR  = BASE_DIR / "scripts"
BACKUP_DIR   = BASE_DIR / "_backup"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
BACKUP_DIR.mkdir(parents=True, exist_ok=True)

# Frozen SCM parameters (identical to hc_t2_reduction.py)
LOGIT_P0:     float = float(np.log(0.55 / 0.45))  # logit(0.55)
GAMMA_B:      float = 0.50
STD_EPS_B:    float = 0.30
STD_EPS_V:    float = 5.00
BETA_U:       float = 0.10
STD_EPS_Y:    float = 0.05
ESC_REDUCTION: float = 0.30
ESC_ON:       float = 1.0 - ESC_REDUCTION   # 0.7
SPEED_MULT:   float = 5.0
ETA_0:        float = -2.0
ETA_T:        float = -0.5
ETA_U:        float =  0.3
ETA_B:        float =  0.2

# PHY_COEF: coefficient for physics-constrained NIE estimator
# NIE_phys = PHY_COEF * (E[S^4|T=1] - E[S^4|T=0])
# Derivation: Y(1,b) = (S*(1-0.3)/100)^4 = (S*ESC_ON/100)^4
#   NIE = E[Y(1,b(1))] - E[Y(1,b(0))] = ESC_ON^4/100^4 * (E[S(1)^4] - E[S(0)^4])
PHY_COEF: float = ESC_ON ** 4 / (100.0 ** 4)   # = 0.7^4 / 10^8 = 2.401e-9

# Experimental parameters
SEEDS:   List[int] = [42, 123, 456, 789, 2026]
N_FINAL: int       = 1_000_000
ALPHA:   float     = 0.05

# G-cell parameters
# alpha_b values from prereg table (not dynamically estimated from data)
G_CELLS: List[Dict] = [
    {"cell": "G1", "alpha_b": 0.300000, "NIE_true": 0.002286},
    {"cell": "G2", "alpha_b": 2.135938, "NIE_true": 0.020670},
    {"cell": "G3", "alpha_b": 3.971875, "NIE_true": 0.048473},
    {"cell": "G4", "alpha_b": 4.997209, "NIE_true": 0.069162},
    {"cell": "G5", "alpha_b": 5.869922, "NIE_true": 0.090215},
]

# T2-cell parameters (results from existing pilot run)
T2_CELLS: List[str] = ["T2-a", "T2-b", "T2-c", "T2-d"]

# Previously established BH p-values
PRIOR_BH_PVALS: Dict[str, Optional[float]] = {
    "H1:HC_NI":        8.135957276033288e-15,
    "H2:HC_T2(b)":     0.0,
    "H5:HC3(G4)":      0.08645660050287927,
}

# Helpers

def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )


def generate_g_cell(alpha_b: float, seed: int, N: int = N_FINAL) -> Dict[str, np.ndarray]:
    """
    Generate N observations for a G-cell (rho=0).
    Returns dict with T, S, Y, b, C, W_ipsw arrays.
    """
    rng = np.random.default_rng(seed)

    U      = rng.standard_normal(N)
    eps_b  = rng.normal(0.0, STD_EPS_B, N)
    eps_v  = rng.normal(0.0, STD_EPS_V, N)
    eps_y  = rng.normal(0.0, STD_EPS_Y, N)

    # rho=0 → no U→T confounding
    P_T    = float(0.55) * np.ones(N)
    T      = rng.binomial(1, P_T).astype(float)

    b       = alpha_b * T + GAMMA_B * U + eps_b
    v_base  = 50.0 + 10.0 * U + eps_v
    v_actual = v_base + SPEED_MULT * b
    S        = v_actual  # speed variable: S = v_actual
    delta_v  = v_actual * (1.0 - ESC_REDUCTION * T)
    Y        = (delta_v / 100.0) ** 4 + BETA_U * U + eps_y

    P_crash  = sigmoid(ETA_0 + ETA_T * T + ETA_U * U + ETA_B * b)
    C        = rng.binomial(1, P_crash).astype(float)
    W_ipsw   = 1.0 / P_crash   # oracle IPSW weight

    return {"T": T, "S": S, "Y": Y, "b": b, "C": C, "W_ipsw": W_ipsw}


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    """Hajek weighted mean."""
    return float(np.dot(weights, values) / weights.sum())


# HC1: Physics-constrained estimator vs naive DR

def run_hc1_seed(alpha_b: float, seed: int) -> Dict:
    """
    HC1 computation for one (G-cell, seed) pair.

    Physics estimator: NIE_phys = PHY_COEF * (mean(S^4|T=1) - mean(S^4|T=0))
    Naive DR:          NIE_naive = alpha_b_hat * beta_b_hat
                       where alpha_b_hat = OLS coefficient T in [b ~ T]
                             beta_b_hat  = OLS coefficient b in [Y ~ T + b]
    CI method:         Asymptotic (CLT-based) 95% CI for NIE_phys.
                       Equivalent to n_bootstrap→inf percentile bootstrap for N=10^6.
    """
    dat = generate_g_cell(alpha_b, seed)
    T, S, Y, b = dat["T"], dat["S"], dat["Y"], dat["b"]
    N = len(T)

    mask_T1 = T == 1
    mask_T0 = T == 0
    n1 = int(mask_T1.sum())
    n0 = int(mask_T0.sum())

    # --- Physics-constrained estimator ---
    S4_T1 = S[mask_T1] ** 4
    S4_T0 = S[mask_T0] ** 4
    mean_S4_T1 = float(S4_T1.mean())
    mean_S4_T0 = float(S4_T0.mean())
    NIE_phys = PHY_COEF * (mean_S4_T1 - mean_S4_T0)

    # Bootstrap-equivalent 95% CI (asymptotic normal, valid for N=10^6)
    # SE = PHY_COEF * sqrt(Var(S4|T=1)/n1 + Var(S4|T=0)/n0)
    # This equals the limit of the bootstrap SE as n_bootstrap -> inf
    var_S4_T1 = float(S4_T1.var(ddof=1))
    var_S4_T0 = float(S4_T0.var(ddof=1))
    SE_phys = PHY_COEF * float(np.sqrt(var_S4_T1 / n1 + var_S4_T0 / n0))
    ci_l = NIE_phys - 1.96 * SE_phys
    ci_u = NIE_phys + 1.96 * SE_phys

    # --- Naive DR: linear OLS mediation ---
    # Step 1: b ~ a0 + a1*T  → alpha_b_hat
    X_med      = np.column_stack([np.ones(N), T])
    coef_med   = np.linalg.lstsq(X_med, b, rcond=None)[0]
    alpha_b_hat = float(coef_med[1])

    # Step 2: Y ~ g0 + g1*T + g2*b  (linear, misspecified for (S*0.7/100)^4)
    X_out     = np.column_stack([np.ones(N), T, b])
    coef_out  = np.linalg.lstsq(X_out, Y, rcond=None)[0]
    beta_b_hat = float(coef_out[2])

    # NIE_naive = product of coefficients (Baron-Kenny mediation)
    NIE_naive = alpha_b_hat * beta_b_hat

    return {
        "seed":         seed,
        "NIE_phys":     float(NIE_phys),
        "SE_phys":      float(SE_phys),
        "ci_l":         float(ci_l),
        "ci_u":         float(ci_u),
        "NIE_naive":    float(NIE_naive),
        "alpha_b_hat":  alpha_b_hat,
        "beta_b_hat":   beta_b_hat,
        "n1":           n1,
        "n0":           n0,
    }


def compute_hc1_cell(cell_info: Dict) -> Dict:
    """
    Run HC1 for one G-cell across all 5 seeds, return per-cell summary + p-value.
    """
    cell      = cell_info["cell"]
    alpha_b   = cell_info["alpha_b"]
    NIE_true  = cell_info["NIE_true"]

    rows  = []
    for seed in SEEDS:
        r = run_hc1_seed(alpha_b, seed)
        r["cell"]     = cell
        r["alpha_b"]  = alpha_b
        r["NIE_true"] = NIE_true
        r["err_phys"]  = r["NIE_phys"] - NIE_true
        r["err_naive"] = r["NIE_naive"] - NIE_true
        r["ci_covers"] = int(r["ci_l"] <= NIE_true <= r["ci_u"])
        rows.append(r)

    df = pd.DataFrame(rows)
    # RMSE per cell
    rmse_phys  = float(np.sqrt((df["err_phys"]  ** 2).mean()))
    rmse_naive = float(np.sqrt((df["err_naive"] ** 2).mean()))

    rmse_ratio = rmse_phys / rmse_naive if rmse_naive > 1e-15 else 0.0

    # Criterion (a): RMSE_phys <= 0.5 * RMSE_naive
    criterion_a = bool(rmse_phys <= 0.5 * rmse_naive)

    # Criterion (b): all 5 CIs cover NIE_true
    n_ci_cover    = int(df["ci_covers"].sum())
    criterion_b   = bool(n_ci_cover == len(SEEDS))

    # Cell-level p-value: paired one-sided t-test
    # H0: E[err_phys^2] >= 0.25 * E[err_naive^2]  (phys is NOT clearly better)
    # H1: E[err_phys^2]  < 0.25 * E[err_naive^2]  (phys IS clearly better)
    # Test statistic: d_s = err_phys_s^2 - 0.25*err_naive_s^2
    d_vals = df["err_phys"] ** 2 - 0.25 * df["err_naive"] ** 2
    d_mean = float(d_vals.mean())
    d_std  = float(d_vals.std(ddof=1))
    if d_std < 1e-20:
        # Near-constant d: if mean < 0, p ≈ 0 (clearly passes); else p = 1
        cell_pval = 0.0 if d_mean < 0 else 1.0
        t_stat    = float(-np.inf) if d_mean < 0 else float(np.inf)
    else:
        t_stat    = float(d_mean / (d_std / np.sqrt(len(SEEDS))))
        cell_pval = float(stats.t.cdf(t_stat, df=len(SEEDS) - 1))  # left tail

    return {
        "cell":          cell,
        "alpha_b":       alpha_b,
        "NIE_true":      NIE_true,
        "rmse_phys":     rmse_phys,
        "rmse_naive":    rmse_naive,
        "rmse_ratio":    rmse_ratio,
        "criterion_a":   criterion_a,
        "n_ci_cover":    n_ci_cover,
        "criterion_b":   criterion_b,
        "criterion_pass": criterion_a and criterion_b,
        "d_mean":        float(d_mean),
        "d_std":         float(d_std),
        "t_stat":        float(t_stat),
        "cell_pval":     float(cell_pval),
        "per_seed_rows": rows,
    }


# HC2: IPSW vs no-IPSW  (G cells only — T2 cells loaded from existing file)

def run_hc2_gcell_seed(alpha_b: float, seed: int) -> Dict:
    """
    HC2 computation for one (G-cell, seed) pair.

    No-IPSW:  NIE_raw  = PHY_COEF*(mean(S^4|T=1,C=1) - mean(S^4|T=0,C=1))  [crash-conditional]
    With-IPSW: NIE_ipsw = PHY_COEF*(wt_mean_w(S^4|T=1,C=1) - wt_mean_w(S^4|T=0,C=1))
    """
    dat = generate_g_cell(alpha_b, seed)
    T, S, C, W_ipsw = dat["T"], dat["S"], dat["C"], dat["W_ipsw"]

    mask_C1 = C == 1
    T_c1    = T[mask_C1]
    S_c1    = S[mask_C1]
    W_c1    = W_ipsw[mask_C1]

    mask_T1_c1 = T_c1 == 1
    mask_T0_c1 = T_c1 == 0

    n_c1 = int(mask_C1.sum())
    n1_c1 = int(mask_T1_c1.sum())
    n0_c1 = int(mask_T0_c1.sum())

    S4_c1 = S_c1 ** 4

    # No-IPSW: raw crash-conditional
    mean_S4_T1_raw = float(S4_c1[mask_T1_c1].mean())
    mean_S4_T0_raw = float(S4_c1[mask_T0_c1].mean())
    NIE_raw = PHY_COEF * (mean_S4_T1_raw - mean_S4_T0_raw)

    # With-IPSW: Hajek-weighted crash-conditional
    W_T1 = W_c1[mask_T1_c1]
    W_T0 = W_c1[mask_T0_c1]
    mean_S4_T1_ipsw = float(
        np.dot(W_T1, S4_c1[mask_T1_c1]) / W_T1.sum()
    )
    mean_S4_T0_ipsw = float(
        np.dot(W_T0, S4_c1[mask_T0_c1]) / W_T0.sum()
    )
    NIE_ipsw = PHY_COEF * (mean_S4_T1_ipsw - mean_S4_T0_ipsw)

    return {
        "seed":             seed,
        "NIE_raw":          NIE_raw,
        "NIE_ipsw":         NIE_ipsw,
        "n_c1":             n_c1,
        "n1_c1":            n1_c1,
        "n0_c1":            n0_c1,
        "mean_S4_T1_raw":   mean_S4_T1_raw,
        "mean_S4_T0_raw":   mean_S4_T0_raw,
        "mean_S4_T1_ipsw":  mean_S4_T1_ipsw,
        "mean_S4_T0_ipsw":  mean_S4_T0_ipsw,
    }


def compute_hc2_gcell(cell_info: Dict) -> Dict:
    """
    HC2 for one G-cell: raw crash-conditional vs IPSW-corrected.
    Reference: NIE_pop_true (from G-cell parameters table).
    """
    cell     = cell_info["cell"]
    alpha_b  = cell_info["alpha_b"]
    NIE_true = cell_info["NIE_true"]

    rows = []
    for seed in SEEDS:
        r = run_hc2_gcell_seed(alpha_b, seed)
        r["cell"]         = cell
        r["NIE_true"]     = NIE_true
        # Relative biases (fraction of |NIE_true|)
        if abs(NIE_true) > 1e-10:
            r["rel_bias_raw"]  = abs(r["NIE_raw"]  - NIE_true) / abs(NIE_true)
            r["rel_bias_ipsw"] = abs(r["NIE_ipsw"] - NIE_true) / abs(NIE_true)
        else:
            r["rel_bias_raw"]  = float("nan")
            r["rel_bias_ipsw"] = float("nan")
        rows.append(r)

    df = pd.DataFrame(rows)

    mean_rel_raw  = float(df["rel_bias_raw"].mean())
    mean_rel_ipsw = float(df["rel_bias_ipsw"].mean())
    criterion_no_ipsw  = bool(mean_rel_raw  >= 0.20)  # >= 20%
    criterion_with_ipsw = bool(mean_rel_ipsw <= 0.05)  # <= 5%
    criterion_pass = criterion_no_ipsw and criterion_with_ipsw

    # Cell p-value: one-sided paired t-test
    # H0: IPSW does not reduce bias  (d <= 0 in mean)
    # H1: IPSW reduces bias           (d > 0)
    # d_s = rel_bias_raw_s - rel_bias_ipsw_s  (positive = IPSW helps)
    d_vals = df["rel_bias_raw"] - df["rel_bias_ipsw"]
    d_mean = float(d_vals.mean())
    d_std  = float(d_vals.std(ddof=1))
    if d_std < 1e-20:
        cell_pval = 0.0 if d_mean > 0 else 1.0
        t_stat    = float(np.inf) if d_mean > 0 else float(-np.inf)
    else:
        t_stat    = float(d_mean / (d_std / np.sqrt(len(SEEDS))))
        cell_pval = float(1.0 - stats.t.cdf(t_stat, df=len(SEEDS) - 1))  # right tail

    return {
        "cell":                cell,
        "NIE_true":            NIE_true,
        "mean_rel_bias_raw":   mean_rel_raw,
        "mean_rel_bias_ipsw":  mean_rel_ipsw,
        "criterion_no_ipsw":   criterion_no_ipsw,
        "criterion_with_ipsw": criterion_with_ipsw,
        "criterion_pass":      criterion_pass,
        "t_stat":              float(t_stat),
        "cell_pval":           float(cell_pval),
        "per_seed_rows":       rows,
    }


def compute_hc2_t2cells() -> List[Dict]:
    """
    Load existing T2-cell results from hc_t2_reduction_per_cell.csv.
    Compute relative bias of tau_IV_C1_raw (no IPSW) and tau_IV_IPSW (with IPSW)
    relative to tau_LATE_true.
    Returns one summary dict per T2 cell.
    """
    csv_path = RESULTS_DIR / "hc_t2_reduction_per_cell.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Required file not found: {csv_path}")

    df_all = pd.read_csv(csv_path)

    summaries = []
    for cell_name in T2_CELLS:
        df_c = df_all[df_all["cell"] == cell_name].copy()
        if len(df_c) == 0:
            raise ValueError(f"No rows for cell {cell_name} in {csv_path}")
        if len(df_c) != len(SEEDS):
            raise ValueError(
                f"Expected {len(SEEDS)} seeds for {cell_name}, got {len(df_c)}"
            )

        # Sort by seed for reproducibility
        df_c = df_c.set_index("seed").loc[SEEDS].reset_index()

        tau_true   = df_c["tau_LATE_true"].values
        tau_no_ipsw = df_c["tau_IV_C1_raw"].values
        tau_ipsw    = df_c["tau_IV_IPSW"].values

        # Use |tau_LATE_true| as the reference magnitude (same sign convention)
        ref_mag = np.abs(tau_true)
        rel_bias_no_ipsw = np.abs(tau_no_ipsw - tau_true) / ref_mag
        rel_bias_ipsw    = np.abs(tau_ipsw    - tau_true) / ref_mag

        mean_rel_no_ipsw  = float(rel_bias_no_ipsw.mean())
        mean_rel_ipsw_mean = float(rel_bias_ipsw.mean())

        criterion_no_ipsw   = bool(mean_rel_no_ipsw  >= 0.20)
        criterion_with_ipsw = bool(mean_rel_ipsw_mean <= 0.05)
        criterion_pass = criterion_no_ipsw and criterion_with_ipsw

        # Cell p-value
        d_vals = rel_bias_no_ipsw - rel_bias_ipsw
        d_mean = float(d_vals.mean())
        d_std  = float(d_vals.std(ddof=1))
        if d_std < 1e-20:
            cell_pval = 0.0 if d_mean > 0 else 1.0
            t_stat    = float(np.inf) if d_mean > 0 else float(-np.inf)
        else:
            t_stat    = float(d_mean / (d_std / np.sqrt(len(SEEDS))))
            cell_pval = float(1.0 - stats.t.cdf(t_stat, df=len(SEEDS) - 1))

        per_seed = [
            {
                "seed":         s,
                "cell":         cell_name,   # include cell name for CSV traceability
                "tau_true":     float(tau_true[i]),
                "tau_no_ipsw":  float(tau_no_ipsw[i]),
                "tau_ipsw":     float(tau_ipsw[i]),
                "rel_bias_raw": float(rel_bias_no_ipsw[i]),
                "rel_bias_ipsw": float(rel_bias_ipsw[i]),
            }
            for i, s in enumerate(SEEDS)
        ]

        summaries.append({
            "cell":                cell_name,
            "NIE_true_label":      "tau_LATE_true",
            "mean_tau_true":       float(tau_true.mean()),
            "mean_rel_bias_raw":   mean_rel_no_ipsw,
            "mean_rel_bias_ipsw":  mean_rel_ipsw_mean,
            "criterion_no_ipsw":   criterion_no_ipsw,
            "criterion_with_ipsw": criterion_with_ipsw,
            "criterion_pass":      criterion_pass,
            "t_stat":              float(t_stat),
            "cell_pval":           float(cell_pval),
            "per_seed_rows":       per_seed,
        })

    return summaries


# Bonferroni-Holm (Holm 1979) correction — K=5 hypotheses

def bonferroni_holm(
    p_values: Dict[str, Optional[float]],
    alpha: float = 0.05,
) -> Dict:
    """
    Bonferroni-Holm stepdown correction.

    p_values: dict mapping hypothesis label to raw p-value or None (pending/excluded).
    Returns dict with adjusted p-values and REJECT/FAIL decisions.
    """
    labels  = list(p_values.keys())
    raw_p   = [p_values[k] for k in labels]
    K_total = len(labels)

    # Separate valid (non-None) p-values
    valid_idx = [i for i, p in enumerate(raw_p) if p is not None]
    K_valid   = len(valid_idx)

    if K_valid == 0:
        return {
            "alpha": alpha, "K_total": K_total, "K_valid": K_valid,
            "labels": labels, "raw_p": raw_p,
            "adj_p": [None] * K_total, "reject": [None] * K_total,
        }

    valid_p     = [raw_p[i] for i in valid_idx]
    sorted_idxs = sorted(range(K_valid), key=lambda k: valid_p[k])

    adj_p_valid = [None] * K_valid
    reject_valid = [None] * K_valid

    cum_reject = True
    for rank, k in enumerate(sorted_idxs):
        threshold = alpha / (K_total - rank)   # Holm uses K_total (not K_valid) in denominator
        adj = min(raw_p[valid_idx[k]] * (K_total - rank), 1.0)
        # Enforce monotonicity in stepdown: once we fail to reject, all lower ranks also fail
        if not cum_reject:
            adj_p_valid[k]   = adj
            reject_valid[k]  = False
        else:
            adj_p_valid[k]  = adj
            if raw_p[valid_idx[k]] <= threshold:
                reject_valid[k] = True
            else:
                reject_valid[k] = False
                cum_reject       = False

    # Reconstruct full arrays (None for pending/excluded)
    adj_p_full   = [None] * K_total
    reject_full  = [None] * K_total
    for j, i in enumerate(valid_idx):
        adj_p_full[i]  = adj_p_valid[j]
        reject_full[i] = reject_valid[j]

    return {
        "alpha":    alpha,
        "K_total":  K_total,
        "K_valid":  K_valid,
        "labels":   labels,
        "raw_p":    raw_p,
        "adj_p":    adj_p_full,
        "reject":   reject_full,
        "method":   "Bonferroni-Holm stepdown (Holm 1979)",
    }


# Environment log

def get_env_log() -> Dict:
    """Collect environment metadata for reproducibility."""
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(BASE_DIR),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        git_commit = "unavailable"

    return {
        "timestamp":       datetime.utcnow().isoformat() + "Z",
        "python_version":  sys.version,
        "numpy_version":   np.__version__,
        "scipy_version":   scipy_version(),
        "pandas_version":  pd.__version__,
        "platform":        platform.platform(),
        "hostname":        socket.gethostname(),
        "git_commit":      git_commit,
        "seeds":           SEEDS,
        "N_per_seed":      N_FINAL,
        "ESC_REDUCTION":   ESC_REDUCTION,
        "PHY_COEF":        PHY_COEF,
        "alpha_level":     ALPHA,
        "script":          Path(__file__).name,
        "GPU":             "none (CPU-only)",
    }


def scipy_version() -> str:
    """Get scipy version string."""
    import scipy
    return scipy.__version__


# Main

def main() -> None:
    print("=" * 70)
    print("HC1 + HC2 Confirmatory Experiment — BH K=5 Completion")
    print(f"Start: {datetime.utcnow().isoformat()}Z")
    print("=" * 70)

    env_log = get_env_log()
    print(f"git: {env_log['git_commit'][:12]}")
    print(f"N={N_FINAL:,} x {len(SEEDS)} seeds")

    # HC1
    print("\n[HC1] Physics-constrained vs naive DR — G1-G5 (rho=0)")
    print("-" * 60)

    hc1_per_seed_rows: List[Dict] = []
    hc1_cell_summaries: List[Dict] = []

    for cell_info in G_CELLS:
        c_res = compute_hc1_cell(cell_info)
        hc1_cell_summaries.append(c_res)
        for row in c_res["per_seed_rows"]:
            hc1_per_seed_rows.append(row)

        print(
            f"  {c_res['cell']:3s}  NIE_true={c_res['NIE_true']:.6f} | "
            f"RMSE_phys={c_res['rmse_phys']:.6f}  RMSE_naive={c_res['rmse_naive']:.6f}  "
            f"ratio={c_res['rmse_ratio']:.4f}  "
            f"crit_a={'PASS' if c_res['criterion_a'] else 'FAIL'}  "
            f"crit_b={c_res['n_ci_cover']}/5  "
            f"p={c_res['cell_pval']:.3e}"
        )

    # Overall HC1 criterion
    hc1_all_a    = all(c["criterion_a"]    for c in hc1_cell_summaries)
    hc1_all_b    = all(c["criterion_b"]    for c in hc1_cell_summaries)
    hc1_all_pass = all(c["criterion_pass"] for c in hc1_cell_summaries)
    hc1_max_pval = float(max(c["cell_pval"] for c in hc1_cell_summaries))

    print(f"\n  HC1 overall: crit_a={'PASS' if hc1_all_a else 'FAIL'}  "
          f"crit_b={'PASS' if hc1_all_b else 'FAIL'}  "
          f"ALL={'PASS' if hc1_all_pass else 'FAIL'}")
    print(f"  HC1 p-value (H3, max across 5 cells): {hc1_max_pval:.4e}")

    # HC2
    print("\n[HC2] IPSW collider correction — G1-G5 + T2-a..d")
    print("-" * 60)

    hc2_per_seed_rows: List[Dict] = []
    hc2_cell_summaries: List[Dict] = []

    # G cells
    for cell_info in G_CELLS:
        c_res = compute_hc2_gcell(cell_info)
        hc2_cell_summaries.append(c_res)
        for row in c_res["per_seed_rows"]:
            row["cell_type"] = "G"
            hc2_per_seed_rows.append(row)

        print(
            f"  {c_res['cell']:4s}  NIE_true={c_res['NIE_true']:.6f} | "
            f"raw_bias={c_res['mean_rel_bias_raw']*100:.1f}%  "
            f"ipsw_bias={c_res['mean_rel_bias_ipsw']*100:.1f}%  "
            f"no_ipsw={'PASS' if c_res['criterion_no_ipsw'] else 'FAIL'}  "
            f"w_ipsw={'PASS' if c_res['criterion_with_ipsw'] else 'FAIL'}  "
            f"p={c_res['cell_pval']:.3e}"
        )

    # T2 cells (from existing results)
    t2_summaries = compute_hc2_t2cells()
    for c_res in t2_summaries:
        hc2_cell_summaries.append(c_res)
        for row in c_res["per_seed_rows"]:
            row["cell_type"] = "T2"
            hc2_per_seed_rows.append(row)

        print(
            f"  {c_res['cell']:4s}  tau_true={c_res['mean_tau_true']:.6f} | "
            f"raw_bias={c_res['mean_rel_bias_raw']*100:.1f}%  "
            f"ipsw_bias={c_res['mean_rel_bias_ipsw']*100:.1f}%  "
            f"no_ipsw={'PASS' if c_res['criterion_no_ipsw'] else 'FAIL'}  "
            f"w_ipsw={'PASS' if c_res['criterion_with_ipsw'] else 'FAIL'}  "
            f"p={c_res['cell_pval']:.3e}"
        )

    # Overall HC2 criterion
    hc2_all_no_ipsw  = all(c["criterion_no_ipsw"]   for c in hc2_cell_summaries)
    hc2_all_w_ipsw   = all(c["criterion_with_ipsw"] for c in hc2_cell_summaries)
    hc2_all_pass     = all(c["criterion_pass"]       for c in hc2_cell_summaries)
    hc2_max_pval     = float(max(c["cell_pval"]      for c in hc2_cell_summaries))

    print(f"\n  HC2 overall: no_ipsw={'PASS' if hc2_all_no_ipsw else 'FAIL'}  "
          f"w_ipsw={'PASS' if hc2_all_w_ipsw else 'FAIL'}  "
          f"ALL={'PASS' if hc2_all_pass else 'FAIL'}")
    print(f"  HC2 p-value (H4, max across 9 cells): {hc2_max_pval:.4e}")

    # BH K=5
    print("\n[Bonferroni-Holm K=5] Complete family H1-H5")
    print("-" * 60)

    full_pvals: Dict[str, Optional[float]] = {
        "H1:HC_NI":         PRIOR_BH_PVALS["H1:HC_NI"],
        "H2:HC_T2(b)":      PRIOR_BH_PVALS["H2:HC_T2(b)"],
        "H3:HC1":           hc1_max_pval,
        "H4:HC2":           hc2_max_pval,
        "H5:HC3(G4)":       PRIOR_BH_PVALS["H5:HC3(G4)"],
    }

    bh = bonferroni_holm(full_pvals, alpha=ALPHA)

    print(f"\n  {'Hypothesis':<22} {'raw_p':>12}  {'adj_p':>12}  {'Decision':>15}")
    print(f"  {'-'*22} {'-'*12}  {'-'*12}  {'-'*15}")
    for lbl, rp, ap, rj in zip(bh["labels"], bh["raw_p"], bh["adj_p"], bh["reject"]):
        rp_str = f"{rp:.4e}" if rp is not None else "N/A"
        ap_str = f"{ap:.4e}" if ap is not None else "N/A"
        rj_str = "REJECT H0" if rj is True else ("FAIL TO REJECT" if rj is False else "PENDING")
        print(f"  {lbl:<22} {rp_str:>12}  {ap_str:>12}  {rj_str:>15}")

    print("\n  Note: HC3 size/power rejection-rate estimation is NOT included in BH per prereg §6.2.")

    # Save
    # Build flat per-seed DataFrame for HC1
    hc1_df = pd.DataFrame([
        {
            "cell":       r["cell"],
            "alpha_b":    r["alpha_b"],
            "NIE_true":   r["NIE_true"],
            "seed":       r["seed"],
            "NIE_phys":   r["NIE_phys"],
            "ci_l":       r["ci_l"],
            "ci_u":       r["ci_u"],
            "ci_covers":  r["ci_covers"],
            "NIE_naive":  r["NIE_naive"],
            "err_phys":   r["err_phys"],
            "err_naive":  r["err_naive"],
        }
        for r in hc1_per_seed_rows
    ])

    # Build flat per-seed DataFrame for HC2
    hc2_g_df = pd.DataFrame([
        {
            "cell":            r.get("cell", ""),
            "cell_type":       r.get("cell_type", "G"),
            "seed":            r.get("seed", ""),
            "NIE_true":        r.get("NIE_true", float("nan")),
            "NIE_raw":         r.get("NIE_raw", float("nan")),
            "NIE_ipsw":        r.get("NIE_ipsw", float("nan")),
            "rel_bias_raw":    r.get("rel_bias_raw", float("nan")),
            "rel_bias_ipsw":   r.get("rel_bias_ipsw", float("nan")),
        }
        for r in hc2_per_seed_rows if r.get("cell_type") == "G"
    ])

    hc2_t2_df = pd.DataFrame([
        {
            "cell":            r.get("cell", ""),
            "cell_type":       "T2",
            "seed":            r.get("seed", ""),
            "NIE_true":        r.get("tau_true", float("nan")),
            "NIE_raw":         r.get("tau_no_ipsw", float("nan")),
            "NIE_ipsw":        r.get("tau_ipsw", float("nan")),
            "rel_bias_raw":    r.get("rel_bias_raw", float("nan")),
            "rel_bias_ipsw":   r.get("rel_bias_ipsw", float("nan")),
        }
        for r in hc2_per_seed_rows if r.get("cell_type") == "T2"
    ])

    hc2_df = pd.concat([hc2_g_df, hc2_t2_df], ignore_index=True)

    # Combined CSV
    out_stem = "hc1_hc2_confirmatory_2026-07-11"
    csv_path = RESULTS_DIR / f"{out_stem}.csv"

    # Mark source experiment in each frame
    hc1_df["experiment"] = "HC1"
    hc2_df["experiment"] = "HC2"

    combined_df = pd.concat(
        [hc1_df.reindex(columns=["experiment","cell","cell_type","seed","NIE_true",
                                   "NIE_phys","ci_l","ci_u","ci_covers","NIE_naive",
                                   "err_phys","err_naive","rel_bias_raw","rel_bias_ipsw",
                                   "NIE_raw","NIE_ipsw"]),
         hc2_df.reindex(columns=["experiment","cell","cell_type","seed","NIE_true",
                                   "NIE_phys","ci_l","ci_u","ci_covers","NIE_naive",
                                   "err_phys","err_naive","rel_bias_raw","rel_bias_ipsw",
                                   "NIE_raw","NIE_ipsw"])],
        ignore_index=True,
    )

    # Assign cell_type for HC1 rows
    combined_df.loc[combined_df["experiment"] == "HC1", "cell_type"] = "G"

    combined_df.to_csv(csv_path, index=False, float_format="%.8f")
    print(f"\nSaved CSV: {csv_path}")

    # JSON aggregate
    aggregate = {
        "env_log":       env_log,
        "HC1_summary": {
            "overall_pass":   hc1_all_pass,
            "criterion_a_pass": hc1_all_a,
            "criterion_b_pass": hc1_all_b,
            "H3_raw_pval":    hc1_max_pval,
            "bootstrap_method": "CLT normal approximation (equivalent to n_bootstrap->inf percentile bootstrap for N=10^6)",
            "per_cell": [
                {k: v for k, v in c.items() if k != "per_seed_rows"}
                for c in hc1_cell_summaries
            ],
        },
        "HC2_summary": {
            "overall_pass":       hc2_all_pass,
            "criterion_no_ipsw":  hc2_all_no_ipsw,
            "criterion_with_ipsw": hc2_all_w_ipsw,
            "H4_raw_pval":        hc2_max_pval,
            "per_cell_G":  [
                {k: v for k, v in c.items() if k != "per_seed_rows"}
                for c in hc2_cell_summaries if c.get("cell", "").startswith("G")
            ],
            "per_cell_T2": [
                {k: v for k, v in c.items() if k != "per_seed_rows"}
                for c in hc2_cell_summaries if c.get("cell", "").startswith("T2")
            ],
        },
        "bonferroni_holm_K5_complete": {
            "alpha":  ALPHA,
            "K":      5,
            "note":   "HC3 size/power rejection-rate estimation excluded per prereg §6.2",
            "hypotheses": [
                {
                    "label":   lbl,
                    "raw_p":   rp,
                    "adj_p":   ap,
                    "reject":  rj,
                    "decision": "REJECT H0" if rj is True else (
                                "FAIL TO REJECT" if rj is False else "PENDING"),
                }
                for lbl, rp, ap, rj in zip(
                    bh["labels"], bh["raw_p"], bh["adj_p"], bh["reject"]
                )
            ],
            "method": bh["method"],
        },
    }

    json_path = RESULTS_DIR / f"{out_stem}.json"
    with open(json_path, "w") as f:
        json.dump(aggregate, f, indent=2, allow_nan=True)
    print(f"Saved JSON: {json_path}")

    # Backup
    import shutil
    for fpath in [csv_path, json_path]:
        shutil.copy2(fpath, BACKUP_DIR / fpath.name)
    print(f"Backup: {BACKUP_DIR}")

    # Summary printout
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"HC1 (H3): {'PASS' if hc1_all_pass else 'FAIL'}  "
          f"raw_p={hc1_max_pval:.4e}")
    print(f"HC2 (H4): {'PASS' if hc2_all_pass else 'FAIL'}  "
          f"raw_p={hc2_max_pval:.4e}")
    print()
    print("Bonferroni-Holm K=5 Complete Table:")
    print(f"  {'Hypothesis':<22} {'raw_p':>12}  {'adj_p':>12}  {'Decision':>15}")
    for lbl, rp, ap, rj in zip(bh["labels"], bh["raw_p"], bh["adj_p"], bh["reject"]):
        rp_str = f"{rp:.4e}" if rp is not None else "N/A"
        ap_str = f"{ap:.4e}" if ap is not None else "N/A"
        rj_str = "REJECT H0" if rj is True else ("FAIL TO REJECT" if rj is False else "PENDING")
        print(f"  {lbl:<22} {rp_str:>12}  {ap_str:>12}  {rj_str:>15}")

    print(f"\nEnd: {datetime.utcnow().isoformat()}Z")
    print("=" * 70)

    # Return key values for post-run verification
    return aggregate


if __name__ == "__main__":
    result = main()
