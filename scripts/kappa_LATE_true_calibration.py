"""
kappa_LATE_true calibration via twin simulation for Tier-2 cells (T2-a, T2-b, T2-c, T2-d).
Twin simulation protocol (7 steps): fix exogenous noise (U, eps_b, eps_v) shared across
counterfactuals → compute P_T(Z=z) for z in {0,1} → draw U_T ~ Uniform → T_i(Z=z) =
1{U_T_i <= P_T_i(Z=z)} → identify compliers (monotonicity by construction) → compute
NIE_LATE/NDE_LATE/kappa_LATE_true restricted to compliers → aggregate per seed.
Outputs: results/kappa_LATE_true_per_cell.csv, results/kappa_LATE_true_aggregate.json
"""

import json
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import scipy

# Paths
BASE_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Frozen SCM parameters
LOGIT_P0: float = float(np.log(0.55 / 0.45))   # logit(0.55) ~= 0.20067
GAMMA_B:  float = 0.50
STD_EPS_B: float = 0.30
STD_EPS_V: float = 5.00
BETA_U:   float = 0.10
STD_EPS_Y: float = 0.05
ESC_REDUCTION: float = 0.30
SPEED_MULT: float = 5.0
ETA_0: float = -2.0
ETA_T: float = -0.5
ETA_U: float =  0.3
ETA_B: float =  0.2

SEEDS:   List[int] = [42, 123, 456, 789, 2026]
N_FINAL: int = 1_000_000
IV_STR:  float = 1.5   # logit-scale shift for IV instrument

# Tier 2 cells: (cell, rho, alpha_b)
TIER2_CELLS: List[Dict] = [
    {"cell": "T2-a", "rho": 0.5, "alpha_b": 0.3},
    {"cell": "T2-b", "rho": 1.0, "alpha_b": 0.3},
    {"cell": "T2-c", "rho": 0.5, "alpha_b": 3.972},
    {"cell": "T2-d", "rho": 1.0, "alpha_b": 3.972},
]


# Helpers

def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )


def _wald_region(
    Y: np.ndarray,
    T: np.ndarray,
    Z: np.ndarray,
    NDE_true: float,
) -> Dict:
    """
    Compute Wald IV estimate of ATE_IV, then derive kappa_IV and 95% CI.

    Parameters
    ----------
    Y, T, Z : arrays of length N (all same mask already applied if C=1 subset)
    NDE_true : structural NDE (used as known physics to extract NIE_IV)

    Returns dict with: FS, RF, ATE_IV, NIE_IV, kappa_IV, SE_kappa_IV,
                       kappa_L, kappa_U, IV_width
    """
    mask_Z1 = Z == 1
    mask_Z0 = Z == 0
    n1 = int(mask_Z1.sum())
    n0 = int(mask_Z0.sum())

    E_Y_Z1 = float(np.mean(Y[mask_Z1]))
    E_Y_Z0 = float(np.mean(Y[mask_Z0]))
    E_T_Z1 = float(np.mean(T[mask_Z1]))
    E_T_Z0 = float(np.mean(T[mask_Z0]))

    FS = E_T_Z1 - E_T_Z0
    RF = E_Y_Z1 - E_Y_Z0

    if abs(FS) < 1e-6:
        raise ValueError(f"Weak IV: first stage = {FS:.8f}")

    ATE_IV   = RF / FS
    NIE_IV   = ATE_IV - NDE_true
    kappa_IV = NIE_IV / abs(NDE_true)

    var_RF = (np.var(Y[mask_Z1], ddof=1) / n1
              + np.var(Y[mask_Z0], ddof=1) / n0)
    var_FS = (E_T_Z1 * (1.0 - E_T_Z1) / n1
              + E_T_Z0 * (1.0 - E_T_Z0) / n0)
    SE_Wald     = float(np.sqrt(max(0.0, var_RF / FS**2 + RF**2 * var_FS / FS**4)))
    SE_kappa_IV = SE_Wald / abs(NDE_true)

    kappa_L = float(kappa_IV - 1.96 * SE_kappa_IV)
    kappa_U = float(kappa_IV + 1.96 * SE_kappa_IV)

    return {
        "FS"          : float(FS),
        "RF"          : float(RF),
        "ATE_IV"      : float(ATE_IV),
        "NIE_IV"      : float(NIE_IV),
        "kappa_IV"    : float(kappa_IV),
        "SE_kappa_IV" : float(SE_kappa_IV),
        "kappa_L"     : kappa_L,
        "kappa_U"     : kappa_U,
        "IV_width"    : float(kappa_U - kappa_L),
    }


# Core: one (cell, seed) run

def run_one_seed(
    cell: str,
    rho: float,
    alpha_b: float,
    seed: int,
    N: int = N_FINAL,
) -> Dict:
    """
    Twin simulation + kappa_LATE_true for one (cell, seed).

    Returns a flat dict of all scalar outputs.
    """
    rng = np.random.default_rng(seed)

    # Step 1: Draw exogenous noise (shared across all counterfactuals)
    U     = rng.standard_normal(N)           # latent risk preference
    eps_b = rng.normal(0.0, STD_EPS_B, N)
    eps_v = rng.normal(0.0, STD_EPS_V, N)
    # eps_y cancels in NIE/NDE differences; included only for Y generation

    v_base = 50.0 + 10.0 * U + eps_v

    # Step 2: P_T(Z=z) for z in {0, 1}
    P_T_Z0 = sigmoid(LOGIT_P0 + rho * U + IV_STR * (0.0 - 0.5))
    P_T_Z1 = sigmoid(LOGIT_P0 + rho * U + IV_STR * (1.0 - 0.5))
    # Sanity: P_T(Z=1) > P_T(Z=0) always (IV_STR > 0 => monotonicity)
    assert np.all(P_T_Z1 > P_T_Z0), "Monotonicity violated (should never happen)"

    # Step 3–4: Twin treatment draws using one latent Uniform per individual
    U_T   = rng.uniform(0.0, 1.0, N)      # shared potential-treatment variate
    T_Z0  = (U_T <= P_T_Z0).astype(float)  # T_i under Z=0
    T_Z1  = (U_T <= P_T_Z1).astype(float)  # T_i under Z=1

    # Step 5: Identify compliers
    is_complier = (T_Z1 > T_Z0)           # bool mask, shape (N,)
    n_complier  = int(is_complier.sum())
    complier_ratio = float(n_complier / N)

    if n_complier < 1000:
        raise RuntimeError(
            f"Too few compliers: {n_complier} in cell={cell} seed={seed}"
        )

    # E[U|complier], SD[U|complier], E[U|population]
    EU_complier  = float(np.mean(U[is_complier]))
    SDU_complier = float(np.std(U[is_complier], ddof=1))
    EU_pop       = float(np.mean(U))   # ~= 0 by construction

    # Step 6: Structural counterfactuals for ALL individuals
    # (same exogenous noise U, eps_b, eps_v for everyone)
    esc_on  = 1.0 - ESC_REDUCTION   # 0.70 when T_out=1
    esc_off = 1.0                   # 1.00 when T_out=0

    b_T0 = 0.0 * alpha_b + GAMMA_B * U + eps_b   # b when T set to 0
    b_T1 = 1.0 * alpha_b + GAMMA_B * U + eps_b   # b when T set to 1

    # Y(T_out, b_in): eps_y cancels in NIE/NDE differences
    def Y_cf(T_out: float, b_in: np.ndarray) -> np.ndarray:
        esc = 1.0 - ESC_REDUCTION * T_out
        dv  = (v_base + SPEED_MULT * b_in) * esc
        return (dv / 100.0) ** 4 + BETA_U * U

    Y_11 = Y_cf(1.0, b_T1)   # Y(T_out=1, b(1))
    Y_10 = Y_cf(1.0, b_T0)   # Y(T_out=1, b(0))
    Y_00 = Y_cf(0.0, b_T0)   # Y(T_out=0, b(0))

    # -- Population-level NIE / NDE / kappa (true structural) --
    NIE_pop   = float(np.mean(Y_11 - Y_10))
    NDE_pop   = float(np.mean(Y_10 - Y_00))
    assert NDE_pop < 0, f"NDE_pop must be negative, got {NDE_pop:.8f}"
    kappa_pop = NIE_pop / abs(NDE_pop)

    # -- Complier-conditional NIE_LATE / NDE_LATE / kappa_LATE_true --
    NIE_LATE   = float(np.mean((Y_11 - Y_10)[is_complier]))
    NDE_LATE   = float(np.mean((Y_10 - Y_00)[is_complier]))
    assert NDE_LATE < 0, f"NDE_LATE must be negative, got {NDE_LATE:.8f}"
    kappa_LATE = float(NIE_LATE / abs(NDE_LATE))

    # Wald IV region (UNCONDITIONAL on C=1)
    # Binary Z ~ Bernoulli(0.5), independent of (U, eps)
    # Re-draw Z here (not reusing U_T draws) for independence
    Z = rng.binomial(1, 0.5, N).astype(float)

    # T_IV: use same rho-driven P_T but with IV additive shift
    P_T_IV = sigmoid(LOGIT_P0 + rho * U + IV_STR * (Z - 0.5))
    T_IV   = rng.binomial(1, P_T_IV).astype(float)

    b_IV     = alpha_b * T_IV + GAMMA_B * U + eps_b
    v_act_IV = v_base + SPEED_MULT * b_IV
    dv_IV    = v_act_IV * (1.0 - ESC_REDUCTION * T_IV)
    eps_y_iv = rng.normal(0.0, STD_EPS_Y, N)
    Y_IV     = (dv_IV / 100.0) ** 4 + BETA_U * U + eps_y_iv

    wald_uncond = _wald_region(Y_IV, T_IV, Z, NDE_pop)

    # Coverage of kappa_LATE_true (NEW HC_T2(b) criterion)
    kappa_L_uc = wald_uncond["kappa_L"]
    kappa_U_uc = wald_uncond["kappa_U"]
    coverage_LATE_uncond = bool(kappa_L_uc <= kappa_LATE <= kappa_U_uc)
    width_uc             = wald_uncond["IV_width"]

    # Also check coverage of population kappa (old criterion, kept for comparison)
    coverage_pop_uncond  = bool(kappa_L_uc <= kappa_pop <= kappa_U_uc)

    # Wald IV region CONDITIONAL on C=1
    # C_IV depends on T_IV, U, b_IV (collider structure)
    P_crash_IV = sigmoid(ETA_0 + ETA_T * T_IV + ETA_U * U + ETA_B * b_IV)
    C_IV       = rng.binomial(1, P_crash_IV).astype(float)
    mask_C1    = (C_IV == 1)

    n_C1 = int(mask_C1.sum())
    crash_rate = float(n_C1 / N)

    if n_C1 < 5000:
        raise RuntimeError(
            f"Too few C=1 records: {n_C1} in cell={cell} seed={seed}"
        )

    Y_C1   = Y_IV[mask_C1]
    T_C1   = T_IV[mask_C1]
    Z_C1   = Z[mask_C1]

    wald_C1 = _wald_region(Y_C1, T_C1, Z_C1, NDE_pop)

    kappa_L_c1 = wald_C1["kappa_L"]
    kappa_U_c1 = wald_C1["kappa_U"]
    coverage_LATE_C1 = bool(kappa_L_c1 <= kappa_LATE <= kappa_U_c1)
    coverage_pop_C1  = bool(kappa_L_c1 <= kappa_pop  <= kappa_U_c1)
    width_c1         = wald_C1["IV_width"]

    # alpha_b(complier) vs alpha_b(population) for HE5
    # In this SCM alpha_b is fixed (structural, not individual-varying).
    # The effective compensation per individual = alpha_b * T + gamma_b * U + eps_b.
    # The T-induced part = alpha_b * T is the same for all.
    # What DOES differ: E[v_base|complier] vs E[v_base|population]
    # This drives kappa_LATE != kappa_population.
    # We report E[U|complier] as the key quantity for HE5.
    # alpha_b(complier) = alpha_b (structural, constant) -- documented explicitly.
    EU_diff_complier_pop = float(EU_complier - EU_pop)
    # Effective kappa-relevant "alpha_b" proxy = speed increase per E[U]
    # since kappa ~ f(alpha_b, E[U|stratum]):
    # complier kappa > pop kappa because E[U|complier] > E[U|pop]
    # The delta_alpha concept in HE5 = kappa_LATE_true - kappa_population
    delta_kappa_LATE_pop = float(kappa_LATE - kappa_pop)

    return {
        "cell"                    : cell,
        "rho"                     : rho,
        "alpha_b"                 : alpha_b,
        "seed"                    : seed,
        "N"                       : N,
        # Complier statistics
        "complier_ratio"          : complier_ratio,
        "n_complier"              : n_complier,
        "EU_complier"             : EU_complier,
        "SDU_complier"            : SDU_complier,
        "EU_population"           : EU_pop,
        "EU_diff_complier_pop"    : EU_diff_complier_pop,
        # Population-level true kappa (structural)
        "NIE_pop"                 : NIE_pop,
        "NDE_pop"                 : NDE_pop,
        "kappa_pop"               : kappa_pop,
        # Complier LATE (twin simulation)
        "NIE_LATE"                : NIE_LATE,
        "NDE_LATE"                : NDE_LATE,
        "kappa_LATE_true"         : kappa_LATE,
        "delta_kappa_LATE_pop"    : delta_kappa_LATE_pop,
        # IV Wald UNCONDITIONAL on C=1
        "kappa_IV_uncond"         : wald_uncond["kappa_IV"],
        "kappa_L_uncond"          : wald_uncond["kappa_L"],
        "kappa_U_uncond"          : wald_uncond["kappa_U"],
        "IV_width_uncond"         : wald_uncond["IV_width"],
        "FS_uncond"               : wald_uncond["FS"],
        "coverage_LATE_uncond"    : coverage_LATE_uncond,
        "coverage_pop_uncond"     : coverage_pop_uncond,
        # IV Wald CONDITIONAL on C=1
        "kappa_IV_C1"             : wald_C1["kappa_IV"],
        "kappa_L_C1"              : wald_C1["kappa_L"],
        "kappa_U_C1"              : wald_C1["kappa_U"],
        "IV_width_C1"             : wald_C1["IV_width"],
        "FS_C1"                   : wald_C1["FS"],
        "coverage_LATE_C1"        : coverage_LATE_C1,
        "coverage_pop_C1"         : coverage_pop_C1,
        "crash_rate"              : crash_rate,
        "n_C1"                    : n_C1,
        # HC_T2(b) new criterion: coverage of kappa_LATE_true + width <= 0.6
        "new_HC_T2b_uncond"       : bool(coverage_LATE_uncond and width_uc <= 0.6),
        "new_HC_T2b_C1"           : bool(coverage_LATE_C1    and width_c1 <= 0.6),
    }


# Aggregate across seeds for one cell

def aggregate_cell(per_seed: List[Dict]) -> Dict:
    """Mean +/- std (ddof=1) across 5 seeds."""
    def ms(key: str):
        vals = [r[key] for r in per_seed]
        return float(np.mean(vals)), float(np.std(vals, ddof=1))

    def fall(key: str):
        return all(r[key] for r in per_seed)

    return {
        "cell"                        : per_seed[0]["cell"],
        "rho"                         : per_seed[0]["rho"],
        "alpha_b"                     : per_seed[0]["alpha_b"],
        "N_per_seed"                  : per_seed[0]["N"],
        # Complier
        "complier_ratio_mean"         : ms("complier_ratio")[0],
        "complier_ratio_std"          : ms("complier_ratio")[1],
        "EU_complier_mean"            : ms("EU_complier")[0],
        "EU_complier_std"             : ms("EU_complier")[1],
        "SDU_complier_mean"           : ms("SDU_complier")[0],
        "EU_pop_mean"                 : ms("EU_population")[0],
        "EU_diff_complier_pop_mean"   : ms("EU_diff_complier_pop")[0],
        # Population kappa
        "NIE_pop_mean"                : ms("NIE_pop")[0],
        "NIE_pop_std"                 : ms("NIE_pop")[1],
        "NDE_pop_mean"                : ms("NDE_pop")[0],
        "kappa_pop_mean"              : ms("kappa_pop")[0],
        "kappa_pop_std"               : ms("kappa_pop")[1],
        # Complier LATE
        "NIE_LATE_mean"               : ms("NIE_LATE")[0],
        "NIE_LATE_std"                : ms("NIE_LATE")[1],
        "NDE_LATE_mean"               : ms("NDE_LATE")[0],
        "NDE_LATE_std"                : ms("NDE_LATE")[1],
        "kappa_LATE_true_mean"        : ms("kappa_LATE_true")[0],
        "kappa_LATE_true_std"         : ms("kappa_LATE_true")[1],
        "delta_kappa_LATE_pop_mean"   : ms("delta_kappa_LATE_pop")[0],
        # IV uncond
        "kappa_IV_uncond_mean"        : ms("kappa_IV_uncond")[0],
        "kappa_IV_uncond_std"         : ms("kappa_IV_uncond")[1],
        "kappa_L_uncond_mean"         : ms("kappa_L_uncond")[0],
        "kappa_U_uncond_mean"         : ms("kappa_U_uncond")[0],
        "IV_width_uncond_mean"        : ms("IV_width_uncond")[0],
        "FS_uncond_mean"              : ms("FS_uncond")[0],
        "coverage_LATE_uncond_all"    : fall("coverage_LATE_uncond"),
        "coverage_pop_uncond_all"     : fall("coverage_pop_uncond"),
        # IV C=1
        "kappa_IV_C1_mean"            : ms("kappa_IV_C1")[0],
        "kappa_IV_C1_std"             : ms("kappa_IV_C1")[1],
        "kappa_L_C1_mean"             : ms("kappa_L_C1")[0],
        "kappa_U_C1_mean"             : ms("kappa_U_C1")[0],
        "IV_width_C1_mean"            : ms("IV_width_C1")[0],
        "FS_C1_mean"                  : ms("FS_C1")[0],
        "coverage_LATE_C1_all"        : fall("coverage_LATE_C1"),
        "coverage_pop_C1_all"         : fall("coverage_pop_C1"),
        "crash_rate_mean"             : ms("crash_rate")[0],
        # HC_T2(b)
        "new_HC_T2b_uncond_all"       : fall("new_HC_T2b_uncond"),
        "new_HC_T2b_C1_all"           : fall("new_HC_T2b_C1"),
        # Per-seed records (embedded)
        "per_seed"                    : per_seed,
    }


# Main

def get_git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(BASE_DIR), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10
        )
        return result.stdout.strip() if result.returncode == 0 else "unavailable"
    except Exception:
        return "unavailable"


def main() -> None:
    t_start = datetime.now()
    git_hash = get_git_commit()

    print("=" * 72)
    print("kappa_LATE_true Calibration (Twin Simulation) — Tier 2 Cells")
    print(f"Date      : {t_start.isoformat()}")
    print(f"Python    : {sys.version}")
    print(f"NumPy     : {np.__version__}")
    print(f"SciPy     : {scipy.__version__}")
    print(f"Pandas    : {pd.__version__}")
    print(f"Host      : {socket.gethostname()}  OS: {platform.platform()}")
    print(f"git commit: {git_hash}")
    print(f"Seeds     : {SEEDS}")
    print(f"N/seed    : {N_FINAL:,}")
    print(f"GPU       : none (CPU-only, DRAFT §4.3.3 rule 5)")
    print("=" * 72)
    print()
    print("Frozen SCM params (DRAFT §4.2):")
    print(f"  LOGIT_P0={LOGIT_P0:.5f}  gamma_b={GAMMA_B}  "
          f"std_eps_b={STD_EPS_B}  std_eps_v={STD_EPS_V}")
    print(f"  beta_u={BETA_U}  ESC_reduction={ESC_REDUCTION}  "
          f"speed_mult={SPEED_MULT}  IV_STR={IV_STR}")
    print(f"  eta=({ETA_0}, {ETA_T}, {ETA_U}, {ETA_B})")
    print()
    print("TWIN SIMULATION: each individual gets T(Z=0) and T(Z=1) from")
    print("  the SAME latent U_T~Uniform(0,1). Complier = T(Z=1)>T(Z=0).")
    print()

    all_per_seed:     List[Dict] = []
    cell_aggregates:  List[Dict] = []

    for spec in TIER2_CELLS:
        cell    = spec["cell"]
        rho     = spec["rho"]
        alpha_b = spec["alpha_b"]

        print(f"\n{'='*65}")
        print(f"Cell {cell}: rho={rho}, alpha_b={alpha_b}")
        print(f"{'='*65}")

        per_seed: List[Dict] = []
        for seed in SEEDS:
            r = run_one_seed(cell, rho, alpha_b, seed)
            per_seed.append(r)
            print(
                f"  seed={seed:4d} | "
                f"complier={r['complier_ratio']:.4f} "
                f"E[U|comp]={r['EU_complier']:+.4f} "
                f"kappa_pop={r['kappa_pop']:.5f} "
                f"kappa_LATE={r['kappa_LATE_true']:+.5f} "
                f"delta={r['delta_kappa_LATE_pop']:+.5f}"
            )
            print(
                f"           | "
                f"kappa_IV_uc={r['kappa_IV_uncond']:.4f} "
                f"[{r['kappa_L_uncond']:.4f},{r['kappa_U_uncond']:.4f}] "
                f"cov_LATE={'Y' if r['coverage_LATE_uncond'] else 'N'} "
                f"cov_pop={'Y' if r['coverage_pop_uncond'] else 'N'} "
                f"w={r['IV_width_uncond']:.4f}"
            )
            print(
                f"           | "
                f"kappa_IV_C1={r['kappa_IV_C1']:.4f} "
                f"[{r['kappa_L_C1']:.4f},{r['kappa_U_C1']:.4f}] "
                f"cov_LATE={'Y' if r['coverage_LATE_C1'] else 'N'} "
                f"cov_pop={'Y' if r['coverage_pop_C1'] else 'N'} "
                f"w={r['IV_width_C1']:.4f} "
                f"crash={r['crash_rate']:.3f}"
            )

        all_per_seed.extend(per_seed)
        agg = aggregate_cell(per_seed)
        cell_aggregates.append(agg)

        print(f"\n  [Aggregate {cell}]")
        print(f"    complier_ratio = {agg['complier_ratio_mean']:.4f} "
              f"+/- {agg['complier_ratio_std']:.2e}")
        print(f"    E[U|complier]  = {agg['EU_complier_mean']:+.4f} "
              f"+/- {agg['EU_complier_std']:.2e}   "
              f"(pop: {agg['EU_pop_mean']:+.4f}  "
              f"delta: {agg['EU_diff_complier_pop_mean']:+.4f})")
        print(f"    kappa_pop      = {agg['kappa_pop_mean']:.6f} "
              f"+/- {agg['kappa_pop_std']:.2e}")
        print(f"    NIE_LATE       = {agg['NIE_LATE_mean']:+.6f} "
              f"+/- {agg['NIE_LATE_std']:.2e}")
        print(f"    NDE_LATE       = {agg['NDE_LATE_mean']:+.6f} "
              f"+/- {agg['NDE_LATE_std']:.2e}")
        print(f"    kappa_LATE_true= {agg['kappa_LATE_true_mean']:.6f} "
              f"+/- {agg['kappa_LATE_true_std']:.2e}  "
              f"(delta from pop: {agg['delta_kappa_LATE_pop_mean']:+.4f})")
        print(f"    kappa_IV_uncond= {agg['kappa_IV_uncond_mean']:.4f}  "
              f"[{agg['kappa_L_uncond_mean']:.4f},{agg['kappa_U_uncond_mean']:.4f}]  "
              f"w={agg['IV_width_uncond_mean']:.4f}  "
              f"cov_LATE={'ALL PASS' if agg['coverage_LATE_uncond_all'] else 'SOME FAIL'}  "
              f"cov_pop={'ALL PASS' if agg['coverage_pop_uncond_all'] else 'SOME FAIL'}")
        print(f"    kappa_IV_C1    = {agg['kappa_IV_C1_mean']:.4f}  "
              f"[{agg['kappa_L_C1_mean']:.4f},{agg['kappa_U_C1_mean']:.4f}]  "
              f"w={agg['IV_width_C1_mean']:.4f}  "
              f"cov_LATE={'ALL PASS' if agg['coverage_LATE_C1_all'] else 'SOME FAIL'}  "
              f"cov_pop={'ALL PASS' if agg['coverage_pop_C1_all'] else 'SOME FAIL'}")
        print(f"    new_HC_T2b_uncond: {'ALL PASS' if agg['new_HC_T2b_uncond_all'] else 'SOME FAIL'}")
        print(f"    new_HC_T2b_C1   : {'ALL PASS' if agg['new_HC_T2b_C1_all'] else 'SOME FAIL'}")

    # -----------------------------------------------------------------------
    # Save per-seed CSV
    # -----------------------------------------------------------------------
    df = pd.DataFrame(all_per_seed)
    # Flatten bool columns for CSV
    for col in df.select_dtypes("bool").columns:
        df[col] = df[col].astype(int)

    csv_path = RESULTS_DIR / "kappa_LATE_true_per_cell.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nPer-seed CSV saved: {csv_path}")

    # -----------------------------------------------------------------------
    # Save aggregate JSON with env_log
    # -----------------------------------------------------------------------
    env_log = {
        "computation_type" : "kappa_LATE_true_twin_simulation_Tier2",
        "date"             : t_start.isoformat(),
        "python_version"   : sys.version,
        "numpy_version"    : np.__version__,
        "scipy_version"    : scipy.__version__,
        "pandas_version"   : pd.__version__,
        "hostname"         : socket.gethostname(),
        "platform"         : platform.platform(),
        "gpu"              : "none (CPU-only, DRAFT §4.3.3 rule 5)",
        "git_commit"       : git_hash,
    }

    output_json = {
        "computation_type"    : "kappa_LATE_true_twin_simulation_Tier2",
        "description"         : (
            "Twin simulation to compute kappa_LATE_true (complier-conditional true kappa) "
            "for Tier 2 cells T2-a/b/c/d. "
            "New HC_T2(b) criterion: IV Wald 95% CI covers kappa_LATE_true (not pop kappa). "
            "Also checks collider conditioning (C=1) effect on Wald estimator. "
            "N=10^6 x 5 seeds, CPU only. "
            "Ref: preregistration_DRAFT_2026-07-10.md §5.1 HC_T2(b)."
        ),
        "preregistration"     : str(BASE_DIR / "preregistration_DRAFT_2026-07-10.md"),
        "frozen_scm_params"   : {
            "LOGIT_P0"           : LOGIT_P0,
            "gamma_b"            : GAMMA_B,
            "std_eps_b"          : STD_EPS_B,
            "std_eps_v"          : STD_EPS_V,
            "beta_u"             : BETA_U,
            "std_eps_y"          : STD_EPS_Y,
            "ESC_reduction"      : ESC_REDUCTION,
            "speed_mult"         : SPEED_MULT,
            "eta_0"              : ETA_0,
            "eta_T"              : ETA_T,
            "eta_U"              : ETA_U,
            "eta_b"              : ETA_B,
            "IV_STR"             : IV_STR,
        },
        "seeds"               : SEEDS,
        "N_per_seed"          : N_FINAL,
        "cell_aggregates"     : cell_aggregates,
        "env_log"             : env_log,
    }

    json_path = RESULTS_DIR / "kappa_LATE_true_aggregate.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output_json, f, indent=2, ensure_ascii=False, default=str)
    print(f"Aggregate JSON saved: {json_path}")

    # -----------------------------------------------------------------------
    # Summary table
    # -----------------------------------------------------------------------
    t_end   = datetime.now()
    elapsed = (t_end - t_start).total_seconds()

    print("\n" + "=" * 80)
    print("SUMMARY TABLE 1: complier statistics and kappa_LATE_true")
    print("=" * 80)
    hdr = (f"{'Cell':<5} {'rho':>4} {'alpha':>7} {'comp%':>6} "
           f"{'E[U|comp]':>10} {'kappa_pop':>10} {'kappa_LATE':>11} {'delta':>8}")
    print(hdr)
    print("-" * 65)
    for agg in cell_aggregates:
        print(
            f"{agg['cell']:<5} {agg['rho']:>4.1f} {agg['alpha_b']:>7.3f} "
            f"{agg['complier_ratio_mean']*100:>6.2f}% "
            f"{agg['EU_complier_mean']:>+10.4f} "
            f"{agg['kappa_pop_mean']:>10.5f} "
            f"{agg['kappa_LATE_true_mean']:>11.5f} "
            f"{agg['delta_kappa_LATE_pop_mean']:>+8.4f}"
        )

    print("\n" + "=" * 80)
    print("SUMMARY TABLE 2: new HC_T2(b) — IV coverage of kappa_LATE_true")
    print("=" * 80)
    hdr2 = (f"{'Cell':<5} {'kappa_LATE':>11} "
            f"{'kappa_IV_uc':>11} {'[L_uc,U_uc]':>20} {'w_uc':>6} {'cov_LATE_uc':>12} "
            f"{'kappa_IV_C1':>11} {'[L_c1,U_c1]':>20} {'w_c1':>6} {'cov_LATE_c1':>12} "
            f"{'HC_T2b_uc':>10} {'HC_T2b_C1':>10}")
    print(hdr2)
    print("-" * 130)
    for agg in cell_aggregates:
        print(
            f"{agg['cell']:<5} "
            f"{agg['kappa_LATE_true_mean']:>11.5f} "
            f"{agg['kappa_IV_uncond_mean']:>11.4f} "
            f"[{agg['kappa_L_uncond_mean']:.4f},{agg['kappa_U_uncond_mean']:.4f}]  "
            f"{agg['IV_width_uncond_mean']:>6.4f} "
            f"{'ALL PASS' if agg['coverage_LATE_uncond_all'] else 'SOME FAIL':>12} "
            f"{agg['kappa_IV_C1_mean']:>11.4f} "
            f"[{agg['kappa_L_C1_mean']:.4f},{agg['kappa_U_C1_mean']:.4f}]  "
            f"{agg['IV_width_C1_mean']:>6.4f} "
            f"{'ALL PASS' if agg['coverage_LATE_C1_all'] else 'SOME FAIL':>12} "
            f"{'PASS' if agg['new_HC_T2b_uncond_all'] else 'FAIL':>10} "
            f"{'PASS' if agg['new_HC_T2b_C1_all'] else 'FAIL':>10}"
        )

    print("\n" + "=" * 80)
    print("SUMMARY TABLE 3: HE5 reference — alpha_b(complier) vs population")
    print("(In this SCM alpha_b is fixed. U-shift is the source of LATE>ATE.)")
    print("=" * 80)
    hdr3 = (f"{'Cell':<5} {'alpha_b':>7} "
            f"{'E[U|pop]':>9} {'E[U|comp]':>10} {'EU_delta':>9} "
            f"{'kappa_pop':>10} {'kappa_LATE':>11} {'delta_kappa':>12}")
    print(hdr3)
    print("-" * 78)
    for agg in cell_aggregates:
        print(
            f"{agg['cell']:<5} {agg['alpha_b']:>7.3f} "
            f"{agg['EU_pop_mean']:>+9.4f} "
            f"{agg['EU_complier_mean']:>+10.4f} "
            f"{agg['EU_diff_complier_pop_mean']:>+9.4f} "
            f"{agg['kappa_pop_mean']:>10.5f} "
            f"{agg['kappa_LATE_true_mean']:>11.5f} "
            f"{agg['delta_kappa_LATE_pop_mean']:>+12.5f}"
        )

    print(f"\nElapsed : {elapsed:.1f} s")
    print(f"git hash: {git_hash}")
    print(f"\nOutputs:")
    print(f"  {csv_path}")
    print(f"  {json_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
