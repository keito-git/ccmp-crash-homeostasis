"""
Tier-2 hierarchy regime calibration (T2-a, T2-b, T2-c, T2-d).
Computes true NIE/NDE/kappa via twin simulation, quantifies naive estimator bias under confounding,
and computes IV identification region [kappa_L, kappa_U] via Wald + delta-method CI.
Outputs: results/hierarchy_regime_per_cell.csv, results/hierarchy_regime_aggregate.json
"""

import json
import os
import platform
import socket
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import scipy

# Paths
BASE_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Frozen SCM parameters
LOGIT_P0      = float(np.log(0.55 / 0.45))  # logit(0.55) ~= 0.20067
GAMMA_B       = 0.50
STD_EPS_B     = 0.30
STD_EPS_V     = 5.00
BETA_U        = 0.10
STD_EPS_Y     = 0.05
ESC_REDUCTION = 0.30
SPEED_MULT    = 5.0
ETA_0 = -2.0
ETA_T = -0.5
ETA_U =  0.3
ETA_B =  0.2

SEEDS   = [42, 123, 456, 789, 2026]
N_FINAL = 1_000_000

# IV instrument strength (shift in logit-scale; gives ~15-20% compliance)
IV_STR = 1.5

# Tier 2 cells: (cell, rho, alpha_b)
TIER2_CELLS: List[Dict] = [
    {"cell": "T2-a", "rho": 0.5, "alpha_b": 0.3},
    {"cell": "T2-b", "rho": 1.0, "alpha_b": 0.3},
    {"cell": "T2-c", "rho": 0.5, "alpha_b": 3.972},
    {"cell": "T2-d", "rho": 1.0, "alpha_b": 3.972},
]


# Helper

def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )


# Core Monte Carlo: one seed

def run_one_seed(
    cell: str,
    rho: float,
    alpha_b: float,
    seed: int,
    N: int = N_FINAL,
) -> Dict:
    """
    Run Monte Carlo for one (cell, rho, alpha_b, seed) combination.

    Returns a dict with structural quantities, naive bias, and IV bounds.
    """
    rng = np.random.default_rng(seed)

    # Exogenous noise (shared across counterfactuals to fix noise)
    U     = rng.standard_normal(N)
    eps_b = rng.normal(0.0, STD_EPS_B, N)
    eps_v = rng.normal(0.0, STD_EPS_V, N)
    # eps_y cancels in NIE and NDE differences; omit for efficiency

    v_base = 50.0 + 10.0 * U + eps_v

    # Observed T (with U->T confounding controlled by rho)
    P_T_obs = sigmoid(LOGIT_P0 + rho * U)
    T_obs   = rng.binomial(1, P_T_obs).astype(float)

    # Empirical corr(T, U)
    corr_TU = float(np.corrcoef(T_obs, U)[0, 1])

    # Empirical P(T=1) -- should be ~0.55 but varies with rho due to sigmoid
    P_T_empirical = float(T_obs.mean())

    # TRUE NIE / NDE (structural counterfactuals, INDEPENDENT of rho)
    # Counterfactual mediators: b(t) = alpha_b*t + gamma_b*U + eps_b
    b_T0 = 0.0 * alpha_b + GAMMA_B * U + eps_b  # b when T set to 0
    b_T1 = 1.0 * alpha_b + GAMMA_B * U + eps_b  # b when T set to 1

    esc_on  = 1.0 - ESC_REDUCTION   # 0.70 when T=1
    esc_off = 1.0                   # 1.00 when T=0

    # NIE = E[Y(T_out=1, b=b(1)) - Y(T_out=1, b=b(0))]
    # Y(t_out, b_in) = ((v_base + speed_mult*b_in)*(1-esc*t_out)/100)^4 + ...
    # beta_u*U and eps_y cancel in the difference
    dv_11  = (v_base + SPEED_MULT * b_T1) * esc_on
    dv_10  = (v_base + SPEED_MULT * b_T0) * esc_on
    NIE    = float(np.mean((dv_11 / 100.0) ** 4 - (dv_10 / 100.0) ** 4))

    # NDE = E[Y(T_out=1, b=b(0)) - Y(T_out=0, b=b(0))]
    dv_01  = (v_base + SPEED_MULT * b_T0) * esc_on
    dv_00  = (v_base + SPEED_MULT * b_T0) * esc_off
    NDE    = float(np.mean((dv_01 / 100.0) ** 4 - (dv_00 / 100.0) ** 4))

    assert NDE < 0, f"NDE must be negative (engineering benefit), got {NDE:.6f}"
    kappa_true = NIE / abs(NDE)
    ATE_true   = NDE + NIE

    # NAIVE PHYSICS ESTIMATOR (biased under rho > 0)
    # Assumes no confounding: uses observed arm-wise S distributions
    # S_obs = v_base + speed_mult * b_obs
    # NIE_naive = (esc_on^4 / 100^4) * (E[S_obs^4|T=1] - E[S_obs^4|T=0])
    b_obs   = alpha_b * T_obs + GAMMA_B * U + eps_b
    S_obs   = v_base + SPEED_MULT * b_obs

    mask_T1 = T_obs == 1
    mask_T0 = T_obs == 0

    E_S4_T1  = float(np.mean(S_obs[mask_T1] ** 4))
    E_S4_T0  = float(np.mean(S_obs[mask_T0] ** 4))

    # Mean S per arm for reporting
    E_S_T1   = float(np.mean(S_obs[mask_T1]))
    E_S_T0   = float(np.mean(S_obs[mask_T0]))

    NIE_naive = (esc_on ** 4 / 100.0 ** 4) * (E_S4_T1 - E_S4_T0)
    naive_bias     = float(NIE_naive - NIE)
    naive_bias_pct = float(naive_bias / abs(NIE) * 100.0)

    # HC_T2 criterion (a): |bias| >= 20% of |NIE_true|
    criterion_a = bool(abs(naive_bias_pct) >= 20.0)

    # IV IDENTIFICATION REGION
    # Binary IV Z ~ Bernoulli(0.5), independent of (U, eps)
    # P_T_IV = sigmoid(logit_p0 + rho*U + IV_STR*(Z - 0.5))
    # Wald ATE_IV = RF / FS
    # NIE_IV = ATE_IV - NDE_true
    # kappa_IV = NIE_IV / |NDE_true|
    # [kappa_L, kappa_U] = kappa_IV +/- 1.96 * SE_kappa_IV (delta method)
    Z        = rng.binomial(1, 0.5, N).astype(float)
    P_T_IV   = sigmoid(LOGIT_P0 + rho * U + IV_STR * (Z - 0.5))
    T_IV     = rng.binomial(1, P_T_IV).astype(float)

    # Generate outcome Y under IV-assigned treatment
    b_IV     = alpha_b * T_IV + GAMMA_B * U + eps_b
    v_act_IV = v_base + SPEED_MULT * b_IV
    dv_IV    = v_act_IV * (1.0 - ESC_REDUCTION * T_IV)
    Y_IV     = (dv_IV / 100.0) ** 4 + BETA_U * U
    # eps_y omitted (zero-mean noise cancels in expectations)

    mask_Z1 = Z == 1
    mask_Z0 = Z == 0
    n1      = int(mask_Z1.sum())
    n0      = int(mask_Z0.sum())

    E_Y_Z1  = float(np.mean(Y_IV[mask_Z1]))
    E_Y_Z0  = float(np.mean(Y_IV[mask_Z0]))
    E_T_Z1  = float(np.mean(T_IV[mask_Z1]))
    E_T_Z0  = float(np.mean(T_IV[mask_Z0]))

    FS = E_T_Z1 - E_T_Z0   # first stage compliance
    RF = E_Y_Z1 - E_Y_Z0   # reduced form intent-to-treat Y

    if abs(FS) < 1e-6:
        raise ValueError(
            f"Weak IV in cell={cell} rho={rho} alpha_b={alpha_b} seed={seed}: "
            f"first stage={FS:.6f}"
        )

    ATE_IV    = RF / FS
    NIE_IV    = ATE_IV - NDE       # use structural NDE as known physics
    kappa_IV  = NIE_IV / abs(NDE)

    # Delta-method SE for Wald estimator
    var_RF = (np.var(Y_IV[mask_Z1], ddof=1) / n1
              + np.var(Y_IV[mask_Z0], ddof=1) / n0)
    # T is Bernoulli, so Var = p*(1-p)
    var_FS = (E_T_Z1 * (1.0 - E_T_Z1) / n1
              + E_T_Z0 * (1.0 - E_T_Z0) / n0)

    SE_Wald     = float(np.sqrt(max(0.0, var_RF / FS ** 2
                                    + RF ** 2 * var_FS / FS ** 4)))
    SE_kappa_IV = SE_Wald / abs(NDE)

    kappa_L     = float(kappa_IV - 1.96 * SE_kappa_IV)
    kappa_U     = float(kappa_IV + 1.96 * SE_kappa_IV)
    IV_coverage = bool(kappa_L <= kappa_true <= kappa_U)
    IV_width    = float(kappa_U - kappa_L)

    # HC_T2 criterion (b): coverage AND width <= 0.6
    criterion_b = bool(IV_coverage and IV_width <= 0.6)

    return {
        "cell"             : cell,
        "rho"              : rho,
        "alpha_b"          : alpha_b,
        "seed"             : seed,
        "N"                : N,
        # Observed T statistics
        "P_T_empirical"    : P_T_empirical,
        "corr_TU"          : corr_TU,
        # Structural true quantities
        "NIE_true"         : NIE,
        "NDE_true"         : NDE,
        "ATE_true"         : ATE_true,
        "kappa_true"       : kappa_true,
        # Naive estimator
        "E_S_obs_T1"       : E_S_T1,
        "E_S_obs_T0"       : E_S_T0,
        "NIE_naive"        : NIE_naive,
        "naive_bias"       : naive_bias,
        "naive_bias_pct"   : naive_bias_pct,
        "criterion_a_pass" : criterion_a,
        # IV identification
        "IV_first_stage"   : FS,
        "IV_RF"            : RF,
        "ATE_IV"           : ATE_IV,
        "NIE_IV"           : NIE_IV,
        "kappa_IV"         : kappa_IV,
        "SE_kappa_IV"      : SE_kappa_IV,
        "kappa_L"          : kappa_L,
        "kappa_U"          : kappa_U,
        "IV_coverage"      : IV_coverage,
        "IV_width"         : IV_width,
        "criterion_b_pass" : criterion_b,
    }


# Aggregate across seeds for a single cell

def aggregate_cell(per_seed_results: List[Dict]) -> Dict:
    """Compute mean +/- std across 5 seeds for a cell."""
    def _ms(key: str):
        vals = [r[key] for r in per_seed_results]
        return float(np.mean(vals)), float(np.std(vals, ddof=1))

    NIE_m,  NIE_s  = _ms("NIE_true")
    NDE_m,  NDE_s  = _ms("NDE_true")
    k_m,    k_s    = _ms("kappa_true")
    cTU_m,  cTU_s  = _ms("corr_TU")
    nb_m,   nb_s   = _ms("naive_bias")
    nbp_m,  nbp_s  = _ms("naive_bias_pct")
    kIV_m,  kIV_s  = _ms("kappa_IV")
    kL_m,   _      = _ms("kappa_L")
    kU_m,   _      = _ms("kappa_U")
    w_m,    w_s    = _ms("IV_width")
    FS_m,   FS_s   = _ms("IV_first_stage")

    all_a = all(r["criterion_a_pass"] for r in per_seed_results)
    all_b = all(r["criterion_b_pass"] for r in per_seed_results)

    return {
        "cell"           : per_seed_results[0]["cell"],
        "rho"            : per_seed_results[0]["rho"],
        "alpha_b"        : per_seed_results[0]["alpha_b"],
        "N_per_seed"     : per_seed_results[0]["N"],
        "NIE_true_mean"  : NIE_m, "NIE_true_std" : NIE_s,
        "NDE_true_mean"  : NDE_m, "NDE_true_std" : NDE_s,
        "kappa_true_mean": k_m,   "kappa_true_std": k_s,
        "corr_TU_mean"   : cTU_m, "corr_TU_std"  : cTU_s,
        "naive_bias_mean": nb_m,  "naive_bias_std": nb_s,
        "naive_bias_pct_mean": nbp_m,
        "criterion_a_all_seeds": all_a,
        "kappa_IV_mean"  : kIV_m, "kappa_IV_std" : kIV_s,
        "kappa_L_mean"   : kL_m,
        "kappa_U_mean"   : kU_m,
        "IV_width_mean"  : w_m,   "IV_width_std" : w_s,
        "IV_FS_mean"     : FS_m,  "IV_FS_std"    : FS_s,
        "criterion_b_all_seeds": all_b,
        "per_seed"       : per_seed_results,
    }


# Main

def main() -> None:
    t_start = datetime.now()
    print("=" * 72)
    print("Tier 2 Hierarchy Regime Calibration: T2-a / T2-b / T2-c / T2-d")
    print(f"Date      : {t_start.isoformat()}")
    print(f"Python    : {sys.version}")
    print(f"NumPy     : {np.__version__}")
    print(f"SciPy     : {scipy.__version__}")
    print(f"Pandas    : {pd.__version__}")
    print(f"Host      : {socket.gethostname()}  OS: {platform.platform()}")
    print(f"Seeds     : {SEEDS}")
    print(f"N/seed    : {N_FINAL:,}")
    print(f"GPU       : none (CPU-only, per DRAFT §4.3.3 rule 5)")
    print("=" * 72)
    print()
    print("Frozen SCM common params (DRAFT §4.2):")
    print(f"  logit_P0={LOGIT_P0:.5f}  gamma_b={GAMMA_B}  std_eps_b={STD_EPS_B}")
    print(f"  std_eps_v={STD_EPS_V}    beta_u={BETA_U}    ESC_reduction={ESC_REDUCTION}")
    print(f"  speed_mult={SPEED_MULT}  eta=({ETA_0},{ETA_T},{ETA_U},{ETA_B})")
    print(f"  IV_STR={IV_STR}")
    print()
    print("KEY: True NIE/NDE are STRUCTURAL and do NOT depend on rho.")
    print("     Naive estimator is biased for rho>0.  IV corrects the bias.")
    print()

    all_per_seed: List[Dict] = []
    cell_aggregates: List[Dict] = []

    for spec in TIER2_CELLS:
        cell    = spec["cell"]
        rho     = spec["rho"]
        alpha_b = spec["alpha_b"]

        print(f"\n{'=' * 65}")
        print(f"Cell {cell}: rho={rho}, alpha_b={alpha_b}")
        print(f"{'=' * 65}")

        per_seed: List[Dict] = []
        for seed in SEEDS:
            r = run_one_seed(cell, rho, alpha_b, seed)
            per_seed.append(r)
            print(
                f"  seed={seed:4d} | corr(T,U)={r['corr_TU']:+.4f} | "
                f"NIE_true={r['NIE_true']:+.6f} | kappa_true={r['kappa_true']:.5f} | "
                f"NIE_naive={r['NIE_naive']:+.6f} | bias%={r['naive_bias_pct']:+.1f}% | "
                f"kappa_IV={r['kappa_IV']:.4f} "
                f"[{r['kappa_L']:.4f},{r['kappa_U']:.4f}] "
                f"cov={'Y' if r['IV_coverage'] else 'N'} "
                f"w={r['IV_width']:.4f} "
                f"FS={r['IV_first_stage']:.4f}"
            )

        all_per_seed.extend(per_seed)
        agg = aggregate_cell(per_seed)
        cell_aggregates.append(agg)

        print(f"\n  [Aggregate {cell}]")
        print(f"    NIE_true  = {agg['NIE_true_mean']:+.6f} +/- {agg['NIE_true_std']:.2e}")
        print(f"    NDE_true  = {agg['NDE_true_mean']:+.6f} +/- {agg['NDE_true_std']:.2e}")
        print(f"    kappa_true= {agg['kappa_true_mean']:.6f} +/- {agg['kappa_true_std']:.2e}")
        print(f"    corr(T,U) = {agg['corr_TU_mean']:+.4f} +/- {agg['corr_TU_std']:.2e}")
        print(f"    naive_bias= {agg['naive_bias_mean']:+.6f} +/- {agg['naive_bias_std']:.2e}  "
              f"({agg['naive_bias_pct_mean']:+.1f}% of |NIE_true|)")
        print(f"    HC_T2(a) bias>=20%: {'ALL PASS' if agg['criterion_a_all_seeds'] else 'SOME FAIL'}")
        print(f"    kappa_IV  = {agg['kappa_IV_mean']:.4f} +/- {agg['kappa_IV_std']:.2e}")
        print(f"    IV region = [{agg['kappa_L_mean']:.4f}, {agg['kappa_U_mean']:.4f}]  "
              f"width={agg['IV_width_mean']:.4f}")
        print(f"    HC_T2(b) coverage+width: {'ALL PASS' if agg['criterion_b_all_seeds'] else 'SOME FAIL'}")

    # Save per-seed CSV
    df = pd.DataFrame(all_per_seed)
    csv_path = RESULTS_DIR / "hierarchy_regime_per_cell.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nPer-seed CSV saved: {csv_path}")

    # Build and save aggregate JSON with env_log
    env_log = {
        "computation_type" : "hierarchy_regime_calibration_Tier2",
        "date"             : t_start.isoformat(),
        "python_version"   : sys.version,
        "numpy_version"    : np.__version__,
        "scipy_version"    : scipy.__version__,
        "pandas_version"   : pd.__version__,
        "hostname"         : socket.gethostname(),
        "platform"         : platform.platform(),
        "gpu"              : "none (CPU-only, DRAFT §4.3.3 rule 5)",
        "git_commit"       : "see pilot/ git log",
    }

    output_json = {
        "computation_type"  : "hierarchy_regime_calibration_Tier2",
        "description"       : (
            "Tier 2 (confounded) regime calibration for Cells T2-a/b/c/d as "
            "defined in preregistration_DRAFT_2026-07-10.md §4.5.1. "
            "True NIE/NDE/kappa computed structurally (N=10^6 x 5 seeds, CPU only). "
            "Naive bias and IV identification region also reported."
        ),
        "date"              : t_start.isoformat(),
        "preregistration"   : str(BASE_DIR / "preregistration_DRAFT_2026-07-10.md"),
        "frozen_scm_params" : {
            "logit_P0"           : LOGIT_P0,
            "gamma_b"            : GAMMA_B,
            "std_eps_b"          : STD_EPS_B,
            "std_eps_v"          : STD_EPS_V,
            "beta_u"             : BETA_U,
            "std_eps_y"          : STD_EPS_Y,
            "ESC_delta_v_reduction": ESC_REDUCTION,
            "speed_mult"         : SPEED_MULT,
            "eta_0"              : ETA_0,
            "eta_T"              : ETA_T,
            "eta_U"              : ETA_U,
            "eta_b"              : ETA_B,
        },
        "IV_str"            : IV_STR,
        "seeds"             : SEEDS,
        "N_per_seed"        : N_FINAL,
        "tier2_cells"       : cell_aggregates,
        "env_log"           : env_log,
    }

    json_path = RESULTS_DIR / "hierarchy_regime_aggregate.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output_json, f, indent=2, ensure_ascii=False, default=str)
    print(f"Aggregate JSON saved: {json_path}")

    # Summary table
    t_end   = datetime.now()
    elapsed = (t_end - t_start).total_seconds()

    print("\n" + "=" * 72)
    print("SUMMARY TABLE (for DRAFT §4.5.1 Tier 2 insertion)")
    print("=" * 72)
    hdr = (f"{'Cell':<5} {'rho':<5} {'alpha_b':<8} {'corr(T,U)':<10} "
           f"{'kappa_true':<12} {'NIE_naive_bias%':<16} {'kappa_IV':<9} "
           f"[kappa_L, kappa_U]         {'HC_T2a':<7} {'HC_T2b':<7}")
    print(hdr)
    print("-" * 110)
    for agg in cell_aggregates:
        print(
            f"{agg['cell']:<5} {agg['rho']:<5.1f} {agg['alpha_b']:<8.3f} "
            f"{agg['corr_TU_mean']:<10.4f} {agg['kappa_true_mean']:<12.5f} "
            f"{agg['naive_bias_pct_mean']:<16.1f} {agg['kappa_IV_mean']:<9.4f} "
            f"[{agg['kappa_L_mean']:.4f}, {agg['kappa_U_mean']:.4f}]  "
            f"{'PASS' if agg['criterion_a_all_seeds'] else 'FAIL':<7} "
            f"{'PASS' if agg['criterion_b_all_seeds'] else 'FAIL':<7}"
        )

    print("=" * 72)
    print(f"\nElapsed: {elapsed:.1f} s")
    print(f"Outputs:")
    print(f"  {csv_path}")
    print(f"  {json_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
