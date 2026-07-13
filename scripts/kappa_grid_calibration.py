"""
Kappa grid calibration: binary-search alpha_b for G2-G5 (target kappa in {0.3, 0.5, 0.7, 1.0}).
Computes true NIE/NDE/kappa via N=10^6 x 5-seed Monte Carlo with calibrated alpha_b.
Outputs: results/kappa_grid_per_cell.csv, results/kappa_grid_aggregate.json
"""

import json
import os
import platform
import socket
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import scipy
import pandas as pd

# Paths
BASE_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Frozen SCM parameters (must match true_nie_kappa_monte_carlo.py)
P_T           = 0.55    # P(T=1)
GAMMA_B       = 0.50    # confounding strength U -> b
STD_EPS_B     = 0.30    # std of eps_b
STD_EPS_V     = 5.00    # std of eps_v
BETA_U        = 0.10    # residual confounding U -> Y (cancels in NIE/NDE)
STD_EPS_Y     = 0.05    # std of eps_y (cancels in NIE/NDE)
ESC_REDUCTION = 0.30    # ESC reduces delta_v by 30%
SPEED_MULT    = 5.0     # v_actual = v_base + SPEED_MULT * b  (frozen at 5)
ETA_0         = -2.0
ETA_T         = -0.5
ETA_U         =  0.3
ETA_B         =  0.2

SEEDS    = [42, 123, 456, 789, 2026]
N_FINAL  = 1_000_000   # per seed for final estimates
N_SEARCH = 500_000     # per iteration during binary search (seed=42)

# Target cells (G1 already confirmed; compute G2-G5)
GRID_SPECS: List[Dict] = [
    {"cell": "G2", "target_kappa": 0.30},
    {"cell": "G3", "target_kappa": 0.70},
    {"cell": "G4", "target_kappa": 1.00},   # null cell -- highest priority
    {"cell": "G5", "target_kappa": 1.30},
]

# G1 reference values (alpha_b=0.3)
G1_REF: Dict = {
    "cell"            : "G1",
    "target_kappa"    : 0.033,
    "alpha_b"         : 0.3,
    "speed_shift_km_h": 1.5,
    "kappa_mean"      : 0.033042,
    "kappa_std"       : 6.38e-06,
    "NIE_mean"        : 0.002286,
    "NIE_std"         : 1.32e-06,
    "NDE_mean"        : -0.069183,
    "NDE_std"         : 5.17e-05,
    "per_seed": [
        {"seed":   42, "NIE": 0.00228705, "NDE": -0.06922376, "kappa": 0.033038},
        {"seed":  123, "NIE": 0.00228671, "NDE": -0.06920490, "kappa": 0.033043},
        {"seed":  456, "NIE": 0.00228685, "NDE": -0.06922990, "kappa": 0.033033},
        {"seed":  789, "NIE": 0.00228413, "NDE": -0.06911333, "kappa": 0.033049},
        {"seed": 2026, "NIE": 0.00228494, "NDE": -0.06914454, "kappa": 0.033046},
    ],
}


# Core MC function

def compute_nie_nde_kappa(
    alpha_b: float,
    N: int,
    seed: int,
    speed_mult: float = SPEED_MULT,
) -> Tuple[float, float, float]:
    """
    Compute oracle NIE, NDE, and kappa via Monte Carlo.

    Exogenous noise (U, eps_b, eps_v) is drawn once and shared across
    the two counterfactual arms.  eps_y and beta_u*U cancel in both
    NIE and NDE and are therefore omitted.

    Parameters
    ----------
    alpha_b : float
        Compensation effect T->b (the grid variable).
    N : int
        Number of Monte Carlo samples.
    seed : int
        RNG seed for reproducibility.
    speed_mult : float
        Multiplier linking b to v_actual (default 5, frozen).

    Returns
    -------
    (NIE, NDE, kappa)
    """
    rng = np.random.default_rng(seed)

    # Exogenous noise (shared across counterfactuals)
    U     = rng.standard_normal(N)
    eps_b = rng.normal(0.0, STD_EPS_B, N)
    eps_v = rng.normal(0.0, STD_EPS_V, N)

    v_base = 50.0 + 10.0 * U + eps_v

    # Counterfactual mediators
    b_T0 = 0.0 * alpha_b + GAMMA_B * U + eps_b   # b when T=0
    b_T1 = 1.0 * alpha_b + GAMMA_B * U + eps_b   # b when T=1
    # Sanity: b_T1 - b_T0 == alpha_b for every unit
    assert np.allclose(b_T1 - b_T0, alpha_b, atol=1e-10), \
        "b difference must equal alpha_b"

    esc_on  = 1.0 - ESC_REDUCTION   # 0.70 when T=1
    esc_off = 1.0                   # 1.00 when T=0

    # NIE = E[Y(T_out=1, b=b(1)) - Y(T_out=1, b=b(0))]
    dv_11 = (v_base + speed_mult * b_T1) * esc_on
    dv_10 = (v_base + speed_mult * b_T0) * esc_on
    Y_11  = (dv_11 / 100.0) ** 4
    Y_10  = (dv_10 / 100.0) ** 4
    NIE   = float(np.mean(Y_11 - Y_10))

    # NDE = E[Y(T_out=1, b=b(0)) - Y(T_out=0, b=b(0))]
    dv_01 = (v_base + speed_mult * b_T0) * esc_on
    dv_00 = (v_base + speed_mult * b_T0) * esc_off
    Y_01  = (dv_01 / 100.0) ** 4
    Y_00  = (dv_00 / 100.0) ** 4
    NDE   = float(np.mean(Y_01 - Y_00))

    if NDE >= 0:
        raise ValueError(
            f"NDE should be negative (engineering benefit), got {NDE:.6f} "
            f"at alpha_b={alpha_b}"
        )

    kappa = NIE / abs(NDE)
    return NIE, NDE, kappa


# Binary search

def binary_search_alpha_b(
    target_kappa: float,
    speed_mult: float = SPEED_MULT,
    tol_kappa: float = 0.01,
    n_search: int = N_SEARCH,
    seed_search: int = 42,
    max_iter: int = 80,
) -> Tuple[float, float]:
    """
    Find alpha_b such that kappa(alpha_b) ~= target_kappa ± tol_kappa.

    Monotone relationship: kappa is strictly increasing in alpha_b
    (larger alpha_b -> larger b(1)-b(0) -> larger speed shift -> larger NIE).

    Returns
    -------
    (alpha_b_calibrated, kappa_at_convergence)
    """
    lo = 0.3    # G1 value; kappa ≈ 0.033 < all targets
    hi = 5.0    # initial upper candidate

    # Expand upper bound until kappa > target_kappa
    _, _, k_hi = compute_nie_nde_kappa(hi, n_search, seed_search, speed_mult)
    print(f"  Initial bounds: lo={lo}(k≈0.033)  hi={hi}(k={k_hi:.4f})")
    expansion_count = 0
    while k_hi < target_kappa:
        hi *= 2.0
        expansion_count += 1
        _, _, k_hi = compute_nie_nde_kappa(hi, n_search, seed_search, speed_mult)
        print(f"  Expanded hi={hi:.2f} -> k={k_hi:.4f}")
        if expansion_count > 20:
            raise RuntimeError(
                f"Cannot bracket target_kappa={target_kappa} with alpha_b <= {hi}"
            )

    print(f"  Bracketed: lo={lo:.4f}(k≈0.033)  hi={hi:.4f}(k={k_hi:.4f})  "
          f"target={target_kappa}")

    # Bisection
    mid = 0.0
    for iteration in range(max_iter):
        mid = (lo + hi) / 2.0
        _, _, k_mid = compute_nie_nde_kappa(mid, n_search, seed_search, speed_mult)
        err = k_mid - target_kappa

        print(f"  iter {iteration+1:3d}: alpha_b={mid:.8f}  kappa={k_mid:.8f}  "
              f"err={err:+.6f}")

        if abs(err) <= tol_kappa:
            print(f"  => Converged at alpha_b={mid:.8f} kappa={k_mid:.6f}")
            return mid, k_mid

        if k_mid < target_kappa:
            lo = mid
        else:
            hi = mid

    print(f"  => max_iter ({max_iter}) reached; using alpha_b={mid:.8f} k={k_mid:.6f}")
    return mid, k_mid


# Full-seeds computation

def run_full_seeds(
    alpha_b: float,
    cell: str,
    target_kappa: float,
    speed_mult: float = SPEED_MULT,
) -> List[Dict]:
    """
    Compute NIE/NDE/kappa for N=10^6 per seed across all 5 seeds.

    Returns list of per-seed result dicts.
    """
    results = []
    for seed in SEEDS:
        NIE, NDE, kappa = compute_nie_nde_kappa(alpha_b, N_FINAL, seed, speed_mult)
        ATE = NDE + NIE
        results.append({
            "cell"            : cell,
            "target_kappa"    : target_kappa,
            "alpha_b"         : alpha_b,
            "speed_shift_km_h": speed_mult * alpha_b,
            "speed_mult"      : speed_mult,
            "seed"            : seed,
            "N_total"         : N_FINAL,
            "NIE"             : NIE,
            "NDE"             : NDE,
            "ATE"             : ATE,
            "kappa"           : kappa,
        })
        print(f"    seed={seed:4d} | NIE={NIE:+.8f} | NDE={NDE:+.8f} | "
              f"ATE={ATE:+.8f} | kappa={kappa:.8f}")
    return results


def aggregate_cell(per_seed: List[Dict]) -> Dict:
    """Compute mean ± std across 5 seeds for a single cell."""
    nie_v   = [r["NIE"]   for r in per_seed]
    nde_v   = [r["NDE"]   for r in per_seed]
    ate_v   = [r["ATE"]   for r in per_seed]
    kappa_v = [r["kappa"] for r in per_seed]
    tgt     = per_seed[0]["target_kappa"]

    kappa_mean = float(np.mean(kappa_v))
    within_tol = bool(abs(kappa_mean - tgt) <= 0.02)

    return {
        "cell"              : per_seed[0]["cell"],
        "target_kappa"      : tgt,
        "alpha_b"           : per_seed[0]["alpha_b"],
        "speed_mult"        : per_seed[0]["speed_mult"],
        "speed_shift_km_h"  : per_seed[0]["speed_shift_km_h"],
        "NIE_mean"          : float(np.mean(nie_v)),
        "NIE_std"           : float(np.std(nie_v, ddof=1)),
        "NDE_mean"          : float(np.mean(nde_v)),
        "NDE_std"           : float(np.std(nde_v, ddof=1)),
        "ATE_mean"          : float(np.mean(ate_v)),
        "ATE_std"           : float(np.std(ate_v, ddof=1)),
        "kappa_mean"        : kappa_mean,
        "kappa_std"         : float(np.std(kappa_v, ddof=1)),
        "kappa_within_tol"  : within_tol,
        "kappa_abs_err"     : float(abs(kappa_mean - tgt)),
        "per_seed"          : per_seed,
    }


# Main

def main() -> None:
    t_start = datetime.now()
    print("=" * 70)
    print("kappa Grid Calibration: G2-G5")
    print(f"Date      : {t_start.isoformat()}")
    print(f"Python    : {sys.version}")
    print(f"NumPy     : {np.__version__}")
    print(f"SciPy     : {scipy.__version__}")
    print(f"Pandas    : {pd.__version__}")
    print(f"Host      : {socket.gethostname()}  OS: {platform.platform()}")
    print(f"Seeds     : {SEEDS}")
    print(f"N/seed    : {N_FINAL:,}  (binary-search: {N_SEARCH:,})")
    print(f"GPU       : none (CPU-only, per DRAFT §4.3.3 rule 5)")
    print("=" * 70)
    print()
    print("Frozen SCM common parameters (DRAFT §4.2):")
    print(f"  P(T=1)       = {P_T}")
    print(f"  gamma_b      = {GAMMA_B}")
    print(f"  std(eps_b)   = {STD_EPS_B}")
    print(f"  std(eps_v)   = {STD_EPS_V}")
    print(f"  beta_u       = {BETA_U}  (cancels in NIE/NDE)")
    print(f"  std(eps_y)   = {STD_EPS_Y}  (cancels in NIE/NDE)")
    print(f"  ESC_reduction= {ESC_REDUCTION} (30%)")
    print(f"  speed_mult   = {SPEED_MULT}  (v_actual = v_base + 5*b)")
    print(f"  eta = ({ETA_0}, {ETA_T}, {ETA_U}, {ETA_B})")
    print()
    print("Grid variable: alpha_b (T->b compensation effect)")
    print("Grid cells to compute: G2 (k=0.3) / G3 (k=0.7) / G4 (k=1.0) / G5 (k=1.3)")
    print()

    # Track whether speed_mult was changed
    speed_mult_used  = SPEED_MULT
    speed_mult_note  = "speed_mult=5 unchanged; alpha_b alone achieved all target kappas."

    all_per_seed: List[Dict]   = []
    cell_aggregates: List[Dict] = []

    for spec in GRID_SPECS:
        cell         = spec["cell"]
        target_kappa = spec["target_kappa"]

        print(f"\n{'=' * 65}")
        print(f"Cell {cell}: target kappa = {target_kappa} (±0.02 tolerance)")
        print(f"{'=' * 65}")

        # Step 1: Binary search
        print("\n[1] Binary search for alpha_b ...")
        alpha_b, kappa_search = binary_search_alpha_b(
            target_kappa=target_kappa,
            speed_mult=speed_mult_used,
            tol_kappa=0.01,
            n_search=N_SEARCH,
            seed_search=42,
            max_iter=80,
        )

        # Check if G4 required speed_mult adjustment
        if cell == "G4" and abs(kappa_search - target_kappa) > 0.02:
            print(f"\n  NOTE: alpha_b alone failed to reach kappa=1.0 within tolerance.")
            print(f"  Adjusting speed_mult per §4.3.4 and re-running all cells ...")
            # This branch is kept for robustness but is not expected to trigger
            speed_mult_used = 10.0
            speed_mult_note = (
                f"speed_mult adjusted from 5 to {speed_mult_used} per §4.3.4 "
                f"because alpha_b alone could not reach kappa=1.0. "
                f"All cells re-computed with new speed_mult."
            )
            print(f"  speed_mult changed to {speed_mult_used}. Re-running from G2.")
            all_per_seed  = []
            cell_aggregates = []
            # Re-run from beginning with new speed_mult -- handled by restart logic
            # For simplicity, raise and let user restart; this should not occur.
            raise RuntimeError(
                "speed_mult adjustment needed. "
                "Update SPEED_MULT constant and re-run."
            )

        speed_shift = speed_mult_used * alpha_b
        print(f"\n  => alpha_b = {alpha_b:.8f}   speed_shift = {speed_shift:.4f} km/h   "
              f"kappa_search = {kappa_search:.6f}")

        # Step 2: Full N=10^6 x 5 seeds
        print(f"\n[2] Full MC: N={N_FINAL:,} x {len(SEEDS)} seeds ...")
        per_seed = run_full_seeds(alpha_b, cell, target_kappa, speed_mult_used)
        all_per_seed.extend(per_seed)

        # Step 3: Aggregate
        agg = aggregate_cell(per_seed)
        cell_aggregates.append(agg)

        print(f"\n  [Aggregate {cell}]")
        print(f"    kappa_mean = {agg['kappa_mean']:.8f} ± {agg['kappa_std']:.2e}")
        print(f"    NIE_mean   = {agg['NIE_mean']:+.8f} ± {agg['NIE_std']:.2e}")
        print(f"    NDE_mean   = {agg['NDE_mean']:+.8f} ± {agg['NDE_std']:.2e}")
        print(f"    |kappa - target| = {agg['kappa_abs_err']:.6f}  "
              f"{'<= 0.02 PASS' if agg['kappa_within_tol'] else '> 0.02 FAIL -- check'}")

    # Save CSV
    df = pd.DataFrame(all_per_seed)
    csv_path = RESULTS_DIR / "kappa_grid_per_cell.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nPer-seed CSV saved: {csv_path}")

    # Build and save JSON
    env_log = {
        "computation_type"  : "kappa_grid_calibration_G2_G5",
        "date"              : t_start.isoformat(),
        "python_version"    : sys.version,
        "numpy_version"     : np.__version__,
        "scipy_version"     : scipy.__version__,
        "pandas_version"    : pd.__version__,
        "hostname"          : socket.gethostname(),
        "platform"          : platform.platform(),
        "gpu"               : "none (CPU-only, DRAFT §4.3.3 rule 5)",
    }

    aggregate_json = {
        "computation_type"  : "kappa_grid_calibration_G2_G5",
        "description"       : (
            "Binary-search calibration of alpha_b for kappa grid cells G2-G5 "
            "as specified in preregistration_DRAFT_2026-07-10.md §4.3.3. "
            "True NIE/NDE/kappa computed via N=10^6 x 5 seeds Monte Carlo (CPU only). "
            "G1 values retained verbatim from pilot/results/true_nie_kappa_aggregate.json."
        ),
        "date"              : t_start.isoformat(),
        "preregistration_doc": str(BASE_DIR / "preregistration_DRAFT_2026-07-10.md"),
        "frozen_scm_params" : {
            "P_T"                 : P_T,
            "gamma_b"             : GAMMA_B,
            "std_eps_b"           : STD_EPS_B,
            "std_eps_v"           : STD_EPS_V,
            "beta_u"              : BETA_U,
            "std_eps_y"           : STD_EPS_Y,
            "ESC_delta_v_reduction": ESC_REDUCTION,
            "speed_mult"          : speed_mult_used,
            "eta_0"               : ETA_0,
            "eta_T"               : ETA_T,
            "eta_U"               : ETA_U,
            "eta_b"               : ETA_B,
        },
        "speed_mult_note"   : speed_mult_note,
        "seeds"             : SEEDS,
        "N_per_seed"        : N_FINAL,
        "N_search_per_iter" : N_SEARCH,
        "G1_reference"      : G1_REF,
        "G2_G5_cells"       : cell_aggregates,
        "env_log"           : env_log,
    }

    json_path = RESULTS_DIR / "kappa_grid_aggregate.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(aggregate_json, f, indent=2, ensure_ascii=False)
    print(f"Aggregate JSON saved: {json_path}")

    # Summary table
    t_end = datetime.now()
    elapsed = (t_end - t_start).total_seconds()

    print("\n" + "=" * 70)
    print("FINAL SUMMARY TABLE (for DRAFT §4.5 insertion)")
    print("=" * 70)
    header = (f"{'Cell':<5} {'target_k':<9} {'alpha_b':<12} "
              f"{'shift(km/h)':<12} {'kappa_mean':<12} {'kappa_std':<12} "
              f"{'NIE_mean':<14} {'tol':<6}")
    print(header)
    print("-" * 82)

    # G1 reference
    print(f"{'G1':<5} {G1_REF['target_kappa']:<9.4f} {G1_REF['alpha_b']:<12.4f} "
          f"{G1_REF['speed_shift_km_h']:<12.2f} {G1_REF['kappa_mean']:<12.6f} "
          f"{G1_REF['kappa_std']:<12.2e} {G1_REF['NIE_mean']:<14.8f} {'(ref)':<6}")

    all_ok = True
    for agg in cell_aggregates:
        ok_str = "OK" if agg["kappa_within_tol"] else "FAIL"
        if not agg["kappa_within_tol"]:
            all_ok = False
        print(f"{agg['cell']:<5} {agg['target_kappa']:<9.4f} {agg['alpha_b']:<12.8f} "
              f"{agg['speed_shift_km_h']:<12.4f} {agg['kappa_mean']:<12.8f} "
              f"{agg['kappa_std']:<12.2e} {agg['NIE_mean']:<14.8f} {ok_str:<6}")

    print("=" * 70)
    print(f"\nAll cells within ±0.02 tolerance: {'YES' if all_ok else 'NO -- see details above'}")
    print(f"Speed multiplier: {speed_mult_used} ({speed_mult_note})")
    print(f"Elapsed time: {elapsed:.1f} s")
    print(f"\nOutputs:")
    print(f"  Per-seed CSV : {csv_path}")
    print(f"  Aggregate JSON: {json_path}")
    print()
    print("Next step: insert computed values into paper Section 4.5")
    print("=" * 70)


if __name__ == "__main__":
    main()
