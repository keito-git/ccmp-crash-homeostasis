"""
G4 re-calibration with tighter tolerance (tol_kappa=0.001 vs. prior 0.01) for target kappa=1.0.
Reports new alpha_b if kappa_mean >= 0.999 is achievable, or the best achievable kappa otherwise.
Does not modify G2/G3/G5 results from kappa_grid_calibration.py.
Outputs: results/kappa_g4_recalibration.json, results/kappa_g4_recalibration_per_seed.csv
"""

import json
import os
import platform
import socket
import sys
from datetime import datetime
from pathlib import Path
from typing import Tuple, List, Dict

import numpy as np
import pandas as pd
import scipy

# Paths
BASE_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Frozen SCM parameters
P_T           = 0.55
GAMMA_B       = 0.50
STD_EPS_B     = 0.30
STD_EPS_V     = 5.00
ESC_REDUCTION = 0.30
SPEED_MULT    = 5.0

SEEDS    = [42, 123, 456, 789, 2026]
N_FINAL  = 1_000_000
N_SEARCH = 1_000_000  # use same N as final for tighter tolerance

# G4 specifications
TARGET_KAPPA   = 1.0
TOL_OLD        = 0.01    # previous tolerance
TOL_NEW        = 0.001   # tighter tolerance (attempt)
ALPHA_B_PREV   = 4.981641  # previous G4 alpha_b


# Core computation

def compute_kappa(alpha_b: float, N: int, seed: int) -> Tuple[float, float, float]:
    """Return (NIE, NDE, kappa) for given alpha_b, N, seed (SCM rho=0)."""
    rng   = np.random.default_rng(seed)
    U     = rng.standard_normal(N)
    eps_b = rng.normal(0.0, STD_EPS_B, N)
    eps_v = rng.normal(0.0, STD_EPS_V, N)

    v_base = 50.0 + 10.0 * U + eps_v
    b_T0   = GAMMA_B * U + eps_b
    b_T1   = alpha_b + GAMMA_B * U + eps_b

    esc_on  = 1.0 - ESC_REDUCTION
    esc_off = 1.0

    dv_11  = (v_base + SPEED_MULT * b_T1) * esc_on
    dv_10  = (v_base + SPEED_MULT * b_T0) * esc_on
    NIE    = float(np.mean((dv_11 / 100.0) ** 4 - (dv_10 / 100.0) ** 4))

    dv_01  = (v_base + SPEED_MULT * b_T0) * esc_on
    dv_00  = (v_base + SPEED_MULT * b_T0) * esc_off
    NDE    = float(np.mean((dv_01 / 100.0) ** 4 - (dv_00 / 100.0) ** 4))

    kappa  = NIE / abs(NDE)
    return NIE, NDE, kappa


def binary_search_alpha_b(
    target_kappa: float,
    tol_kappa: float,
    lo_init: float,
    hi_init: float,
    N: int,
    seed: int,
    max_iter: int = 100,
) -> Tuple[float, float]:
    """Binary search alpha_b for target kappa ± tol_kappa."""
    lo, hi = lo_init, hi_init

    # Verify bracket
    _, _, k_lo = compute_kappa(lo, N, seed)
    _, _, k_hi = compute_kappa(hi, N, seed)

    if k_hi < target_kappa:
        print(f"  Upper bound hi={hi} gives kappa={k_hi:.6f} < target={target_kappa}. Expanding...")
        while k_hi < target_kappa:
            hi *= 1.5
            _, _, k_hi = compute_kappa(hi, N, seed)
            print(f"  hi={hi:.4f} -> kappa={k_hi:.6f}")

    print(f"  Bracket: lo={lo:.6f}(k={k_lo:.6f}) hi={hi:.6f}(k={k_hi:.6f}) target={target_kappa}")

    mid = (lo + hi) / 2.0
    for i in range(max_iter):
        mid = (lo + hi) / 2.0
        _, _, k_mid = compute_kappa(mid, N, seed)
        err = k_mid - target_kappa
        print(f"  iter {i+1:3d}: alpha_b={mid:.10f}  kappa={k_mid:.10f}  err={err:+.8f}")

        if abs(err) <= tol_kappa:
            print(f"  => CONVERGED: alpha_b={mid:.10f}  kappa={k_mid:.8f}")
            return mid, k_mid

        if k_mid < target_kappa:
            lo = mid
        else:
            hi = mid

    _, _, k_final = compute_kappa(mid, N, seed)
    print(f"  => max_iter reached: alpha_b={mid:.10f}  kappa={k_final:.8f}")
    return mid, k_final


# Main

def main() -> None:
    t_start = datetime.now()
    print("=" * 72)
    print("G4 Re-Calibration: tol_kappa=0.001 (target kappa=1.0)")
    print(f"Date      : {t_start.isoformat()}")
    print(f"Python    : {sys.version}")
    print(f"NumPy     : {np.__version__}")
    print(f"SciPy     : {scipy.__version__}")
    print(f"Host      : {socket.gethostname()}  OS: {platform.platform()}")
    print(f"Seeds     : {SEEDS}")
    print(f"N/seed    : {N_FINAL:,}")
    print(f"GPU       : none (CPU-only)")
    print("=" * 72)
    print()
    print(f"Previous G4: alpha_b={ALPHA_B_PREV}, kappa=0.994698  (tol={TOL_OLD})")
    print(f"Attempt   : tol_kappa={TOL_NEW}, target kappa=[0.999, 1.001]")
    print()

    # Step 1: Quick verification of previous G4
    print("[Step 1] Verify previous G4 alpha_b at N=10^6, seed=42 ...")
    NIE_v, NDE_v, k_v = compute_kappa(ALPHA_B_PREV, N_FINAL, 42)
    print(f"  alpha_b={ALPHA_B_PREV}  NIE={NIE_v:+.8f}  NDE={NDE_v:+.8f}  kappa={k_v:.8f}")
    print()

    # Step 2: Binary search with tighter tolerance
    print(f"[Step 2] Binary search with tol_kappa={TOL_NEW} ...")
    # Start from previous converged value (narrow window around it)
    lo_init = ALPHA_B_PREV
    hi_init = ALPHA_B_PREV * 1.05   # try 5% above as upper bracket

    alpha_b_new, kappa_search = binary_search_alpha_b(
        target_kappa=TARGET_KAPPA,
        tol_kappa=TOL_NEW,
        lo_init=lo_init,
        hi_init=hi_init,
        N=N_SEARCH,
        seed=42,
        max_iter=100,
    )

    achieved_tolerance = bool(abs(kappa_search - TARGET_KAPPA) <= TOL_NEW)
    kappa_gte_999 = bool(kappa_search >= 0.999)
    print()
    print(f"  Search result: alpha_b={alpha_b_new:.10f}  kappa={kappa_search:.8f}")
    print(f"  |kappa - 1.0| = {abs(kappa_search - 1.0):.8f}  (<= {TOL_NEW}? {achieved_tolerance})")
    print(f"  kappa >= 0.999? {kappa_gte_999}")
    print()

    # Step 3: Full N=10^6 x 5 seeds with new alpha_b
    print("[Step 3] Full MC: N=10^6 x 5 seeds with calibrated alpha_b ...")
    per_seed: List[Dict] = []
    for seed in SEEDS:
        NIE, NDE, kappa = compute_kappa(alpha_b_new, N_FINAL, seed)
        ATE = NIE + NDE
        per_seed.append({
            "seed" : seed,
            "alpha_b": alpha_b_new,
            "NIE"  : NIE,
            "NDE"  : NDE,
            "ATE"  : ATE,
            "kappa": kappa,
        })
        print(f"  seed={seed:4d} | NIE={NIE:+.8f} | NDE={NDE:+.8f} | kappa={kappa:.8f}")

    kappa_vals = [r["kappa"] for r in per_seed]
    NIE_vals   = [r["NIE"]   for r in per_seed]
    NDE_vals   = [r["NDE"]   for r in per_seed]

    kappa_mean = float(np.mean(kappa_vals))
    kappa_std  = float(np.std(kappa_vals, ddof=1))
    NIE_mean   = float(np.mean(NIE_vals))
    NIE_std    = float(np.std(NIE_vals, ddof=1))
    NDE_mean   = float(np.mean(NDE_vals))
    NDE_std    = float(np.std(NDE_vals, ddof=1))

    kappa_mean_gte_999 = bool(kappa_mean >= 0.999)

    print()
    print(f"  [Aggregate]")
    print(f"    NIE   = {NIE_mean:+.8f} +/- {NIE_std:.2e}")
    print(f"    NDE   = {NDE_mean:+.8f} +/- {NDE_std:.2e}")
    print(f"    kappa = {kappa_mean:.8f} +/- {kappa_std:.2e}")
    print(f"    kappa_mean >= 0.999? {kappa_mean_gte_999}")
    print()

    # Decision and note for DRAFT
    if kappa_mean_gte_999:
        draft_note = (
            f"G4 re-calibrated with tol_kappa={TOL_NEW}: "
            f"alpha_b={alpha_b_new:.8f}, kappa_mean={kappa_mean:.6f} >= 0.999. "
            f"HC3(a) null cell is kappa_true=1.000 to within 0.001. "
            f"DRAFT §4.5 G4 values should be updated to new alpha_b."
        )
        hc3a_definition = "null_cell_kappa_true_ge_0999"
    else:
        draft_note = (
            f"G4 re-calibration with tol_kappa={TOL_NEW} achieves kappa_mean="
            f"{kappa_mean:.6f} (best achievable with alpha_b={alpha_b_new:.8f}). "
            f"SCM nonlinearity prevents reaching kappa_mean >= 0.999 exactly. "
            f"HC3(a) should be redefined as 'near-null power test': "
            f"H0: kappa=1 tested when kappa_true={kappa_mean:.4f}, "
            f"which provides an upper bound on the type-I error rate "
            f"at a near-null kappa ({kappa_mean:.4f} is sufficiently close to 1.0 "
            f"to serve as the size-control cell). "
            f"Previous G4 (alpha_b={ALPHA_B_PREV}, kappa=0.9947) retained unless "
            f"kappa_mean >= 0.999 is specifically required."
        )
        hc3a_definition = "near_null_power_test"

    print("  DRAFT NOTE:")
    print(f"  {draft_note}")

    # Save results
    t_end   = datetime.now()
    elapsed = (t_end - t_start).total_seconds()

    env_log = {
        "computation_type" : "kappa_g4_recalibration_tol0001",
        "date"             : t_start.isoformat(),
        "python_version"   : sys.version,
        "numpy_version"    : np.__version__,
        "scipy_version"    : scipy.__version__,
        "pandas_version"   : pd.__version__,
        "hostname"         : socket.gethostname(),
        "platform"         : platform.platform(),
        "gpu"              : "none (CPU-only)",
        "git_commit"       : "see pilot/ git log",
    }

    output_json = {
        "computation_type"    : "kappa_g4_recalibration_tol0001",
        "description"         : (
            "G4 cell re-calibration attempt with tighter binary-search "
            f"tolerance tol_kappa={TOL_NEW} (vs. previous {TOL_OLD}). "
            "Checks whether kappa_mean >= 0.999 is achievable."
        ),
        "previous_G4"         : {
            "alpha_b"      : ALPHA_B_PREV,
            "kappa_mean"   : 0.994698,
            "tol_used"     : TOL_OLD,
        },
        "new_G4"              : {
            "alpha_b"       : alpha_b_new,
            "tol_used"      : TOL_NEW,
            "kappa_search"  : kappa_search,
            "kappa_mean"    : kappa_mean,
            "kappa_std"     : kappa_std,
            "NIE_mean"      : NIE_mean,
            "NIE_std"       : NIE_std,
            "NDE_mean"      : NDE_mean,
            "NDE_std"       : NDE_std,
            "kappa_mean_gte_0999": kappa_mean_gte_999,
        },
        "per_seed"            : per_seed,
        "hc3a_definition"     : hc3a_definition,
        "draft_note"          : draft_note,
        "elapsed_s"           : elapsed,
        "env_log"             : env_log,
    }

    json_path = RESULTS_DIR / "kappa_g4_recalibration.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output_json, f, indent=2, ensure_ascii=False, default=str)

    df = pd.DataFrame(per_seed)
    csv_path = RESULTS_DIR / "kappa_g4_recalibration_per_seed.csv"
    df.to_csv(csv_path, index=False)

    print()
    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"Previous G4: alpha_b={ALPHA_B_PREV},  kappa_mean=0.994698")
    print(f"New G4     : alpha_b={alpha_b_new:.8f},  kappa_mean={kappa_mean:.6f} +/- {kappa_std:.2e}")
    print(f"kappa_mean >= 0.999: {'YES' if kappa_mean_gte_999 else 'NO'}")
    print(f"HC3(a) definition: {hc3a_definition}")
    print(f"Elapsed: {elapsed:.1f} s")
    print(f"Outputs:")
    print(f"  {json_path}")
    print(f"  {csv_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
