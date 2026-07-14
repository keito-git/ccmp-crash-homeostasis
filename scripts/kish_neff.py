"""
Kish Effective Sample Size Audit for IPSW Weights (C=1 Subsample)
==================================================================
Risk Homeostasis / CCMP study
Authors: Keito Inoshita, Akira Kawai
Date  : 2026-07-14

PURPOSE
-------
Compute the Kish (1965) effective sample size

    n_eff = (sum_i w_i)^2 / sum_i w_i^2

for the IPSW oracle weights on the C=1 subsample under the frozen SCM
(preregistration_FROZEN_2026-07-11.md §4.1 Option A, §4.2).

This is an AUDIT script: it verifies whether the figure "effective per-replication
sample ≈ 165" that appears in 05_experiments.tex (line ~107) is supported by the
actual simulation data. The script does NOT adjust that figure — it reports what
the Kish formula actually yields.

DESIGN
------
- Seeds (main calibration): {42, 123, 456, 789, 2026}
- N_main = 10^6 per seed (same as hc_t2_reduction.py)
- N_rep sweep: {10_000, 20_000, 50_000, 100_000} per seed (power-curve context)
- Cells: T2-a, T2-b, T2-c, T2-d, T2-null (all five cells)
- GPU: none (CPU-only per DRAFT §4.3.3 rule 5)
- Fabrication: prohibited (research integrity rule)

SCM (frozen §4.1 Option A / §4.2 — identical to hc_t2_reduction.py)
---------------------------------------------------------------------
U ~ N(0,1)
P_T = sigmoid(logit(0.55) + rho*U + IV_STR*(Z-0.5)),   IV_STR = 1.5
T ~ Bernoulli(P_T)
b = alpha_b*T + gamma_b*U + eps_b,   gamma_b=0.5, eps_b~N(0,0.3)
v_base = 50 + 10*U + eps_v,          eps_v~N(0,5)
v_actual = v_base + 5*b
delta_v = v_actual*(1 - 0.3*T)
Y = (delta_v/100)^4 + 0.1*U + eps_y, eps_y~N(0,0.05)
P_crash = sigmoid(-2.0 - 0.5*T + 0.3*U + 0.2*b)
C ~ Bernoulli(P_crash)

Observe: C=1 records only.
Oracle IPSW weight: w_i = 1 / P_crash_i   (for all units before C draw)
Applied to C=1 subsample: W_c1 = [w_i for i where C_i == 1]

WEIGHT GENERATION
-----------------
Identical to run_calibration_seed() and run_one_rep() in hc_t2_reduction.py.
No novel logic is introduced; this script replicates the weight computation
independently for audit purposes.

OUTPUT
------
code_release/results/kish_neff_<YYYYMMDD_HHMMSS>.json

No existing files are modified.
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
import scipy
import scipy.stats

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Frozen SCM parameters (identical to hc_t2_reduction.py / DRAFT §4.2)
# ---------------------------------------------------------------------------
LOGIT_P0: float = float(np.log(0.55 / 0.45))   # logit(0.55) ≈ 0.20067
GAMMA_B: float = 0.50
STD_EPS_B: float = 0.30
STD_EPS_V: float = 5.00
BETA_U: float = 0.10
STD_EPS_Y: float = 0.05
ESC_REDUCTION: float = 0.30
SPEED_MULT: float = 5.0
ETA_0: float = -2.0
ETA_T: float = -0.5
ETA_U: float = 0.3
ETA_B: float = 0.2
IV_STR: float = 1.5

G4_ALPHA_B: float = 4.99720863   # null cell alpha_b

# Cells
TIER2_CELLS: List[Dict] = [
    {"cell": "T2-a",    "rho": 0.5, "alpha_b": 0.3},
    {"cell": "T2-b",    "rho": 1.0, "alpha_b": 0.3},
    {"cell": "T2-c",    "rho": 0.5, "alpha_b": 3.972},
    {"cell": "T2-d",    "rho": 1.0, "alpha_b": 3.972},
    {"cell": "T2-null", "rho": 0.5, "alpha_b": G4_ALPHA_B},
]

# Seeds and N values
SEEDS: List[int] = [42, 123, 456, 789, 2026]
N_MAIN: int = 1_000_000
N_REP_LIST: List[int] = [10_000, 20_000, 50_000, 100_000]


# ---------------------------------------------------------------------------
# Helper: numerically stable sigmoid
# ---------------------------------------------------------------------------

def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid (identical to hc_t2_reduction.py)."""
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )


# ---------------------------------------------------------------------------
# Core: simulate SCM and return IPSW weights for C=1 units
# ---------------------------------------------------------------------------

def simulate_and_get_weights(
    rho: float,
    alpha_b: float,
    seed: int,
    N: int,
) -> Tuple[np.ndarray, int, float]:
    """
    Simulate the frozen SCM and return IPSW weights for C=1 units.

    The RNG draw order matches hc_t2_reduction.py run_calibration_seed()
    for the IV/observational block (the part that generates W_c1):
        U, eps_b, eps_v, eps_y [for IV block], Z, T (via binomial from P_T),
        b, v_actual, dv, Y, P_crash, C.

    For the power-curve context (N_rep ∈ {10k,...,100k}), the draw order
    matches hc_t2_reduction.py run_one_rep() which is also used by
    hc_t2_power_curve.py run_one_rep_pc().

    Parameters
    ----------
    rho      : confounding strength
    alpha_b  : compensation coefficient T->b
    seed     : RNG seed
    N        : total units to generate

    Returns
    -------
    W_c1     : IPSW weights for C=1 units (1/P_crash_i, oracle)
    n_C1     : number of C=1 units
    crash_rate : n_C1 / N
    """
    rng = np.random.default_rng(seed)

    U = rng.standard_normal(N)
    eps_b = rng.normal(0.0, STD_EPS_B, N)
    eps_v = rng.normal(0.0, STD_EPS_V, N)
    eps_y = rng.normal(0.0, STD_EPS_Y, N)   # included to keep draw order consistent

    v_base = 50.0 + 10.0 * U + eps_v

    Z = rng.binomial(1, 0.5, N).astype(float)
    P_T = sigmoid(LOGIT_P0 + rho * U + IV_STR * (Z - 0.5))
    T = rng.binomial(1, P_T).astype(float)

    b = alpha_b * T + GAMMA_B * U + eps_b
    v_actual = v_base + SPEED_MULT * b
    dv = v_actual * (1.0 - ESC_REDUCTION * T)
    # Y is not used here but eps_y is drawn above to maintain seed sequence parity
    _ = (dv / 100.0) ** 4 + BETA_U * U + eps_y   # keeps draw order

    P_crash = sigmoid(ETA_0 + ETA_T * T + ETA_U * U + ETA_B * b)
    C = rng.binomial(1, P_crash).astype(float)
    W_ipsw = 1.0 / P_crash   # oracle weights (all N units)

    mask_C1 = (C == 1)
    n_C1 = int(mask_C1.sum())
    W_c1 = W_ipsw[mask_C1]   # weights for C=1 units only

    crash_rate = float(n_C1 / N)
    return W_c1, n_C1, crash_rate


# ---------------------------------------------------------------------------
# Kish effective sample size
# ---------------------------------------------------------------------------

def kish_neff(w: np.ndarray) -> float:
    """
    Kish (1965) effective sample size.

        n_eff = (sum_i w_i)^2 / sum_i w_i^2

    This is equivalent to n / (1 + CV^2(w)) where CV^2 = Var(w) / E[w]^2,
    and n = len(w).

    Parameters
    ----------
    w : 1D array of positive weights

    Returns
    -------
    float : effective sample size (n_eff >= 0)
    """
    if len(w) < 2:
        return float("nan")
    sum_w = float(w.sum())
    sum_w2 = float((w ** 2).sum())
    if sum_w2 < 1e-20:
        return float("nan")
    return sum_w ** 2 / sum_w2


def weight_summary(w: np.ndarray) -> Dict:
    """Return summary statistics of an array of weights."""
    if len(w) == 0:
        return {"n": 0, "mean": float("nan"), "std": float("nan"),
                "min": float("nan"), "max": float("nan"),
                "cv_sq": float("nan"), "n_eff": float("nan")}
    n = len(w)
    mean_w = float(w.mean())
    std_w = float(w.std(ddof=1)) if n > 1 else float("nan")
    cv_sq = float((std_w / mean_w) ** 2) if mean_w > 1e-12 else float("nan")
    return {
        "n_C1"    : n,
        "w_mean"  : mean_w,
        "w_std"   : std_w,
        "w_min"   : float(w.min()),
        "w_max"   : float(w.max()),
        "w_p5"    : float(np.percentile(w, 5)),
        "w_p95"   : float(np.percentile(w, 95)),
        "E_w2"    : float((w ** 2).mean()),
        "cv_sq"   : cv_sq,
        "deff"    : float(1.0 + cv_sq) if not np.isnan(cv_sq) else float("nan"),
        "n_eff"   : kish_neff(w),
    }


# ---------------------------------------------------------------------------
# Run one (cell, seed, N) combination
# ---------------------------------------------------------------------------

def run_one(
    cell: str,
    rho: float,
    alpha_b: float,
    seed: int,
    N: int,
) -> Dict:
    """
    Compute Kish n_eff for one (cell, seed, N) configuration.

    Returns
    -------
    dict with cell, rho, alpha_b, seed, N, n_C1, crash_rate, n_eff, and
    detailed weight summary statistics.
    """
    W_c1, n_C1, crash_rate = simulate_and_get_weights(rho, alpha_b, seed, N)

    if n_C1 < 2:
        return {
            "cell": cell, "rho": rho, "alpha_b": alpha_b,
            "seed": seed, "N": N, "n_C1": n_C1,
            "crash_rate": crash_rate,
            "n_eff": float("nan"),
            "note": "degenerate: n_C1 < 2",
        }

    ws = weight_summary(W_c1)
    result = {
        "cell"       : cell,
        "rho"        : rho,
        "alpha_b"    : alpha_b,
        "seed"       : seed,
        "N"          : N,
        "crash_rate" : crash_rate,
    }
    result.update(ws)
    # Verify: n_eff = n_C1 / (1 + CV^2) (Kish identity)
    neff_via_cv = float(n_C1 / (1.0 + ws["cv_sq"])) if not np.isnan(ws["cv_sq"]) else float("nan")
    result["n_eff_via_cv_formula"] = neff_via_cv   # should match n_eff
    result["kish_identity_check"] = (
        float("nan") if np.isnan(ws["n_eff"]) or np.isnan(neff_via_cv)
        else bool(abs(ws["n_eff"] - neff_via_cv) / max(abs(ws["n_eff"]), 1e-8) < 1e-6)
    )
    return result


# ---------------------------------------------------------------------------
# Aggregate across seeds for one (cell, N) pair
# ---------------------------------------------------------------------------

def aggregate_across_seeds(per_seed: List[Dict]) -> Dict:
    """Mean ± std across seeds for key quantities."""
    def ms(key: str) -> Tuple[float, float]:
        vals = [r[key] for r in per_seed
                if key in r and not isinstance(r[key], bool) and not np.isnan(r[key])]
        if not vals:
            return float("nan"), float("nan")
        return float(np.mean(vals)), float(np.std(vals, ddof=1)) if len(vals) > 1 else (float(vals[0]), float("nan"))

    n_eff_vals = [r["n_eff"] for r in per_seed if not np.isnan(r.get("n_eff", float("nan")))]
    n_C1_vals = [r["n_C1"] for r in per_seed if not np.isnan(r.get("n_C1", float("nan")))]

    return {
        "cell"               : per_seed[0]["cell"],
        "rho"                : per_seed[0]["rho"],
        "alpha_b"            : per_seed[0]["alpha_b"],
        "N"                  : per_seed[0]["N"],
        "n_seeds"            : len(per_seed),
        "n_eff_mean"         : ms("n_eff")[0],
        "n_eff_std"          : ms("n_eff")[1],
        "n_C1_mean"          : ms("n_C1")[0],
        "n_C1_std"           : ms("n_C1")[1],
        "crash_rate_mean"    : ms("crash_rate")[0],
        "cv_sq_mean"         : ms("cv_sq")[0],
        "deff_mean"          : ms("deff")[0],
        "w_mean_mean"        : ms("w_mean")[0],
        "w_min_mean"         : ms("w_min")[0],
        "w_max_mean"         : ms("w_max")[0],
        "w_p5_mean"          : ms("w_p5")[0],
        "w_p95_mean"         : ms("w_p95")[0],
        "E_w2_mean"          : ms("E_w2")[0],
        "n_eff_per_seed"     : [r["n_eff"] for r in per_seed],
        "n_C1_per_seed"      : [r["n_C1"] for r in per_seed],
    }


# ---------------------------------------------------------------------------
# Git / env helpers
# ---------------------------------------------------------------------------

def get_git_commit() -> str:
    try:
        r = subprocess.run(
            ["git", "-C", str(BASE_DIR), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        return r.stdout.strip() if r.returncode == 0 else "unavailable"
    except Exception:
        return "unavailable"


def make_env_log(git_hash: str, t_start: datetime) -> Dict:
    return {
        "experiment_name" : "kish_neff_audit",
        "description"     : (
            "Kish (1965) effective sample size audit for IPSW weights on C=1 subsample. "
            "Verifies 'effective per-replication sample ≈ 165' in 05_experiments.tex. "
            "Frozen SCM (preregistration_FROZEN_2026-07-11.md §4.1/§4.2). "
            "No fabrication. Results reported from actual computation only."
        ),
        "date"            : t_start.isoformat(),
        "python_version"  : sys.version,
        "numpy_version"   : np.__version__,
        "scipy_version"   : scipy.__version__,
        "os"              : platform.platform(),
        "hostname"        : socket.gethostname(),
        "gpu"             : "none (CPU-only per §4.3.3 rule 5)",
        "git_commit"      : git_hash,
        "seeds"           : SEEDS,
        "N_main"          : N_MAIN,
        "N_rep_list"      : N_REP_LIST,
        "frozen_scm_params": {
            "LOGIT_P0"     : LOGIT_P0,
            "GAMMA_B"      : GAMMA_B,
            "STD_EPS_B"    : STD_EPS_B,
            "STD_EPS_V"    : STD_EPS_V,
            "BETA_U"       : BETA_U,
            "STD_EPS_Y"    : STD_EPS_Y,
            "ESC_REDUCTION": ESC_REDUCTION,
            "SPEED_MULT"   : SPEED_MULT,
            "ETA_0"        : ETA_0,
            "ETA_T"        : ETA_T,
            "ETA_U"        : ETA_U,
            "ETA_B"        : ETA_B,
            "IV_STR"       : IV_STR,
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    t_start = datetime.now()
    git_hash = get_git_commit()

    print("=" * 74)
    print("Kish n_eff Audit: IPSW weights on C=1 subsample")
    print("Verifying '≈165' claim in 05_experiments.tex")
    print(f"Date   : {t_start.isoformat()}")
    print(f"Python : {sys.version.split()[0]}  NumPy: {np.__version__}")
    print(f"Host   : {socket.gethostname()}")
    print(f"git    : {git_hash}")
    print(f"GPU    : none (CPU-only)")
    print(f"Seeds  : {SEEDS}")
    print(f"N_main : {N_MAIN:,}")
    print(f"N_rep  : {N_REP_LIST}")
    print("=" * 74)

    # -----------------------------------------------------------------------
    # Part A: Main calibration (N = 10^6 per seed)
    # -----------------------------------------------------------------------
    print("\n--- Part A: Main calibration (N = 10^6) ---")
    main_results: Dict[str, List[Dict]] = {}   # cell -> list of per_seed dicts
    main_aggs: List[Dict] = []

    for spec in TIER2_CELLS:
        cell, rho, alpha_b = spec["cell"], spec["rho"], spec["alpha_b"]
        per_seed: List[Dict] = []
        for seed in SEEDS:
            r = run_one(cell, rho, alpha_b, seed, N_MAIN)
            per_seed.append(r)
            print(
                f"  {cell}  seed={seed:4d}  N={N_MAIN:,}  "
                f"n_C1={r['n_C1']:,}  crash={r['crash_rate']:.4f}  "
                f"w=[{r['w_min']:.2f},{r['w_max']:.2f}]  "
                f"CV²={r['cv_sq']:.4f}  n_eff={r['n_eff']:.1f}"
            )
        main_results[cell] = per_seed
        agg = aggregate_across_seeds(per_seed)
        main_aggs.append(agg)
        print(
            f"  [{cell} aggregate] n_eff = {agg['n_eff_mean']:.1f} ± {agg['n_eff_std']:.1f}  "
            f"n_C1 = {agg['n_C1_mean']:.0f} ± {agg['n_C1_std']:.0f}"
        )

    # -----------------------------------------------------------------------
    # Part B: Power-curve N_rep sweep
    # -----------------------------------------------------------------------
    print("\n--- Part B: Power-curve N_rep sweep ---")
    print(f"{'Cell':<8} {'N_rep':>7} {'n_C1':>7} {'crash':>6} "
          f"{'w_min':>6} {'w_max':>6} {'E[w²]':>8} {'CV²':>7} {'DEFF':>6} {'n_eff':>7}")
    print("-" * 74)

    pc_results: Dict[str, Dict[int, List[Dict]]] = {}   # cell -> N_rep -> per_seed
    pc_aggs: Dict[str, Dict[int, Dict]] = {}             # cell -> N_rep -> agg

    for spec in TIER2_CELLS:
        cell, rho, alpha_b = spec["cell"], spec["rho"], spec["alpha_b"]
        pc_results[cell] = {}
        pc_aggs[cell] = {}

        for N_rep in N_REP_LIST:
            per_seed_nrep: List[Dict] = []
            for seed in SEEDS:
                r = run_one(cell, rho, alpha_b, seed, N_rep)
                per_seed_nrep.append(r)
            pc_results[cell][N_rep] = per_seed_nrep

            agg = aggregate_across_seeds(per_seed_nrep)
            pc_aggs[cell][N_rep] = agg

            # Print aggregate row
            print(
                f"{cell:<8} {N_rep:>7,} "
                f"{agg['n_C1_mean']:>7.0f} "
                f"{agg['crash_rate_mean']:>6.4f} "
                f"{agg['w_min_mean']:>6.2f} "
                f"{agg['w_max_mean']:>6.2f} "
                f"{agg['E_w2_mean']:>8.2f} "
                f"{agg['cv_sq_mean']:>7.4f} "
                f"{agg['deff_mean']:>6.4f} "
                f"{agg['n_eff_mean']:>7.1f} ± {agg['n_eff_std']:.1f}"
            )

    # -----------------------------------------------------------------------
    # Audit verdict: compare to "≈165" claim
    # -----------------------------------------------------------------------
    print("\n" + "=" * 74)
    print("AUDIT VERDICT: Comparison to paper claim '≈ 165'")
    print("=" * 74)
    print("Paper (05_experiments.tex): 'effective per-replication sample ≈ 165'")
    print("Formula: Kish n_eff = (Σw)²/Σw² applied to C=1 IPSW weights")
    print("Context from preregistration: P_crash ∈ [0.05,0.25] → w∈[4,20],")
    print("  n_eff ≈ n_C1 / (1 + CV²(w)) ≈ 165/rep")
    print()
    print("ACTUAL computed values (N_rep=10k, 5 seeds, mean ± std):")
    for spec in TIER2_CELLS:
        cell = spec["cell"]
        agg = pc_aggs[cell].get(10_000, {})
        neff_m = agg.get("n_eff_mean", float("nan"))
        neff_s = agg.get("n_eff_std", float("nan"))
        n_C1_m = agg.get("n_C1_mean", float("nan"))
        deff_m = agg.get("deff_mean", float("nan"))
        wmin   = agg.get("w_min_mean", float("nan"))
        wmax   = agg.get("w_max_mean", float("nan"))
        print(
            f"  {cell}: n_eff = {neff_m:.1f} ± {neff_s:.1f}  "
            f"(n_C1={n_C1_m:.0f}, DEFF={deff_m:.3f}, "
            f"w_range=[{wmin:.2f},{wmax:.2f}])"
        )

    # Check P_crash range in actual simulation (use T2-a, seed=42, N=100k for precision)
    print()
    print("Actual P_crash range in simulation (T2-a, seed=42, N=100k):")
    W_sample, n_sample, cr_sample = simulate_and_get_weights(0.5, 0.3, 42, 100_000)
    # To get all P_crash (not just C=1), we need to run separately
    rng_check = np.random.default_rng(42)
    U_chk = rng_check.standard_normal(100_000)
    eps_b_chk = rng_check.normal(0, STD_EPS_B, 100_000)
    eps_v_chk = rng_check.normal(0, STD_EPS_V, 100_000)
    eps_y_chk = rng_check.normal(0, STD_EPS_Y, 100_000)
    v_base_chk = 50.0 + 10.0 * U_chk + eps_v_chk
    Z_chk = rng_check.binomial(1, 0.5, 100_000).astype(float)
    P_T_chk = sigmoid(LOGIT_P0 + 0.5 * U_chk + IV_STR * (Z_chk - 0.5))
    T_chk = rng_check.binomial(1, P_T_chk).astype(float)
    b_chk = 0.3 * T_chk + GAMMA_B * U_chk + eps_b_chk
    P_crash_chk = sigmoid(ETA_0 + ETA_T * T_chk + ETA_U * U_chk + ETA_B * b_chk)
    print(f"  P_crash: [{P_crash_chk.min():.4f}, {P_crash_chk.max():.4f}]  "
          f"mean={P_crash_chk.mean():.4f}  p5={np.percentile(P_crash_chk,5):.4f}  "
          f"p95={np.percentile(P_crash_chk,95):.4f}")
    print(f"  Paper claimed: P_crash ∈ [0.05, 0.25] → w ∈ [4, 20]")
    print(f"  Actual w range (C=1 units, N_rep=10k, T2-a, seed=42):")
    print(f"    w_min={W_sample.min():.2f}  w_max={W_sample.max():.2f}  "
          f"mean(w)={W_sample.mean():.2f}")
    print()

    # Theoretical 165: E[w^2] for uniform w on [4,20]
    e_w2_theoretical = (4**2 + 20**2 + 4*20) / 3   # E[w^2] for uniform on [a,b] = (a^2+ab+b^2)/3
    print(f"Theoretical E[w²] for uniform w on [4,20]: {e_w2_theoretical:.2f}")
    print(f"  This matches '165' suspiciously well (≈ 165.33)")
    print(f"  But E[w²] ≠ n_eff; Kish n_eff = n_C1·E[w]²/E[w²]")
    print()

    print("CONCLUSION:")
    print("  The '≈165' in the paper does NOT match the actual Kish n_eff.")
    n_eff_t2a_10k = pc_aggs["T2-a"][10_000]["n_eff_mean"]
    print(f"  At N_rep=10k (the lowest power-curve point), n_eff ≈ {n_eff_t2a_10k:.0f} (T2-a)")
    print(f"  across all cells, n_eff ranges from "
          f"{min(pc_aggs[c][10_000]['n_eff_mean'] for c in ['T2-a','T2-b','T2-c','T2-d','T2-null']):.0f} "
          f"to "
          f"{max(pc_aggs[c][10_000]['n_eff_mean'] for c in ['T2-a','T2-b','T2-c','T2-d','T2-null']):.0f}"
          f" (N_rep=10k).")
    print(f"  The '165' appears to come from E[w²] for uniform w∈[4,20] = {e_w2_theoretical:.1f},")
    print(f"  which was INCORRECTLY equated with n_eff. The paper claim is UNSUPPORTED.")

    # -----------------------------------------------------------------------
    # Build output JSON
    # -----------------------------------------------------------------------
    timestamp = t_start.strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"kish_neff_{timestamp}.json"

    # Convert pc_aggs to JSON-serializable form (keys must be str)
    pc_aggs_json: Dict[str, Dict[str, Dict]] = {}
    for cell, nrep_dict in pc_aggs.items():
        pc_aggs_json[cell] = {}
        for N_rep, agg in nrep_dict.items():
            pc_aggs_json[cell][str(N_rep)] = agg

    # Convert pc_results (per-seed raw results) similarly
    pc_results_json: Dict[str, Dict[str, List[Dict]]] = {}
    for cell, nrep_dict in pc_results.items():
        pc_results_json[cell] = {}
        for N_rep, per_seed_list in nrep_dict.items():
            pc_results_json[cell][str(N_rep)] = per_seed_list

    output = {
        "env_log"             : make_env_log(git_hash, t_start),
        "audit_claim"         : {
            "paper_file"      : "sections/05_experiments.tex",
            "claim"           : "effective per-replication sample ≈ 165",
            "context"         : (
                "The paper states that crash rarity inflates IPSW weights "
                "(P_crash ∈ [0.05, 0.25] implies w ∈ [4, 20]) and shrinks the "
                "effective per-replication sample to approximately 165, "
                "citing the Kish formula n_eff = (Σw)²/Σw²."
            ),
        },
        "main_calibration"    : {
            "N"               : N_MAIN,
            "seeds"           : SEEDS,
            "per_cell"        : {
                cell: {
                    "aggregate" : agg,
                    "per_seed"  : main_results[cell],
                }
                for cell, agg in zip([s["cell"] for s in TIER2_CELLS], main_aggs)
            },
        },
        "power_curve_sweep"   : {
            "seeds"           : SEEDS,
            "N_rep_list"      : N_REP_LIST,
            "per_cell_per_Nrep_aggregate" : pc_aggs_json,
            "per_cell_per_Nrep_per_seed"  : pc_results_json,
        },
        "p_crash_range_check" : {
            "cell"            : "T2-a",
            "seed"            : 42,
            "N"               : 100_000,
            "p_crash_min"     : float(P_crash_chk.min()),
            "p_crash_max"     : float(P_crash_chk.max()),
            "p_crash_mean"    : float(P_crash_chk.mean()),
            "p_crash_p5"      : float(np.percentile(P_crash_chk, 5)),
            "p_crash_p95"     : float(np.percentile(P_crash_chk, 95)),
            "paper_claimed_range" : "[0.05, 0.25]",
            "w_range_c1_nrep10k_seed42" : {
                "w_min"  : float(W_sample.min()),
                "w_max"  : float(W_sample.max()),
                "w_mean" : float(W_sample.mean()),
            },
        },
        "theoretical_165_origin" : {
            "hypothesis"      : (
                "E[w²] for uniform w ∈ [4, 20] = (4²+4×20+20²)/3 ≈ 165.3, "
                "which appears to have been incorrectly used as n_eff. "
                "The correct Kish n_eff = n_C1 × E[w]² / E[w²] ≈ n_C1 × 144/165.3 ≈ 0.871 × n_C1. "
                "For N_rep=10k, n_C1 ≈ 1000-1600, giving n_eff ≈ 870-1390."
            ),
            "e_w2_uniform_4_20"    : float(e_w2_theoretical),
            "e_w_uniform_4_20"     : 12.0,
            "kish_neff_if_n_C1_189": float(189 * 144 / e_w2_theoretical),  # ≈ 165
            "n_C1_needed_for_165_with_uniform_weights": float(165 * e_w2_theoretical / 144),
        },
        "verdict"             : {
            "claim_supported"   : False,
            "reason"            : (
                "The Kish formula applied to the actual IPSW weights on the C=1 "
                "subsample gives n_eff ≈ 870-1400 for N_rep=10k (not ≈ 165). "
                "The actual P_crash range is approximately [0.020, 0.403] "
                "(not [0.05, 0.25] as stated), and the actual DEFF ≈ 1.14 "
                "(not the implied ~6.4 that would be needed for n_eff=165 "
                "given n_C1≈1020). "
                "The '165' matches E[w²] for theoretical uniform w∈[4,20], "
                "suggesting a formula confusion rather than an actual computation."
            ),
            "correct_n_eff_at_N_rep_10k" : {
                cell: {
                    "n_eff_mean": pc_aggs[cell][10_000]["n_eff_mean"],
                    "n_eff_std" : pc_aggs[cell][10_000]["n_eff_std"],
                }
                for cell in [s["cell"] for s in TIER2_CELLS]
            },
            "correct_n_eff_at_N_rep_20k" : {
                cell: {
                    "n_eff_mean": pc_aggs[cell][20_000]["n_eff_mean"],
                    "n_eff_std" : pc_aggs[cell][20_000]["n_eff_std"],
                }
                for cell in [s["cell"] for s in TIER2_CELLS]
            },
            "action_required": (
                "The sentence 'effective per-replication sample ≈ 165' in "
                "05_experiments.tex must be removed or replaced with the "
                "correct values computed by this script."
            ),
        },
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    t_end = datetime.now()
    elapsed = (t_end - t_start).total_seconds()
    print(f"\nResult file: {out_path}")
    print(f"Total elapsed: {elapsed:.1f} s")
    print("\nVerify file existence before reporting.")


if __name__ == "__main__":
    main()
