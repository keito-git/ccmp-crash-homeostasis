"""
Physics-misspecification robustness (AAP R1 revision)
=====================================================
Risk Homeostasis / CCMP study
Authors: Keito Inoshita, Akira Kawai

PURPOSE
-------
Reviewer requests: (i) robustness of the Delta-v power exponent, (ii) the fixed
30% attenuation of impact velocity.  This script varies the TRUE physics of the
data-generating process,

    Y = (delta_v / 100)^k_true + 0.1*U + eps_y,
    delta_v = (v_base + 5*b) * (1 - c_true*T),

while the analyst keeps the frozen working physics (k = 4, c = 0.3) when
constructing the magnitude bounds [D_L, D_U].  It reports, per configuration:

  * tau_LATE_true (twin simulation) and coverage of the IPSW IV Wald CI
    -> the point test never uses the physics, so its validity should not move;
  * D_true, the analyst's [D_L, D_U] under the working physics, and whether
    the Imbens-Manski region for kappa_LATE still covers kappa_LATE_true;
  * the same bounds recomputed with the correct physics (oracle analyst).

The SCM, selection model, oracle IPSW weights, seeds, and N are identical to
hc_t2_reduction.py; only the two physics constants are varied.  Existing
scripts are imported, not modified.

GPU: none (CPU-only).  Fabrication: prohibited; all numbers come from this run.
"""

from __future__ import annotations

import json
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hc_t2_reduction as base  # noqa: E402  (frozen SCM constants and helpers)

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

K_WORK: float = 4.0     # analyst's working exponent (frozen)
C_WORK: float = 0.30    # analyst's working attenuation (frozen)

PHYSICS_CONFIGS: List[Dict] = [
    {"k_true": 2.0, "c_true": 0.30},
    {"k_true": 3.0, "c_true": 0.30},
    {"k_true": 4.0, "c_true": 0.30},   # frozen baseline
    {"k_true": 5.0, "c_true": 0.30},
    {"k_true": 4.0, "c_true": 0.20},
    {"k_true": 4.0, "c_true": 0.40},
]

CELLS: List[Dict] = [
    {"cell": "T2-a", "rho": 0.5, "alpha_b": 0.3},
    {"cell": "T2-c", "rho": 0.5, "alpha_b": 3.972},
]


def phi_const(k: float, c: float) -> float:
    """D = phi_const * E[S(0)^k | complier] under physics (k, c)."""
    return (1.0 - (1.0 - c) ** k) / (100.0 ** k)


def hm_bounds(S0: np.ndarray, pi_c: float, k: float, c: float) -> tuple:
    """Horowitz-Manski trimming of S(0)^k at complier share pi_c."""
    Ws = np.sort(S0 ** k)
    m = max(1, int(round(pi_c * len(Ws))))
    pc = phi_const(k, c)
    return float(pc * Ws[:m].mean()), float(pc * Ws[-m:].mean())


def im_interval(tau_hat: float, se_tau: float, D_L: float, D_U: float) -> tuple:
    """Imbens-Manski two-layer interval for kappa = 1 + tau/D.

    Mirrors hc_t2_reduction.run_calibration_seed: each endpoint carries its own
    SE (SE_tau/D), with one shared critical value from the larger SE.  The
    endpoints are ordered so that the same rule also holds when tau_hat > 0.
    """
    ends = sorted([(1.0 + tau_hat / D_L, se_tau / D_L),
                   (1.0 + tau_hat / D_U, se_tau / D_U)])
    (lo, se_lo), (hi, se_hi) = ends
    c_im = base.imbens_manski_cval(hi - lo, max(se_lo, se_hi))
    return float(lo - c_im * se_lo), float(hi + c_im * se_hi)


def run_seed(rho: float, alpha_b: float, k: float, c: float, seed: int,
             N: int = base.N_FINAL) -> Dict:
    rng = np.random.default_rng(seed)
    U = rng.standard_normal(N)
    eps_b = rng.normal(0.0, base.STD_EPS_B, N)
    eps_v = rng.normal(0.0, base.STD_EPS_V, N)
    v_base = 50.0 + 10.0 * U + eps_v

    # Twin simulation for complier truth
    P0 = base.sigmoid(base.LOGIT_P0 + rho * U + base.IV_STR * (0.0 - 0.5))
    P1 = base.sigmoid(base.LOGIT_P0 + rho * U + base.IV_STR * (1.0 - 0.5))
    U_T = rng.uniform(0.0, 1.0, N)
    is_c = (U_T <= P1) & ~(U_T <= P0)
    pi_c = float(is_c.mean())

    b0 = base.GAMMA_B * U + eps_b
    b1 = alpha_b + base.GAMMA_B * U + eps_b

    def y_cf(t: float, b_in: np.ndarray) -> np.ndarray:
        dv = (v_base + base.SPEED_MULT * b_in) * (1.0 - c * t)
        return (dv / 100.0) ** k + base.BETA_U * U

    Y11, Y10, Y00 = y_cf(1.0, b1), y_cf(1.0, b0), y_cf(0.0, b0)
    nie = float(np.mean((Y11 - Y10)[is_c]))
    nde = float(np.mean((Y10 - Y00)[is_c]))
    tau_true = nie + nde
    D_true = abs(nde)
    kappa_true = nie / D_true

    S0 = v_base + base.SPEED_MULT * b0
    D_L_w, D_U_w = hm_bounds(S0, pi_c, K_WORK, C_WORK)   # working physics
    D_L_o, D_U_o = hm_bounds(S0, pi_c, k, c)             # correct physics

    # Observed IV data under C = 1 with oracle IPSW (identical to base)
    Z = rng.binomial(1, 0.5, N).astype(float)
    T = rng.binomial(1, base.sigmoid(base.LOGIT_P0 + rho * U
                                     + base.IV_STR * (Z - 0.5))).astype(float)
    b = alpha_b * T + base.GAMMA_B * U + eps_b
    dv = (v_base + base.SPEED_MULT * b) * (1.0 - c * T)
    Y = (dv / 100.0) ** k + base.BETA_U * U + rng.normal(0.0, base.STD_EPS_Y, N)
    P_crash = base.sigmoid(base.ETA_0 + base.ETA_T * T + base.ETA_U * U
                           + base.ETA_B * b)
    C = rng.binomial(1, P_crash).astype(bool)
    r = base._ipsw_wald(Y[C], T[C], Z[C], 1.0 / P_crash[C])

    im_w = im_interval(r["tau"], r["SE_tau"], D_L_w, D_U_w)
    im_o = im_interval(r["tau"], r["SE_tau"], D_L_o, D_U_o)
    return {
        "seed": seed, "pi_c": pi_c,
        "tau_true": tau_true, "D_true": D_true, "kappa_true": kappa_true,
        "tau_hat": r["tau"], "SE_tau": r["SE_tau"],
        "tau_ci_covers": bool(r["CI_L"] <= tau_true <= r["CI_U"]),
        "reject_H0": bool(r["reject"]),
        "D_L_work": D_L_w, "D_U_work": D_U_w,
        "D_in_work_bounds": bool(D_L_w <= D_true <= D_U_w),
        "IM_work": im_w,
        "IM_work_covers": bool(im_w[0] <= kappa_true <= im_w[1]),
        "IM_work_below_1": bool(im_w[1] < 1.0),
        "D_L_oracle": D_L_o, "D_U_oracle": D_U_o,
        "D_in_oracle_bounds": bool(D_L_o <= D_true <= D_U_o),
        "IM_oracle": im_o,
        "IM_oracle_covers": bool(im_o[0] <= kappa_true <= im_o[1]),
    }


def main() -> None:
    t0 = datetime.now()
    out: List[Dict] = []
    for cell in CELLS:
        for pc in PHYSICS_CONFIGS:
            rows = [run_seed(cell["rho"], cell["alpha_b"], pc["k_true"],
                             pc["c_true"], s) for s in base.SEEDS]
            agg = {
                **cell, **pc,
                "tau_true_mean": float(np.mean([r["tau_true"] for r in rows])),
                "tau_hat_mean": float(np.mean([r["tau_hat"] for r in rows])),
                "tau_ci_coverage": f"{sum(r['tau_ci_covers'] for r in rows)}/5",
                "reject_H0": f"{sum(r['reject_H0'] for r in rows)}/5",
                "kappa_true_mean": float(np.mean([r["kappa_true"] for r in rows])),
                "D_true_mean": float(np.mean([r["D_true"] for r in rows])),
                "D_work_bounds_mean": [float(np.mean([r["D_L_work"] for r in rows])),
                                       float(np.mean([r["D_U_work"] for r in rows]))],
                "D_in_work_bounds": f"{sum(r['D_in_work_bounds'] for r in rows)}/5",
                "IM_work_mean": [float(np.mean([r["IM_work"][0] for r in rows])),
                                 float(np.mean([r["IM_work"][1] for r in rows]))],
                "IM_work_coverage": f"{sum(r['IM_work_covers'] for r in rows)}/5",
                "IM_work_below_1": f"{sum(r['IM_work_below_1'] for r in rows)}/5",
                "IM_oracle_coverage": f"{sum(r['IM_oracle_covers'] for r in rows)}/5",
                "per_seed": rows,
            }
            out.append(agg)
            print(f"{cell['cell']} k={pc['k_true']} c={pc['c_true']}: "
                  f"tau_true={agg['tau_true_mean']:+.5f} tau_hat={agg['tau_hat_mean']:+.5f} "
                  f"cov={agg['tau_ci_coverage']} rej={agg['reject_H0']} | "
                  f"kappa={agg['kappa_true_mean']:.4f} D={agg['D_true_mean']:.5f} "
                  f"Dw=[{agg['D_work_bounds_mean'][0]:.5f},{agg['D_work_bounds_mean'][1]:.5f}] "
                  f"Din={agg['D_in_work_bounds']} IMw=[{agg['IM_work_mean'][0]:.3f},"
                  f"{agg['IM_work_mean'][1]:.3f}] IMcov_w={agg['IM_work_coverage']} "
                  f"<1={agg['IM_work_below_1']} IMcov_oracle={agg['IM_oracle_coverage']}",
                  flush=True)
    stamp = t0.strftime("%Y%m%d_%H%M%S")
    path = RESULTS_DIR / f"r1_physics_misspecification_{stamp}.json"
    with open(path, "w") as f:
        json.dump({"env": {"python": platform.python_version(),
                           "numpy": np.__version__, "started": t0.isoformat(),
                           "finished": datetime.now().isoformat(),
                           "K_WORK": K_WORK, "C_WORK": C_WORK,
                           "seeds": base.SEEDS, "N": base.N_FINAL},
                   "results": out}, f, indent=2)
    print("saved", path)


if __name__ == "__main__":
    main()
