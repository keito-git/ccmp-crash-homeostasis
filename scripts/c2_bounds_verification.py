"""
C2: First-principles verification of kappa_LATE identification on Tier-2 SCM cells.
Confirms kappa_LATE = 1 + tau/D where tau = ATE_LATE is point-identified by IV Wald
and D = |NDE_LATE| is only partially identified via Horowitz-Manski trimming bounds.
Outputs: printed table (no files saved).
"""
from __future__ import annotations
import numpy as np

# Frozen SCM constants
LOGIT_P0    = 0.20067069546215124
GAMMA_B     = 0.5
STD_EPS_B   = 0.3
STD_EPS_V   = 5.0
BETA_U      = 0.1
ESC_RED     = 0.3
SPEED_MULT  = 5.0
IV_STR      = 1.5
PHI_C       = (1.0 - ESC_RED ** 4) / 100.0 ** 4   # = (1 - 0.7^4)/100^4, D = PHI_C * E[S(0)^4|.]


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def twin(cell, rho, alpha_b, seed=20260711, N=3_000_000):
    rng = np.random.default_rng(seed)
    U     = rng.standard_normal(N)
    eps_b = rng.normal(0.0, STD_EPS_B, N)
    eps_v = rng.normal(0.0, STD_EPS_V, N)
    v_base = 50.0 + 10.0 * U + eps_v

    P0 = sigmoid(LOGIT_P0 + rho * U + IV_STR * (0.0 - 0.5))
    P1 = sigmoid(LOGIT_P0 + rho * U + IV_STR * (1.0 - 0.5))
    UT = rng.uniform(0.0, 1.0, N)
    T0 = (UT <= P0)
    T1 = (UT <= P1)
    comp = T1 & (~T0)               # compliers
    pi_c = comp.mean()

    b0 = GAMMA_B * U + eps_b                    # b when T=0
    b1 = alpha_b + GAMMA_B * U + eps_b          # b when T=1
    S0 = v_base + SPEED_MULT * b0               # untreated physical speed S(0)
    S1 = v_base + SPEED_MULT * b1               # S(1) (compensated)

    def Y(T_out, b_in):
        dv = (v_base + SPEED_MULT * b_in) * (1.0 - ESC_RED * T_out)
        return (dv / 100.0) ** 4 + BETA_U * U

    Y11, Y10, Y00 = Y(1.0, b1), Y(1.0, b0), Y(0.0, b0)

    NIE_pop = (Y11 - Y10).mean();  NDE_pop = (Y10 - Y00).mean()
    NIE_LATE = (Y11 - Y10)[comp].mean();  NDE_LATE = (Y10 - Y00)[comp].mean()
    kap_pop  = NIE_pop / abs(NDE_pop)
    kap_LATE = NIE_LATE / abs(NDE_LATE)
    tau      = NIE_LATE + NDE_LATE                 # ATE_LATE = complier total effect
    D_true   = abs(NDE_LATE)

    # IV Wald (unconditional on C): tau is point-identified
    # First stage and reduced form under Z in {0,1}, 50/50.
    Z  = rng.integers(0, 2, N).astype(bool)
    Tobs = np.where(Z, T1, T0)
    Yobs = np.where(Tobs, Y11, Y00)                # observed world: T on => (1,b1); T off => (0,b0)
    FS = Tobs[Z].mean() - Tobs[~Z].mean()
    RF = Yobs[Z].mean() - Yobs[~Z].mean()
    tau_IV = RF / FS                               # Wald = ATE_LATE (point-ID)

    # CLAIM 4: Horowitz-Manski trimming bounds on E[S0^4 | complier]
    # Population marginal of W = S(0)^4 over ALL eligible units; complier is a pi_c-mass subset.
    W = S0 ** 4
    Ws = np.sort(W)
    k = int(round(pi_c * N))
    D_L = PHI_C * Ws[:k].mean()                    # lowest pi_c mass  -> smallest D
    D_U = PHI_C * Ws[-k:].mean()                   # highest pi_c mass -> largest D
    # kappa set: kappa = 1 + tau/D, increasing in D for tau<0
    kap_from_D = lambda D: 1.0 + tau_IV / D
    kap_lo = kap_from_D(D_L)                        # smallest D -> most negative -> lower kappa
    kap_hi = kap_from_D(D_U)                        # largest D  -> closest to 1  -> upper kappa

    return dict(cell=cell, rho=rho, alpha_b=alpha_b, pi_c=pi_c,
                EU_comp=U[comp].mean(),
                NDE_pop=NDE_pop, NDE_LATE=NDE_LATE, D_true=D_true,
                NIE_LATE=NIE_LATE, tau=tau, tau_IV=tau_IV,
                kap_pop=kap_pop, kap_LATE=kap_LATE,
                kap_recon=1.0 + tau / D_true,       # CLAIM 1 check
                kap_naive=1.0 + tau_IV / abs(NDE_pop),  # CLAIM 2: plug NDE_pop
                D_L=D_L, D_U=D_U, kap_lo=kap_lo, kap_hi=kap_hi)


CELLS = [("T2-a", 0.5, 0.3), ("T2-b", 1.0, 0.3),
         ("T2-c", 0.5, 3.972), ("T2-d", 1.0, 3.972)]

print("="*118)
print("C2 first-principles verification of kappa_LATE identification")
print("="*118)
for cell, rho, ab in CELLS:
    r = twin(cell, rho, ab)
    print(f"\n---- {r['cell']}  (rho={rho}, alpha_b={ab}, pi_c={r['pi_c']:.3f}, E[U|comp]={r['EU_comp']:+.3f}) ----")
    print(f"  NDE_pop  = {r['NDE_pop']:+.6f}   NDE_LATE = {r['NDE_LATE']:+.6f}   "
          f"(binding gap NDE_LATE-NDE_pop = {r['NDE_LATE']-r['NDE_pop']:+.6f})")
    print(f"  tau=ATE_LATE(struct) = {r['tau']:+.6f}   tau_IV(Wald) = {r['tau_IV']:+.6f}   "
          f"[point-ID check: match={abs(r['tau']-r['tau_IV'])<2e-3}]")
    print(f"  CLAIM1  kappa_LATE_true = {r['kap_LATE']:.5f}  vs  1+tau/D = {r['kap_recon']:.5f}  "
          f"[match={abs(r['kap_LATE']-r['kap_recon'])<1e-6}]")
    print(f"  CLAIM2  naive (plug NDE_pop) kappa_hat = {r['kap_naive']:.5f}   "
          f"bias = {r['kap_naive']-r['kap_LATE']:+.5f}  <- NDE_LATE!=NDE_pop drives it")
    print(f"  CLAIM3  H0 kappa=1 <=> tau=0.  tau_IV={r['tau_IV']:+.5f} (!=0) => kappa!=1. "
          f"sign(kappa-1)=sign(tau): kappa_LATE={r['kap_LATE']:.4f} {'<1' if r['kap_LATE']<1 else '>=1'}")
    print(f"  CLAIM4  D bounds [{r['D_L']:.6f}, {r['D_U']:.6f}] contain D_true={r['D_true']:.6f}: "
          f"{r['D_L']<=r['D_true']<=r['D_U']}")
    print(f"          => kappa_LATE set [{r['kap_lo']:.4f}, {r['kap_hi']:.4f}]  "
          f"contains truth {r['kap_LATE']:.4f}: {r['kap_lo']<=r['kap_LATE']<=r['kap_hi']}; "
          f"entire set < 1: {r['kap_hi']<1}")
print("\n" + "="*118)
print("SUMMARY: tau=ATE_LATE point-ID by IV; kappa_LATE=1 <=> tau=0 (test needs no physics/D);")
print("kappa magnitude only partially-ID via D=|NDE_LATE| bounds; naive NDE_pop plug is biased.")
print("="*118)
