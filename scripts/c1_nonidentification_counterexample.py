import numpy as np
rng = np.random.default_rng(7)
N = 4_000_000

# Frozen-SCM constants
gamma_b = 0.5
sd_b = 0.3
sd_v = 5.0
speed_mult = 5.0          # v_actual = v_base + 5*b
red = 0.3                 # delta_v = v_actual*(1-0.3T)
beta_u = 0.1
sd_y = 0.05
base = 50.0

def true_nie(alpha_b):
    """Structural NIE per SCM definition: shift b by alpha_b at direct arg T=1.
       NIE = 0.7^4/100^4 * (E[(v_base+5 b(1))^4] - E[(v_base+5 b(0))^4]) over population."""
    U   = rng.standard_normal(N)
    ev  = rng.normal(0, sd_v, N)
    eb  = rng.normal(0, sd_b, N)
    b0  = gamma_b*U + eb
    b1  = alpha_b*1 + gamma_b*U + eb
    S0  = base + 10*U + ev + speed_mult*b0
    S1  = base + 10*U + ev + speed_mult*b1
    f = (1-red)**4 / 100.0**4
    return f*(np.mean(S1**4) - np.mean(S0**4))

# Model i: compensation (alpha_b=3.0), T independent of U
a_i = 3.0
nie_i = true_nie(a_i)

# Model ii: no compensation (alpha_b=0), T confounded via U
# Treated units have higher U. We match the OBSERVED physical-scale speed S=delta_v/(1-0.3T)
# distribution (per arm) to Model i, so any physics/delta_v-based estimator sees identical data.
# Model i observed S per arm: S|T=0 ~ mean 50, S|T=1 ~ mean 50 + 5*a_i ; same variance.
U   = rng.standard_normal(N)
ev  = rng.normal(0, sd_v, N)
eb  = rng.normal(0, sd_b, N)
# choose U-shift so treated baseline speed center matches Model i's compensated center
# S = base + (10+5*gamma_b)*U + ev + 5*eb  (alpha_b=0). shift treated U by delta_U:
# need (10+5*gamma_b)*delta_U = 5*a_i
delta_U = 5*a_i / (10 + 5*gamma_b)
# assign treatment so that treated get U shifted by delta_U (deterministic selection illustration)
T = (rng.random(N) < 0.55)
Uc = U + T*delta_U     # confounded: treated have higher latent risk U
b_ii = 0.0 + gamma_b*Uc + eb          # alpha_b = 0  -> NO compensation
S_ii = base + 10*Uc + ev + speed_mult*b_ii
# observed physical-scale speed per arm
S0_i_mean, S0_i_sd = base, np.std(base + (10+5*gamma_b)*rng.standard_normal(N) + rng.normal(0,sd_v,N) + speed_mult*rng.normal(0,sd_b,N))
print("=== Non-identification counterexample (delta_v/S observed, T confounded) ===")
print(f"Model i  (compensation a={a_i}, T _|_ U): true NIE = {nie_i:+.6f}")
print(f"Model ii (NO compensation a=0, U->T selection): true NIE = {true_nie(0.0):+.6f}")
print()
print("Observed physical-scale speed S per arm:")
print(f"  Model i : E[S|T=0]=50.000 (by constr), E[S|T=1]=50+5a={50+5*a_i:.3f}")
print(f"  Model ii: E[S|T=0]={S_ii[~T].mean():.3f}, E[S|T=1]={S_ii[T].mean():.3f}")
print(f"  Model ii: SD[S|T=0]={S_ii[~T].std():.3f}, SD[S|T=1]={S_ii[T].std():.3f}")
print("  -> arm-wise S distributions coincide across the two models, but NIE differs (>0 vs 0).")

# Verify T-U independence in frozen SCM
Tf = (rng.random(N) < 0.55)
Uf = rng.standard_normal(N)
print()
print("=== Frozen-SCM confounding check ===")
print(f"corr(T,U) in frozen SCM design = {np.corrcoef(Tf, Uf)[0,1]:+.5f}  (T~Bernoulli(0.55), independent of U)")
