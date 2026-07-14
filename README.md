# Compensation-Collider Mediation under Crash-Conditional Sampling — Code

Reproduction code for the paper *"An Identification Hierarchy for
Compensation-Collider Mediation under Crash-Conditional Sampling"*
(Keito Inoshita, Akira Kawai).

The paper studies whether driver **risk compensation** cancels the engineering
benefit of a safety device, when the compensation behavior is a latent
mediator, device adoption is confounded, and crash records are observed only
conditional on a crash (Berkson selection). The main results are obtained on a
**semi-synthetic structural causal model (SCM)** frozen in the study
pre-registration; the real-data components use **public data only** (see below).

## Requirements

- Python >= 3.10
- Install dependencies:

```bash
pip install -r requirements.txt
```

`geopandas`, `shapely`, and `requests` are only needed for the optional
`data_checks/` scripts.

## Layout

```
code_release/
  scripts/       Semi-synthetic experiments (main + exploratory) and calibration
  data_checks/   Public-data feasibility checks (auto-download from NHTSA/FHWA)
  results/       Expected numerical outputs (CSV/JSON) for the semi-synthetic runs
  requirements.txt
  LICENSE        (MIT)
```

Every script resolves paths relative to its own location, so the tree can be
placed anywhere. Outputs are written to `code_release/results/`.

## How to run

The confirmatory scripts reuse the calibrated cell parameters and some
intermediate CSVs already provided in `results/`. To reproduce the main
confirmatory family (Bonferroni-Holm K=5):

```bash
cd scripts
python confirmatory_rerun_postfreeze.py   # H1 (HC_NI), H2 (HC_T2), H5 (HC3 null)
python hc1_hc2_confirmatory.py            # H3 (HC1), H4 (HC2); completes Holm K=5
```

Each script prints a summary and writes CSV/JSON to `../results/`. Runtime is
CPU-only (no GPU); the largest runs use N = 10^6 per seed over 5 seeds and take
a few minutes each on a modern laptop.

## Script → paper mapping

**Confirmatory (main results)**

| Script | Produces |
|---|---|
| `hc_t2_reduction.py` | Homeostasis point test via the identity kappa_LATE = 1 + tau/D; IV-Wald size/power (Sec. 5.3) |
| `hc_t2_power_curve.py` | Power curve vs. replication count (Fig. 4, Sec. 5.3) |
| `non_identification_counterexample.py` | Tier-3 non-identification lemma HC_NI (Sec. 5.2) |
| `c1_nonidentification_counterexample.py` | Compact first-principles version of the same counterexample |
| `nc1_b2_exclusion_sensitivity.py` | Robustness to instrument exclusion-restriction violation NC1 (Fig. 3, Sec. 5.5) |
| `confirmatory_rerun_postfreeze.py` | Deterministic post-freeze re-run of H1/H2/H5 with Bonferroni-Holm |
| `hc1_hc2_confirmatory.py` | HC1 (physics-constrained DR) and HC2 (IPSW collider correction); completes Holm K=5 |

**SCM calibration (Sec. 5.1 regime setup)**

| Script | Produces |
|---|---|
| `hierarchy_regime_calibration.py` | Confounding/effect-size grid for the identification-strength tiers |
| `kappa_grid_calibration.py`, `kappa_grid_g4_recalibration.py` | kappa grid and the kappa = 1 null cell (G4) |
| `kappa_LATE_true_calibration.py`, `true_nie_kappa_monte_carlo.py` | Ground-truth LATE / NIE / kappa values by Monte Carlo |

**Planned exploratory analyses (Sec. 5.6)**

| Script | Produces |
|---|---|
| `exp2_bounds_vs_pointtest.py` | Partial-identification bounds test vs. the point test |
| `exp3_overid_test.py` | Over-identification (second-instrument) check |
| `exp4_ablation.py` | Ablation isolating the role of the known physics |
| `pa1_proximal_mediation.py` | Proximal-mediation baseline (targets population ATE, complementary estimand) |
| `pa2_h_sensitivity.py` | Sensitivity of the collider correction to exposure-model misspecification |

**Theory verification**

| Script | Produces |
|---|---|
| `c2_bounds_verification.py` | First-principles check of the kappa_LATE = 1 + tau/D identification map |

**Real-data analysis on CRSS (exploratory; Sec. 5.6)**

These run on the NHTSA Crash Report Sampling System (CRSS) 2016-2021, with the
treatment constructed from the NHTSA 5-Star Safety Ratings API. This section of
the paper is exploratory and was not part of the pre-registered confirmatory
family. Its headline finding is negative: the mandate-year instrument is
invalid.

| Script | Produces |
|---|---|
| `realdata_tier2_pointtest.py` | Tier-2 point test on CRSS; builds T from the NHTSA 5-Star Safety Ratings API |
| `realdata_tier2_fastresult.py` | Same estimator, cached-API fast path (v1, no vehicle-type filter) |
| `realdata_tier2_fastresult_v2.py` | v2: restricts the population to FMVSS 126 scope via `BODY_TYP`; supersedes v1 |
| `realdata_falsification_test.py` | **Negative-control falsification of the instrument (Table 3).** Placebo stratum (stopped, struck front-to-rear) vs. rollover stratum |
| `realdata_changeover_design.py` | Within-make-model changeover diagnostic. Reported as a diagnostic only: the first stage is unity, so this is difference-in-differences, not IV, and lies outside Theorem 3 |
| `realdata_ds_sensitivity.py` | Sensitivity of the real-data estimates to the treatment-coding rule |

**Public-data feasibility checks (`data_checks/`)**

These download public data on first run and cache it under
`code_release/data/raw/`. They document which assumptions of the identification
map the U.S. crash files can and cannot support.

| Script | Data source (public) | Purpose |
|---|---|---|
| `step1_fars_coord_check.py` | NHTSA FARS (accident files) | Coordinate coverage of crash records |
| `step1b_spatial_join_test.py` | NHTSA FARS + FHWA HPMS / BTS NTAD | FARS-to-road-network spatial-join feasibility |
| `step2_deltav_check.py` | NHTSA FARS / CRSS | Confirms Delta_v is absent from the public files |
| `audit_us_crash_databases.py` | NHTSA FARS / CRSS / CISS / vPIC | Which variable each tier of the map needs, against what each file records |

## Data availability and integrity

- **Public data only.** Real-data components use the NHTSA CRSS, FARS, the NHTSA
  5-Star Safety Ratings API, and FHWA HPMS, all openly available. Restricted
  crash-severity datasets are **not** used. No individual-level data is required.
  CRSS files are downloaded from NHTSA on first run and are not redistributed here.
- **Semi-synthetic confirmatory core.** The pre-registered confirmatory results
  are computed on a frozen SCM. The CRSS analysis in `realdata_*.py` is
  exploratory and is labelled as such in the paper.
- **No fabrication, no cherry-picking.** All seeds are reported; pre-specified
  threshold failures are reported as-is in the accompanying result files. The
  real-data section reports a falsification of the paper's own preferred
  instrument, and analyses that were run but not used to support any claim in the
  paper (the changeover design, the narrow-Z contrast) are included here in full.

## License

MIT (see `LICENSE`).
