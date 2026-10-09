"""
Stratum contrasts, detectable effects, and forest plot (AAP R1 revision)
========================================================================
Reads the frozen falsification-test result file and derives, without touching
the CRSS microdata:

  * pairwise z-tests of stratum-specific Wald estimates
    (full sample, negative control, rollover);
  * the minimum detectable |tau| at 80% power, two-sided alpha = 0.05,
    MDE = (z_{0.975} + z_{0.80}) * SE;
  * a forest plot of the three estimates with 95% CIs.

The negative-control and rollover strata are essentially disjoint (9 shared
records in 2016), so their contrast uses independent SEs.  Each stratum is a
subset of the full sample; the positive covariance with the full-sample
estimate makes an independence-based SE conservative for those contrasts.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "results" / "falsification_test_20260714_044932.json"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "font.size": 10,
})


def main() -> None:
    d = json.load(open(SRC))
    strata = {
        "full": d["full_sample"]["wald"],
        "negative_control": d["crash_type_groups"]["placebo_stopped_rear_ended"],
        "rollover": d["crash_type_groups"]["esc_active_rollover"],
    }
    z_mde = scipy.stats.norm.ppf(0.975) + scipy.stats.norm.ppf(0.80)
    out = {"source": SRC.name, "created": datetime.now().isoformat(),
           "mde_multiplier": float(z_mde), "strata": {}, "contrasts": {}}
    for k, s in strata.items():
        out["strata"][k] = {"tau": s["tau"], "SE": s["SE_tau"],
                            "CI": [s["CI_L"], s["CI_U"]], "p": s["p_value"],
                            "RF": s["RF"], "FS": s["FS"],
                            "MDE_80": float(z_mde * s["SE_tau"])}
    for a, b, note in [("negative_control", "rollover", "independent"),
                       ("negative_control", "full", "conservative"),
                       ("rollover", "full", "conservative")]:
        diff = strata[a]["tau"] - strata[b]["tau"]
        se = float(np.hypot(strata[a]["SE_tau"], strata[b]["SE_tau"]))
        z = diff / se
        out["contrasts"][f"{a}_minus_{b}"] = {
            "diff": float(diff), "SE": se, "z": float(z),
            "p": float(2 * scipy.stats.norm.sf(abs(z))), "SE_type": note}

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    res_path = ROOT / "results" / f"r1_stratum_contrasts_{stamp}.json"
    json.dump(out, open(res_path, "w"), indent=2)
    print(json.dumps(out, indent=2))

    # Forest plot (top to bottom: full, negative control, rollover)
    labels = ["Full sample",
              "Stopped, struck from behind\n(negative control)",
              "Rollover"]
    keys = ["full", "negative_control", "rollover"]
    ns = [d["full_sample"]["n_analytic"],
          d["crash_type_groups"]["placebo_stopped_rear_ended"]["n_subset"],
          d["crash_type_groups"]["esc_active_rollover"]["n_subset"]]
    y = np.arange(len(keys))[::-1]
    fig, ax = plt.subplots(figsize=(6.3, 2.5))
    ax.axvline(0.0, color="0.45", lw=0.9, ls="--", zorder=1)
    for yi, k in zip(y, keys):
        s = out["strata"][k]
        color = "#B23A48" if k == "negative_control" else "#2F4B7C"
        ax.plot(s["CI"], [yi, yi], color=color, lw=1.6, zorder=2)
        ax.plot(s["tau"], yi, "o", color=color, ms=6, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel(r"Wald estimate $\hat{\tau}$ with 95% confidence interval")
    ax.set_xlim(-0.085, 0.045)
    ax.set_ylim(-0.6, 2.6)
    for yi, k, n in zip(y, keys, ns):
        p = out["strata"][k]["p"]
        ptxt = "p < 0.0001" if p < 1e-4 else f"p = {p:.2f}"
        ax.text(1.02, yi, f"n = {n:,}\n{ptxt}", transform=ax.get_yaxis_transform(),
                va="center", ha="left", fontsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.subplots_adjust(left=0.30, right=0.80, bottom=0.22, top=0.97)
    fig_dir = ROOT / "results"
    fig.savefig(fig_dir / "fig_r1_falsification_forest.pdf")
    fig.savefig(fig_dir / "fig_r1_falsification_forest.png", dpi=300)
    print("saved", res_path, fig_dir / "fig_r1_falsification_forest.pdf")


if __name__ == "__main__":
    main()
