"""Hubble diagram of Union2.1 with the best-fit flat ΛCDM model and residuals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from union21 import distance_modulus, load_union21

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
DATA = "#8a8984"
MODEL = "#2a78d6"  # categorical slot 1


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_2)
    ax.tick_params(colors=INK_2, labelcolor=INK_2)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mu-z", required=True)
    ap.add_argument("--constraint", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    sn = load_union21(args.mu_z)
    c = json.loads(Path(args.constraint).read_text())
    om, off = c["omega_m"], c["offset_mag"]

    z, mu, sig = sn["z"], sn["mu"], sn["sigma_mu"]
    resid = mu - (distance_modulus(z, om) + off)
    zz = np.geomspace(0.01, 1.6, 400)
    model = distance_modulus(zz, om) + off

    # Inverse-variance weighted residual means in log-z bins (diagonal errors; display only).
    edges = np.geomspace(z.min(), z.max() * 1.0001, 13)
    idx = np.digitize(z, edges) - 1
    bz, bm, be = [], [], []
    for k in range(edges.size - 1):
        sel = idx == k
        if sel.sum() < 3:
            continue
        w = 1.0 / sig[sel] ** 2
        bz.append(np.exp(np.average(np.log(z[sel]), weights=w)))
        bm.append(np.average(resid[sel], weights=w))
        be.append(1.0 / np.sqrt(w.sum()))

    fig, (ax, axr) = plt.subplots(
        2, 1, figsize=(7.2, 6.0), sharex=True, gridspec_kw={"height_ratios": [3, 1.3], "hspace": 0.06}
    )
    fig.patch.set_facecolor(SURFACE)
    for a in (ax, axr):
        style(a)

    ax.errorbar(z, mu, yerr=sig, fmt="o", ms=2.5, color=DATA, ecolor=DATA, elinewidth=0.5, alpha=0.55, lw=0)
    ax.plot(zz, model, color=MODEL, lw=2)
    ax.annotate(
        f"flat ΛCDM, Ωm = {om:.3f}",
        xy=(0.55, distance_modulus(0.55, om)[0] + off),
        xytext=(0.06, 44.6),
        color=INK,
        fontsize=10,
        arrowprops={"arrowstyle": "-", "color": INK_2, "lw": 0.7},
    )
    ax.set_ylabel("distance modulus μ  [mag]", color=INK)
    ax.set_title(
        f"Union2.1 Hubble diagram — {c['n_sn']} SNe Ia ({c['covariance'].replace('_', '+')} errors)",
        color=INK, fontsize=11, loc="left",
    )

    axr.axhline(0, color=MODEL, lw=2)
    axr.errorbar(z, resid, yerr=sig, fmt="o", ms=2.5, color=DATA, ecolor=DATA, elinewidth=0.5, alpha=0.4, lw=0)
    axr.errorbar(bz, bm, yerr=be, fmt="o", ms=5, color=INK, ecolor=INK, elinewidth=1.2, capsize=0, lw=0,
                 markeredgecolor=SURFACE, markeredgewidth=1.0, label="binned weighted mean")
    axr.set_ylim(-1.0, 1.0)
    axr.set_xscale("log")
    axr.set_xlim(0.012, 1.6)
    axr.set_xlabel("redshift z", color=INK)
    axr.set_ylabel("μ − model", color=INK)
    axr.legend(loc="lower left", frameon=False, fontsize=8, labelcolor=INK_2)
    axr.set_xticks([0.02, 0.05, 0.1, 0.2, 0.5, 1.0])
    axr.set_xticklabels(["0.02", "0.05", "0.1", "0.2", "0.5", "1"])

    fig.savefig(args.output, dpi=160, bbox_inches="tight", facecolor=SURFACE)


if __name__ == "__main__":
    main()
