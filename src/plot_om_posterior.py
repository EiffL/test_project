"""Marginal posterior of Ωm, with the published Suzuki et al. (2012) values marked."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from union21 import read_posterior

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
FIT = "#2a78d6"  # categorical slot 1: this analysis
PUBLISHED = "#eb6834"  # categorical slot 2: Suzuki+2012


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--posterior", required=True)
    ap.add_argument("--constraint", required=True)
    ap.add_argument("--published-stat", nargs=3, type=float, required=True,
                    metavar=("VALUE", "MINUS", "PLUS"))
    ap.add_argument("--published-sys", nargs=3, type=float, required=True,
                    metavar=("VALUE", "MINUS", "PLUS"))
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    meta, cols = read_posterior(args.posterior)
    c = json.loads(Path(args.constraint).read_text())
    om, post = cols["omega_m"], cols["posterior"]
    post = post / post.max()

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.grid(True, axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(INK_2)
    ax.tick_params(colors=INK_2, labelcolor=INK_2)
    ax.set_yticks([])

    band = (om >= c["omega_m_lower"]) & (om <= c["omega_m_upper"])
    ax.fill_between(om[band], 0, post[band], color=FIT, alpha=0.18, lw=0)
    ax.plot(om, post, color=FIT, lw=2, label="this analysis")
    ax.vlines(c["omega_m"], 0, np.interp(c["omega_m"], om, post), color=FIT, lw=1.2, ls="--")
    ax.annotate(
        f"this analysis\nΩm = {c['omega_m']:.3f} −{c['sigma_minus']:.3f} +{c['sigma_plus']:.3f}",
        xy=(c["omega_m_upper"], np.interp(c["omega_m_upper"], om, post)),
        xytext=(c["omega_m_upper"] + 0.04, 0.72), color=INK, fontsize=9,
        arrowprops={"arrowstyle": "-", "color": INK_2, "lw": 0.7},
    )

    # Published values as points with asymmetric error bars above the curve.
    for y, (val, lo, hi), label, marker in (
        (1.18, args.published_sys, "Suzuki+12 stat+sys", "s"),
        (1.08, args.published_stat, "Suzuki+12 stat", "o"),
    ):
        ax.errorbar(val, y, xerr=[[lo], [hi]], fmt=marker, ms=7, color=PUBLISHED, ecolor=PUBLISHED,
                    elinewidth=2, capsize=0, markeredgecolor=SURFACE, markeredgewidth=1.5, label=label)
        ax.text(val + hi + 0.008, y, f"{label}: {val:.3f} −{lo:.3f} +{hi:.3f}",
                va="center", fontsize=8.5, color=INK_2)

    xmid = c["omega_m"]
    ax.set_xlim(max(0.0, xmid - 0.2), min(1.0, xmid + 0.25))
    ax.set_ylim(0, 1.27)
    ax.set_xlabel("Ωm  (flat ΛCDM)", color=INK)
    est = {"best_fit_dchi2": "best fit, Δχ² = 1", "posterior_median": "median, 16–84%"}[c["estimator"]]
    ax.set_title(
        f"Ωm posterior — {meta['covariance'].replace('_', '+')} errors, offset {meta['offset_treatment']}, {est}",
        color=INK, fontsize=10.5, loc="left",
    )
    ax.legend(loc="upper left", frameon=False, fontsize=8, labelcolor=INK_2)

    fig.savefig(args.output, dpi=160, bbox_inches="tight", facecolor=SURFACE)


if __name__ == "__main__":
    main()
