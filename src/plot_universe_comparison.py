"""Overlay the Ωm posteriors of several universes, each beside its published value."""

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
# Categorical slots in fixed order: the n-th --universe gets the n-th hue.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
COV_LABEL = {"stat": "stat only", "stat_sys": "stat + sys"}
PUB_Y0, PUB_DY = 1.14, 0.09  # heights of the published-value rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--universe", nargs=3, action="append", required=True,
                    metavar=("NAME", "POSTERIOR", "CONSTRAINT"))
    ap.add_argument("--published", nargs=4, action="append", default=[],
                    metavar=("NAME", "VALUE", "MINUS", "PLUS"),
                    help="Published value to set beside universe NAME.")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    if len(args.universe) > len(SERIES):
        raise SystemExit(f"at most {len(SERIES)} universes")
    published = {name: tuple(map(float, rest)) for name, *rest in args.published}

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.grid(True, axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(INK_2)
    ax.tick_params(colors=INK_2, labelcolor=INK_2)
    ax.set_yticks([])

    lows, highs, series = [], [], []
    for k, (name, post_path, cons_path) in enumerate(args.universe):
        color = SERIES[k]
        meta, cols = read_posterior(post_path)
        c = json.loads(Path(cons_path).read_text())
        om, post = cols["omega_m"], cols["posterior"]
        post = post / post.max()
        cov = COV_LABEL.get(meta["covariance"], meta["covariance"])

        band = (om >= c["omega_m_lower"]) & (om <= c["omega_m_upper"])
        ax.fill_between(om[band], 0, post[band], color=color, alpha=0.15, lw=0)
        ax.plot(om, post, color=color, lw=2,
                label=f"{name} ({cov}): {c['omega_m']:.3f} −{c['sigma_minus']:.3f} +{c['sigma_plus']:.3f}")
        ax.vlines(c["omega_m"], 0, 1, color=color, lw=1.0, ls="--")
        lows.append(c["omega_m_lower"] - 2 * c["sigma_minus"])
        highs.append(c["omega_m_upper"] + 2 * c["sigma_plus"])

        series.append((name, c, om, post))

        if name in published:
            val, lo, hi = published[name]
            y = PUB_Y0 + PUB_DY * k
            ax.errorbar(val, y, xerr=[[lo], [hi]], fmt="s", ms=6.5, color=color, ecolor=color,
                        elinewidth=2, capsize=0, markeredgecolor=SURFACE, markeredgewidth=1.5,
                        label=f"Suzuki+12 ({cov}): {val:.3f} −{lo:.3f} +{hi:.3f}")

    # Direct labels: the narrowest curve is labelled over its peak, where it
    # stands above the others; each wider one on its right tail, where it is
    # the outermost curve, at successively lower heights.
    by_width = sorted(series, key=lambda s: s[1]["sigma_minus"] + s[1]["sigma_plus"])
    for rank, (name, c, om, post) in enumerate(by_width):
        if rank == 0:
            ax.text(c["omega_m"], 1.025, name, color=INK, fontsize=9.5, ha="center", va="bottom")
            continue
        level = 0.45 - 0.12 * (rank - 1)
        right = om > om[np.argmax(post)]
        x_at = float(np.interp(level, post[right][::-1], om[right][::-1]))
        ax.annotate(name, xy=(x_at, level), xytext=(x_at + 0.03, level + 0.08),
                    color=INK, fontsize=9.5, va="center", ha="left",
                    arrowprops={"arrowstyle": "-", "color": INK_2, "lw": 0.7})

    ax.set_xlim(max(0.0, min(lows)), min(1.0, max(highs)))
    ax.set_ylim(0, PUB_Y0 + PUB_DY * len(args.universe) + 0.04)
    ax.set_xlabel("Ωm  (flat ΛCDM)", color=INK)
    ax.set_title("Union2.1 SN-only Ωm: universes compared (lines) with Suzuki+12 Table 7 (squares)",
                 color=INK, fontsize=10.5, loc="left")
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, -0.14), ncol=2, frameon=False,
              fontsize=8, labelcolor=INK_2)

    fig.savefig(args.output, dpi=160, bbox_inches="tight", facecolor=SURFACE)


if __name__ == "__main__":
    main()
