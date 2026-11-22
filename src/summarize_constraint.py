"""Summarize the Ωm grid posterior into a point estimate and 68% interval."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from union21 import read_posterior


def best_fit_dchi2(om: np.ndarray, chi2: np.ndarray) -> tuple[float, float, float, float]:
    """Minimum χ² (parabolic refinement) and Δχ² = 1 crossings on each side."""
    i = int(np.argmin(chi2))
    x_min, c_min = om[i], chi2[i]
    if 0 < i < om.size - 1:
        coef = np.polyfit(om[i - 1 : i + 2], chi2[i - 1 : i + 2], 2)
        if coef[0] > 0:
            x_min = -coef[1] / (2 * coef[0])
            c_min = np.polyval(coef, x_min)
    target = c_min + 1.0
    # Left: chi2 decreasing towards the minimum; right: increasing away from it.
    left, right = chi2[: i + 1], chi2[i:]
    lo = float(np.interp(target, left[::-1], om[: i + 1][::-1])) if left.max() > target else float(om[0])
    hi = float(np.interp(target, right, om[i:])) if right.max() > target else float(om[-1])
    return float(x_min), lo, hi, float(c_min)


def posterior_median(om: np.ndarray, post: np.ndarray) -> tuple[float, float, float]:
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (post[1:] + post[:-1]) * np.diff(om))])
    cdf /= cdf[-1]
    lo, med, hi = np.interp([0.15865525, 0.5, 0.84134475], cdf, om)
    return float(med), float(lo), float(hi)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--posterior", required=True)
    ap.add_argument("--estimator", required=True, choices=["best_fit_dchi2", "posterior_median"])
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    meta, cols = read_posterior(args.posterior)
    om, chi2, post, offset = cols["omega_m"], cols["chi2"], cols["posterior"], cols["offset"]

    x_bf, lo_bf, hi_bf, chi2_min = best_fit_dchi2(om, chi2)
    if args.estimator == "best_fit_dchi2":
        x, lo, hi = x_bf, lo_bf, hi_bf
    else:
        x, lo, hi = posterior_median(om, post)

    n_sn, n_params = int(meta["n_sn"]), int(meta["n_params"])
    dof = n_sn - n_params
    result = {
        "omega_m": round(x, 5),
        "omega_m_lower": round(lo, 5),
        "omega_m_upper": round(hi, 5),
        "sigma_minus": round(x - lo, 5),
        "sigma_plus": round(hi - x, 5),
        "estimator": args.estimator,
        "covariance": meta["covariance"],
        "offset_treatment": meta["offset_treatment"],
        "omega_m_at_chi2_min": round(x_bf, 5),
        "chi2_min": round(chi2_min, 3),
        "dof": dof,
        "chi2_per_dof": round(chi2_min / dof, 4),
        "offset_mag": round(float(np.interp(x, om, offset)), 5),
        "h0_reference": float(meta["h0_reference"]),
        "n_sn": n_sn,
        "interval_hits_prior_edge": bool(lo <= om[0] or hi >= om[-1]),
    }
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
