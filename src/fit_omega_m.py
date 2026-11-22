"""Grid posterior of Ωm in flat ΛCDM from the Union2.1 Hubble diagram.

χ²(Ωm, m) = (Δ − m·1)ᵀ C⁻¹ (Δ − m·1),  Δ = μ_obs − μ_th(z; Ωm, h=0.70)

The additive offset m absorbs the SN absolute magnitude and H0. With a flat
prior on m, marginalizing gives −2 ln L = Δᵀ C⁻¹ Δ − B²/A + const, where
A = 1ᵀ C⁻¹ 1 and B = 1ᵀ C⁻¹ Δ. That equals the profile χ² minimized over m
(at m = B/A), so `chi2` below is the same either way. With `fixed_h70`, m = 0.
"""

from __future__ import annotations

import argparse

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from union21 import distance_modulus, load_union21, write_posterior

OM_GRID = np.linspace(0.0, 1.0, 4001)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mu-z", required=True)
    ap.add_argument("--cov-stat", required=True)
    ap.add_argument("--cov-sys", required=True)
    ap.add_argument("--covariance", required=True, choices=["stat", "stat_sys"])
    ap.add_argument("--offset", required=True, choices=["marginalized", "fixed_h70"])
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    sn = load_union21(args.mu_z)
    n = sn["z"].size
    cov = np.loadtxt(args.cov_stat if args.covariance == "stat" else args.cov_sys)
    if cov.shape != (n, n):
        raise SystemExit(f"covariance shape {cov.shape} does not match {n} SNe")

    # Residuals for every grid point at once: shape (n_sn, n_grid).
    mu_th = np.column_stack([distance_modulus(sn["z"], om) for om in OM_GRID])
    delta = sn["mu"][:, None] - mu_th

    cho = cho_factor(cov, lower=True)
    cinv_delta = cho_solve(cho, delta)
    chi2_raw = np.einsum("ij,ij->j", delta, cinv_delta)

    if args.offset == "marginalized":
        cinv_one = cho_solve(cho, np.ones(n))
        a = cinv_one.sum()
        b = cinv_one @ delta
        chi2 = chi2_raw - b**2 / a
        offset = b / a
        n_params = 2
    else:
        chi2 = chi2_raw
        offset = np.zeros_like(OM_GRID)
        n_params = 1

    post = np.exp(-0.5 * (chi2 - chi2.min()))
    post /= np.trapezoid(post, OM_GRID)

    write_posterior(
        args.output,
        meta={
            "covariance": args.covariance,
            "offset_treatment": args.offset,
            "n_sn": n,
            "n_params": n_params,
            "prior": "flat omega_m in [0, 1]",
            "h0_reference": 70.0,
        },
        columns={"omega_m": OM_GRID, "chi2": chi2, "posterior": post, "offset": offset},
    )


if __name__ == "__main__":
    main()
