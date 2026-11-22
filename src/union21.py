"""Shared helpers: Union2.1 table I/O, flat ΛCDM distances, posterior CSV I/O."""

from __future__ import annotations

from pathlib import Path

import numpy as np

C_KM_S = 299_792.458
# The published Union2.1 distance moduli assume h = 0.70 (see the file header).
H0_PUBLISHED = 70.0

_GL_X, _GL_W = np.polynomial.legendre.leggauss(64)


def load_union21(path: str | Path) -> dict[str, np.ndarray]:
    """Read SCPUnion2.1_mu_vs_z.txt: name, z, mu, sigma_mu, P(low-mass host)."""
    names, z, mu, sig, plow = [], [], [], [], []
    for line in Path(path).read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        cols = line.split()
        names.append(cols[0])
        z.append(float(cols[1]))
        mu.append(float(cols[2]))
        sig.append(float(cols[3]))
        plow.append(float(cols[4]))
    return {
        "name": np.array(names),
        "z": np.array(z),
        "mu": np.array(mu),
        "sigma_mu": np.array(sig),
        "p_lowmass": np.array(plow),
    }


def comoving_distance_hubble_units(z: np.ndarray, omega_m: float) -> np.ndarray:
    """D_C / D_H in flat ΛCDM: ∫_0^z dz'/E(z'), by 64-point Gauss–Legendre."""
    z = np.atleast_1d(np.asarray(z, dtype=float))
    zz = 0.5 * z[:, None] * (_GL_X[None, :] + 1.0)
    inv_e = 1.0 / np.sqrt(omega_m * (1.0 + zz) ** 3 + (1.0 - omega_m))
    return 0.5 * z * (inv_e @ _GL_W)


def distance_modulus(z: np.ndarray, omega_m: float, h0: float = H0_PUBLISHED) -> np.ndarray:
    """μ(z) = 5 log10(D_L / Mpc) + 25 in flat ΛCDM (radiation neglected)."""
    z = np.atleast_1d(np.asarray(z, dtype=float))
    d_l = (1.0 + z) * (C_KM_S / h0) * comoving_distance_hubble_units(z, omega_m)
    return 5.0 * np.log10(d_l) + 25.0


POSTERIOR_COLUMNS = ("omega_m", "chi2", "posterior", "offset")


def write_posterior(path: str | Path, meta: dict, columns: dict[str, np.ndarray]) -> None:
    """CSV with `# key=value` metadata lines, then a header row and the grid."""
    lines = [f"# {k}={v}" for k, v in meta.items()]
    lines.append(",".join(POSTERIOR_COLUMNS))
    rows = np.column_stack([columns[c] for c in POSTERIOR_COLUMNS])
    lines += [",".join(f"{x:.10g}" for x in row) for row in rows]
    Path(path).write_text("\n".join(lines) + "\n")


def read_posterior(path: str | Path) -> tuple[dict[str, str], dict[str, np.ndarray]]:
    meta: dict[str, str] = {}
    body: list[str] = []
    for line in Path(path).read_text().splitlines():
        if line.startswith("#"):
            key, _, value = line[1:].strip().partition("=")
            meta[key] = value
        elif line.strip():
            body.append(line)
    header = body[0].split(",")
    data = np.array([[float(x) for x in row.split(",")] for row in body[1:]])
    return meta, {name: data[:, i] for i, name in enumerate(header)}
