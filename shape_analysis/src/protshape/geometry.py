"""Point-cloud shape metrics: size, gyration-tensor descriptors, convex hull, contacts.

All functions take plain arrays so they can be unit-tested on synthetic point clouds.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import ConvexHull, cKDTree
from scipy.spatial.distance import pdist

# 14 directions (6 axes + 8 cube diagonals) used to expand atoms to their vdW surface for the hull.
_DIRS = np.vstack([np.eye(3), -np.eye(3),
                   np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)]) / np.sqrt(3)])


def radius_of_gyration(x: np.ndarray, w: np.ndarray | None = None) -> float:
    w = np.ones(len(x)) if w is None else np.asarray(w, float)
    c = (x * w[:, None]).sum(0) / w.sum()
    return float(np.sqrt((w * ((x - c) ** 2).sum(1)).sum() / w.sum()))


def gyration_tensor(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """S = (1/N) sum (r - r_cm)(r - r_cm)^T. Returns (eigenvalues l1>=l2>=l3, eigenvectors as columns, centre)."""
    c = x.mean(0)
    d = x - c
    s = d.T @ d / len(x)
    vals, vecs = np.linalg.eigh(s)
    order = np.argsort(vals)[::-1]
    return np.clip(vals[order], 0, None), vecs[:, order], c


def gyration_descriptors(x: np.ndarray, prefix: str = "") -> dict:
    """Equivalent-ellipsoid axes, axial ratios, anisotropy, asphericity, acylindricity, prolateness."""
    l, vecs, _ = gyration_tensor(x)
    l1, l2, l3 = l
    tr = l.sum()
    rg2 = tr
    a, b, c = np.sqrt(5 * l1), np.sqrt(5 * l2), np.sqrt(5 * l3)
    lbar = tr / 3
    eps = 1e-12
    out = {
        "lambda1": l1, "lambda2": l2, "lambda3": l3,
        "ax_a": a, "ax_b": b, "ax_c": c,
        "a_over_c": a / max(c, eps), "a_over_b": a / max(b, eps), "b_over_c": b / max(c, eps),
        "kappa2": 1 - 3 * (l1 * l2 + l2 * l3 + l3 * l1) / max(tr ** 2, eps),
        "asphericity": (l1 - (l2 + l3) / 2) / max(rg2, eps),
        "acylindricity": (l2 - l3) / max(rg2, eps),
        "prolateness": float(np.prod(l - lbar) / max(lbar ** 3, eps)),
        "pa_x": vecs[0, 0], "pa_y": vecs[1, 0], "pa_z": vecs[2, 0],
    }
    return {prefix + k: float(v) for k, v in out.items()}


def rmin_empirical(n_res: int) -> float:
    """Compact-protein reference radius of gyration, 2.2 N^0.38 A."""
    return 2.2 * n_res ** 0.38


def rmin_sphere(volume: float) -> float:
    """Rg of a uniform sphere with the given volume: sqrt(3/5) R."""
    return float(np.sqrt(3 / 5) * (3 * volume / (4 * np.pi)) ** (1 / 3))


def sphericity(volume: float, area: float) -> float:
    """pi^(1/3) (6V)^(2/3) / A: 1 for a sphere, smaller otherwise."""
    if not area or not np.isfinite(area):
        return float("nan")
    return float(np.pi ** (1 / 3) * (6 * volume) ** (2 / 3) / area)


def hull_metrics(x: np.ndarray, radii: np.ndarray | None = None) -> dict:
    """Convex hull of the atoms' vdW surfaces (14-point expansion) and Dmax over hull vertices of centres."""
    out = {"dmax": np.nan, "v_hull": np.nan, "a_hull": np.nan, "sphericity_hull": np.nan}
    if len(x) < 4:
        return out
    try:
        hc = ConvexHull(x)
        v = x[hc.vertices]
        out["dmax"] = float(pdist(v).max()) if len(v) > 1 else 0.0
        pts = x if radii is None else (x[:, None, :] + radii[:, None, None] * _DIRS[None]).reshape(-1, 3)
        h = ConvexHull(pts)
        out["v_hull"], out["a_hull"] = float(h.volume), float(h.area)
        out["sphericity_hull"] = sphericity(h.volume, h.area)
    except Exception:  # degenerate (coplanar) point sets
        pass
    return out


def contact_density(ca: np.ndarray, cutoff: float = 8.0) -> float:
    """Mean number of CA neighbours within `cutoff` A (excluding self)."""
    if len(ca) < 2:
        return 0.0
    tree = cKDTree(ca)
    counts = tree.query_ball_point(ca, cutoff, return_length=True)
    return float(np.mean(np.asarray(counts) - 1))


def sasa(x: np.ndarray, radii: np.ndarray) -> np.ndarray:
    """Per-atom solvent-accessible surface area (A^2), Lee-Richards via freesasa, probe 1.4 A."""
    import freesasa

    freesasa.setVerbosity(freesasa.silent)
    res = freesasa.calcCoord(np.ascontiguousarray(x, dtype=float).ravel().tolist(), radii.tolist())
    return np.array([res.atomArea(i) for i in range(len(x))])
