"""1 A voxel pipeline: vdW and probe masks, flood-filled solvent, molecular (SES) interior,
cavities, burial depth, and the distribution of hydrophobicity inside the protein.

Solvent is the connected component of probe-centre-accessible space that touches the box boundary;
the molecular interior is everything farther than one probe radius from that solvent. Two kinds of
empty interior space are reported:

* ``void_interstitial`` - the spec's literal definition, interior & ~atom_mask: every gap between
  vdW spheres inside the molecular surface (typically ~25-35% of the molecular volume);
* ``cavity_*`` - enclosed cavities: probe-centre-accessible space NOT connected to the outside,
  grown by the probe radius and minus the atoms (the usual definition of an internal cavity).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from .scales import PROBE_RADIUS


class BoxTooLarge(Exception):
    pass


# Fraction of a voxel by which the discrete solvent boundary lies beyond the true one (calibrated on
# spheres in tests/test_voxel_synthetic.py).
BOUNDARY_CORRECTION = 0.2


def _stamp(mask: np.ndarray, centres_idx: np.ndarray, radii_vox: np.ndarray, spacing: float) -> None:
    """Set mask voxels whose centres lie within radius of each atom centre (all in voxel units)."""
    for r in np.unique(np.round(radii_vox, 3)):
        sel = np.isclose(radii_vox, r, atol=1e-3)
        pts = centres_idx[sel]
        k = int(np.ceil(r)) + 1
        g = np.arange(-k, k + 1)
        offs = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
        base = np.floor(pts).astype(int)
        for i in range(0, len(pts), 1500):
            b = base[i:i + 1500]
            p = pts[i:i + 1500]
            cand = b[:, None, :] + offs[None]                      # (n, m, 3)
            d2 = ((cand - p[:, None, :]) ** 2).sum(-1)
            ok = d2 <= r * r
            c = cand[ok]
            inside = np.all((c >= 0) & (c < np.array(mask.shape)), axis=1)
            c = c[inside]
            mask[c[:, 0], c[:, 1], c[:, 2]] = True


def _mesh_area(mask: np.ndarray, spacing: float) -> float:
    from skimage.measure import marching_cubes, mesh_surface_area

    if mask.sum() < 2:
        return float("nan")
    vol = ndimage.gaussian_filter(np.pad(mask, 2).astype(np.float32), sigma=0.75)
    if vol.max() <= 0.5:          # region too thin to have a surface at this resolution
        return float("nan")
    verts, faces, _, _ = marching_cubes(vol, level=0.5, spacing=(spacing,) * 3)
    return float(mesh_surface_area(verts, faces))


def voxel_metrics(coords: np.ndarray, radii: np.ndarray, atom_h: dict[str, np.ndarray],
                  res_index: np.ndarray, principal_axes: np.ndarray, spacing: float = 1.0,
                  pad: float = 10.0, max_box: int = 256, probe: float = PROBE_RADIUS) -> tuple[dict, dict]:
    """Return (unit metrics, per-atom arrays {'depth': ...}).

    atom_h maps scale name -> per-atom hydrophobicity; principal_axes are the gyration-tensor
    eigenvectors (columns, longest first) of the unit.
    """
    margin = pad + float(radii.max())
    lo = coords.min(0) - margin
    shape = np.ceil((coords.max(0) + margin - lo) / spacing).astype(int) + 1
    if shape.max() > max_box:
        raise BoxTooLarge(f"box {tuple(shape)} exceeds {max_box} voxels per side")
    centres = (coords - lo) / spacing                              # voxel units; voxel i centre at i
    atom_mask = np.zeros(shape, bool)
    probe_mask = np.zeros(shape, bool)
    _stamp(atom_mask, centres, radii / spacing, spacing)
    _stamp(probe_mask, centres, (radii + probe) / spacing, spacing)

    free = ~probe_mask
    lab, nlab = ndimage.label(free)
    faces = np.concatenate([lab[0].ravel(), lab[-1].ravel(), lab[:, 0].ravel(), lab[:, -1].ravel(),
                            lab[:, :, 0].ravel(), lab[:, :, -1].ravel()])
    boundary = np.unique(faces[faces > 0])
    solvent = np.isin(lab, boundary)
    enclosed_probe = free & ~solvent                               # probe centres inside the protein

    # Distances are measured to the centre of the nearest solvent voxel; the solvent region itself
    # starts half a voxel earlier, so subtract that half voxel to avoid inflating the interior.
    dist_solv = ndimage.distance_transform_edt(~solvent) * spacing - BOUNDARY_CORRECTION * spacing
    interior = dist_solv > probe                                   # inside the molecular surface
    vox = spacing ** 3
    v_atoms = float(atom_mask.sum() * vox)
    v_ses = float(interior.sum() * vox)
    out = {"v_vdw": v_atoms, "v_ses": v_ses, "a_ses": _mesh_area(interior, spacing),
           "void_interstitial": float((interior & ~atom_mask).sum() * vox)}
    out["void_interstitial_frac"] = out["void_interstitial"] / v_ses if v_ses else np.nan

    # enclosed cavities
    if enclosed_probe.any():
        cav = (ndimage.distance_transform_edt(~enclosed_probe) * spacing <= probe) & ~atom_mask & interior
    else:
        cav = np.zeros(shape, bool)
    clab, ncav = ndimage.label(cav)
    sizes = np.bincount(clab.ravel())[1:] * vox if ncav else np.array([])
    keep = sizes >= 10.0                                           # ignore sub-10 A^3 discretisation specks
    out.update({
        "cavity_n": int(keep.sum()), "cavity_volume": float(sizes[keep].sum()),
        "cavity_largest": float(sizes[keep].max()) if keep.any() else 0.0,
    })
    out["cavity_frac"] = out["cavity_volume"] / v_atoms if v_atoms else np.nan
    cav_kept = np.isin(clab, np.flatnonzero(keep) + 1) if keep.any() else np.zeros(shape, bool)
    out["cavity_lining_area"] = _mesh_area(cav_kept, spacing) if keep.any() else 0.0

    # nearest atom of every voxel (for hydrophobicity assignment and cavity lining)
    seed = np.ones(shape, bool)
    cidx = np.clip(np.rint(centres).astype(int), 0, shape - 1)
    seed[cidx[:, 0], cidx[:, 1], cidx[:, 2]] = False
    atom_at = -np.ones(shape, dtype=np.int32)
    atom_at[cidx[:, 0], cidx[:, 1], cidx[:, 2]] = np.arange(len(coords), dtype=np.int32)
    _, ind = ndimage.distance_transform_edt(seed, return_indices=True)
    nearest = atom_at[ind[0], ind[1], ind[2]]
    del ind

    if keep.any():
        lining = np.unique(nearest[cav_kept])
        out["cavity_lining_atoms"] = int(len(lining))
        out["cavity_lining_h_atom"] = float(atom_h["atom"][lining].mean())
        out["cavity_lining_h_kd"] = float(atom_h["kd"][lining].mean())
    else:
        out.update({"cavity_lining_atoms": 0, "cavity_lining_h_atom": np.nan, "cavity_lining_h_kd": np.nan})

    # burial depth below the molecular surface
    depth_vox = np.clip(dist_solv - probe, 0, None)
    atom_depth = ndimage.map_coordinates(depth_vox, centres.T, order=1, mode="nearest")

    # hydrophobicity distribution over interior voxels
    iv = np.flatnonzero(interior.ravel())
    pos = np.stack(np.unravel_index(iv, shape), 1) * spacing + lo
    near = nearest.ravel()[iv]
    d = depth_vox.ravel()[iv]
    dmax = d.max() if len(d) else 0.0
    dn = d / dmax if dmax > 0 else np.zeros_like(d)
    centre = pos.mean(0)
    rg_geom = np.sqrt(((pos - centre) ** 2).sum(1).mean())
    shell = d <= 3.0                                               # outermost 3 A: the lipid/water-facing shell
    for s, h_atom in atom_h.items():
        h = h_atom[near]
        out.update(_depth_profile(h, dn, s))
        w = np.clip(h, 0, None)
        out[f"hydrophobic_rg_ratio_{s}"] = float(np.sqrt((w * ((pos - centre) ** 2).sum(1)).sum() / w.sum()) / rg_geom) \
            if w.sum() > 0 and rg_geom > 0 else np.nan
        # belt along principal axis 1 (spec) over all interior voxels, and the best of 3 axes on the shell
        out.update(_belt(pos, h, centre, principal_axes[:, 0], f"_{s}"))
        scan = slab_scan(pos[shell], h[shell], centre)
        out[f"belt_contrast_scan_{s}"] = scan["contrast"]
        out[f"belt_normal_angle_pa1_{s}"] = float(np.degrees(np.arccos(min(1.0, abs(scan["normal"] @ principal_axes[:, 0])))))\
            if scan["normal"] is not None else np.nan
        if scan["normal"] is not None:
            b = _belt(pos[shell], h[shell], centre, scan["normal"], "")
            out[f"belt_width_scan_A_{s}"] = b["belt_width_A"]
            out[f"belt_center_scan_A_{s}"] = b["belt_center_A"]
            out[f"belt_profile_contrast_scan_{s}"] = b["belt_contrast"]
        else:
            out.update({f"belt_width_scan_A_{s}": np.nan, f"belt_center_scan_A_{s}": np.nan,
                        f"belt_profile_contrast_scan_{s}": np.nan})
        if s == "kd":
            out["radial_profile_kd"] = _radial(pos, h, centre, principal_axes[:, 0])
            rp = out["radial_profile_kd"]
            out["radial_slope_kd"] = float(np.polyfit(np.arange(len(rp)) * 2.0, rp, 1)[0]) if len(rp) >= 3 else np.nan
    return out, {"depth": atom_depth}


def _depth_profile(h: np.ndarray, dn: np.ndarray, s: str) -> dict:
    bins = np.minimum((dn * 10).astype(int), 9)
    prof = [float(h[bins == b].mean()) if np.any(bins == b) else np.nan for b in range(10)]
    slope = float(np.polyfit(dn, h, 1)[0]) if len(h) > 2 and np.ptp(dn) > 0 else np.nan
    outer, inner = h[dn <= 0.2], h[dn >= 0.8]
    inv = float(outer.mean() - inner.mean()) if len(outer) and len(inner) else np.nan
    return {f"h_depth_profile_{s}": prof, f"h_depth_slope_{s}": slope, f"inversion_index_{s}": inv,
            f"h_mean_{s}": float(h.mean()) if len(h) else np.nan}


def _belt(pos: np.ndarray, h: np.ndarray, centre: np.ndarray, axis: np.ndarray, suffix: str,
          slab: float = 4.0, min_vox: int = 20) -> dict:
    """Widest contiguous run of 4 A slabs (along `axis`) whose mean H exceeds the unit's median voxel H."""
    nan = {f"belt_width_A{suffix}": np.nan, f"belt_center_A{suffix}": np.nan, f"belt_contrast{suffix}": np.nan}
    if len(h) < min_vox * 3:
        return nan
    t = (pos - centre) @ axis
    k = np.floor((t - t.min()) / slab).astype(int)
    nb = k.max() + 1
    cnt = np.bincount(k, minlength=nb)
    mean = np.bincount(k, weights=h, minlength=nb) / np.maximum(cnt, 1)
    med = np.median(h)
    above = (mean > med) & (cnt >= min_vox)
    best_len, best_start, run, start = 0, -1, 0, 0
    for i, a in enumerate(above):
        if a:
            if run == 0:
                start = i
            run += 1
            if run > best_len:
                best_len, best_start = run, start
        else:
            run = 0
    if best_len == 0:
        return {f"belt_width_A{suffix}": 0.0, f"belt_center_A{suffix}": np.nan, f"belt_contrast{suffix}": 0.0}
    inrun = (k >= best_start) & (k < best_start + best_len)
    centre_t = (best_start + best_len / 2) * slab + t.min()
    contrast = float(h[inrun].mean() - h[~inrun].mean()) if (~inrun).any() else np.nan
    return {f"belt_width_A{suffix}": float(best_len * slab), f"belt_center_A{suffix}": float(centre_t),
            f"belt_contrast{suffix}": contrast}


def _hemisphere(n: int = 200) -> np.ndarray:
    """Quasi-uniform unit vectors on a hemisphere (Fibonacci lattice)."""
    i = np.arange(n) + 0.5
    z = i / n                                       # z in (0, 1]: one of each +-v pair
    phi = np.pi * (1 + 5 ** 0.5) * i
    r = np.sqrt(1 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


DIRECTIONS = _hemisphere(200)


def slab_scan(pos: np.ndarray, h: np.ndarray, centre: np.ndarray, thickness: float = 30.0,
              min_flank: float = 0.10) -> dict:
    """Most hydrophobic 30 A slab of the surface shell over 200 directions and all slab positions.

    contrast = mean H inside the slab - mean H outside it, requiring at least `min_flank` of the shell
    on EACH side of the slab (a belt must be flanked by surface on both faces).
    """
    best = {"contrast": np.nan, "normal": None, "offset": np.nan}
    if len(h) < 100:
        return best
    d = pos - centre
    tot_n, tot_h = len(h), h.sum()
    for v in DIRECTIONS:
        t = d @ v
        k = np.floor(t - t.min()).astype(int)                     # 1 A bins
        cnt = np.bincount(k)
        sh = np.bincount(k, weights=h)
        cc, ch = np.concatenate([[0], np.cumsum(cnt)]), np.concatenate([[0], np.cumsum(sh)])
        w = int(thickness)
        if len(cnt) <= w:
            continue
        lo = np.arange(0, len(cnt) - w + 1)
        n_in = cc[lo + w] - cc[lo]
        h_in = ch[lo + w] - ch[lo]
        n_below, n_above = cc[lo], tot_n - cc[lo + w]
        ok = (n_below >= min_flank * tot_n) & (n_above >= min_flank * tot_n) & (n_in > 0)
        if not ok.any():
            continue
        con = np.where(ok, h_in / np.maximum(n_in, 1) - (tot_h - h_in) / np.maximum(tot_n - n_in, 1), -np.inf)
        i = int(np.argmax(con))
        if not np.isfinite(best["contrast"]) or con[i] > best["contrast"]:
            best = {"contrast": float(con[i]), "normal": v, "offset": float(lo[i] + w / 2 + t.min())}
    return best


def _radial(pos: np.ndarray, h: np.ndarray, centre: np.ndarray, axis: np.ndarray, step: float = 2.0) -> list:
    d = pos - centre
    rho = np.linalg.norm(d - np.outer(d @ axis, axis), axis=1)
    b = (rho / step).astype(int)
    cnt = np.bincount(b)
    s = np.bincount(b, weights=h)
    return [float(s[i] / cnt[i]) for i in range(len(cnt)) if cnt[i] >= 10]
