"""Voxel pipeline on pseudo-atom assemblies with known volumes, cavities and hydrophobicity patterns."""
import numpy as np
import pytest

from protshape.geometry import gyration_tensor, sphericity
from protshape.voxel import BoxTooLarge, voxel_metrics


def lattice_in(fn, extent, step=1.8):
    g = np.arange(-extent, extent + 1e-9, step)
    p = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
    return p[fn(p)]


def run(coords, h=None, radius=1.7, **kw):
    h = np.zeros(len(coords)) if h is None else h
    _, vecs, _ = gyration_tensor(coords)
    return voxel_metrics(coords, np.full(len(coords), radius), {"kd": h, "ww": h, "atom": h},
                         np.arange(len(coords)), vecs, **kw)


def test_single_large_sphere_volume_area_depth():
    R = 8.0
    m, per_atom = run(np.zeros((1, 3)), radius=R)
    v = 4 / 3 * np.pi * R ** 3
    assert m["v_vdw"] == pytest.approx(v, rel=0.03)
    assert m["v_ses"] == pytest.approx(v, rel=0.06)            # convex body: SES = vdW surface
    assert m["a_ses"] == pytest.approx(4 * np.pi * R ** 2, rel=0.06)
    assert sphericity(m["v_ses"], m["a_ses"]) == pytest.approx(1.0, abs=0.06)
    assert m["cavity_n"] == 0
    assert per_atom["depth"][0] == pytest.approx(R, abs=1.0)


def test_hollow_shell_has_one_enclosed_cavity():
    R = 12.0
    shell = lattice_in(lambda p: np.abs(np.linalg.norm(p, axis=1) - R) < 1.6, R + 2, step=1.2)
    m, _ = run(shell)
    assert m["cavity_n"] == 1
    inner = R - 1.6 - 1.7                                        # inner wall surface radius (approx.)
    assert m["cavity_volume"] == pytest.approx(4 / 3 * np.pi * inner ** 3, rel=0.35)
    solid = lattice_in(lambda p: np.linalg.norm(p, axis=1) < R + 1.6, R + 2, step=1.2)
    m2, _ = run(solid)
    assert m2["cavity_n"] == 0


def test_hydrophobic_core_vs_inverted():
    p = lattice_in(lambda p: np.linalg.norm(p, axis=1) < 14, 14)
    r = np.linalg.norm(p, axis=1)
    core = np.where(r < 8, 1.0, -1.0)
    m, _ = run(p, h=core)
    assert m["inversion_index_kd"] < -1.0 and m["h_depth_slope_kd"] > 0
    assert m["hydrophobic_rg_ratio_kd"] < 0.8                   # hydrophobic mass concentrated inside
    m2, _ = run(p, h=-core)
    assert m2["inversion_index_kd"] > 1.0 and m2["h_depth_slope_kd"] < 0
    assert m2["hydrophobic_rg_ratio_kd"] > 1.0


def test_belt_on_cylinder_with_hydrophobic_band():
    p = lattice_in(lambda p: (np.hypot(p[:, 0], p[:, 1]) < 10) & (np.abs(p[:, 2]) < 30), 30)
    h = np.where(np.abs(p[:, 2]) < 15, 1.0, -1.0)
    m, _ = run(p, h=h)
    assert 24 <= m["belt_width_A_kd"] <= 36                     # principal axis = cylinder axis
    assert m["belt_contrast_kd"] > 1.0
    assert 24 <= m["belt_width_scan_A_kd"] <= 36 and m["belt_contrast_scan_kd"] > 1.0
    assert m["belt_normal_angle_pa1_kd"] < 10                  # scanned normal found the cylinder axis
    # uniform cylinder: no band
    m0, _ = run(p, h=np.zeros(len(p)))
    assert np.nan_to_num(m0["belt_contrast_kd"]) == pytest.approx(0, abs=1e-9)


def test_box_cap():
    p = np.array([[0, 0, 0], [400, 0, 0]], float)
    with pytest.raises(BoxTooLarge):
        run(p, max_box=256)


@pytest.mark.parametrize("R", [8.0, 12.0, 16.0])
def test_ses_volume_and_area_calibration(R):
    """Molecular volume and area of a single sphere within 4% across protein-like curvatures."""
    rng = np.random.default_rng(int(R))
    for _ in range(3):
        m, _ = run(rng.random((1, 3)), radius=R)
        assert m["v_ses"] == pytest.approx(4 / 3 * np.pi * R ** 3, rel=0.04)
        assert m["a_ses"] == pytest.approx(4 * np.pi * R ** 2, rel=0.04)


def test_tiny_region_has_no_surface_instead_of_error():
    from protshape.voxel import _mesh_area
    m = np.zeros((10, 10, 10), bool)
    m[5, 5, 5] = True
    m[5, 5, 6] = True
    assert np.isnan(_mesh_area(m, 1.0))


def test_belt_scan_finds_band_off_the_principal_axis():
    """Flat disc whose membrane normal is its SHORTEST axis: the principal-axis profile cannot see the
    band, the direction scan must."""
    p = lattice_in(lambda p: (np.hypot(p[:, 0], p[:, 1]) < 30) & (np.abs(p[:, 2]) < 24), 30, step=2.0)
    h = np.where(np.abs(p[:, 2]) < 14, 1.0, -1.0)
    m, _ = run(p, h=h)
    assert m["belt_normal_angle_pa1_kd"] > 80                  # normal is perpendicular to axis 1
    assert 20 <= m["belt_width_scan_A_kd"] <= 36 and m["belt_contrast_scan_kd"] > 1.0
