"""Synthetic point clouds with known answers (stage 4 requirement: must pass before real structures)."""
import numpy as np
import pytest

from protshape import geometry as G

RNG = np.random.default_rng(0)


def ball(n, r=1.0):
    x = RNG.normal(size=(n, 3))
    x /= np.linalg.norm(x, axis=1)[:, None]
    return x * r * RNG.random(n)[:, None] ** (1 / 3)


def ellipsoid(n, a, b, c):
    return ball(n) * np.array([a, b, c])


def analytic(a, b, c):
    lam = np.array(sorted([a * a, b * b, c * c], reverse=True)) / 5
    tr = lam.sum()
    k2 = 1 - 3 * (lam[0] * lam[1] + lam[1] * lam[2] + lam[2] * lam[0]) / tr ** 2
    lb = tr / 3
    return k2, np.prod(lam - lb) / lb ** 3


def test_uniform_sphere():
    x = ball(60000, 20.0)
    d = G.gyration_descriptors(x)
    assert G.radius_of_gyration(x) == pytest.approx(np.sqrt(3 / 5) * 20, rel=0.01)
    assert d["kappa2"] < 0.002 and abs(d["prolateness"]) < 0.002
    assert d["a_over_c"] < 1.03
    assert d["ax_a"] == pytest.approx(20, rel=0.02)            # equivalent-ellipsoid semi-axis = R
    h = G.hull_metrics(x)
    assert h["sphericity_hull"] > 0.98
    assert h["dmax"] == pytest.approx(40, rel=0.01)


@pytest.mark.parametrize("axes,expect_sign", [((30, 10, 10), 1), ((30, 30, 10), -1)])
def test_ellipsoids(axes, expect_sign):
    x = ellipsoid(80000, *axes)
    d = G.gyration_descriptors(x)
    k2, s = analytic(*axes)
    assert d["a_over_c"] == pytest.approx(3.0, rel=0.02)
    assert d["kappa2"] == pytest.approx(k2, abs=0.01)
    assert d["prolateness"] == pytest.approx(s, abs=0.02)
    assert np.sign(d["prolateness"]) == expect_sign
    assert d["ax_a"] == pytest.approx(30, rel=0.02) and d["ax_c"] == pytest.approx(10, rel=0.02)
    if expect_sign > 0:   # prolate: principal axis along x
        assert abs(d["pa_x"]) > 0.99


def test_thin_rod():
    x = np.column_stack([np.linspace(-50, 50, 2000), RNG.normal(0, 0.05, 2000), RNG.normal(0, 0.05, 2000)])
    d = G.gyration_descriptors(x)
    assert d["kappa2"] > 0.99 and d["prolateness"] > 1.9 and d["a_over_c"] > 100
    assert d["asphericity"] == pytest.approx(1.0, abs=0.01)


def test_dumbbell():
    s1 = ball(20000, 10) + [-25, 0, 0]
    s2 = ball(20000, 10) + [25, 0, 0]
    x = np.vstack([s1, s2])
    d = G.gyration_descriptors(x)
    h = G.hull_metrics(x)
    assert d["prolateness"] > 0.5 and d["kappa2"] > 0.5
    # solidity: union of the two balls / their hull (capsule-like) is well below 1
    v_balls = 2 * 4 / 3 * np.pi * 10 ** 3
    assert v_balls / h["v_hull"] < 0.6
    assert h["sphericity_hull"] < 0.9


def test_reference_radii_and_contacts():
    assert G.rmin_empirical(100) == pytest.approx(2.2 * 100 ** 0.38)
    v = 4 / 3 * np.pi * 15 ** 3
    assert G.rmin_sphere(v) == pytest.approx(np.sqrt(3 / 5) * 15)
    assert G.sphericity(v, 4 * np.pi * 15 ** 2) == pytest.approx(1.0)
    line = np.array([[0, 0, 0], [5, 0, 0], [10, 0, 0]], float)
    assert G.contact_density(line, 8.0) == pytest.approx(4 / 3)


def test_hull_of_expanded_atoms_contains_vdw_shell():
    x = np.array([[0, 0, 0], [10, 0, 0], [0, 10, 0], [0, 0, 10]], float)
    h0 = G.hull_metrics(x)
    h1 = G.hull_metrics(x, radii=np.full(4, 1.7))
    assert h1["v_hull"] > h0["v_hull"]
