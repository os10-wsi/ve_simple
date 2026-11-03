import numpy as np

from ve_simple.predictors.lm_common import logodds_to_frame, window_bounds


def test_window_bounds():
    assert window_bounds(500, 10) == (0, 500)
    s, e = window_bounds(3000, 10, 1022)
    assert (s, e) == (0, 1022)
    s, e = window_bounds(3000, 2990, 1022)
    assert (s, e) == (3000 - 1022, 3000)
    s, e = window_bounds(3000, 1500, 1022)
    assert e - s == 1022 and s <= 1500 < e


def test_logodds_to_frame():
    mat = np.arange(40, dtype=float).reshape(2, 20)
    df = logodds_to_frame(mat, "AC", "esm1v")
    assert len(df) == 38
    assert df[(df.position == 1) & (df.mut == "C")]["esm1v"].iloc[0] == 1.0
