import numpy as np

from ve_simple.constants import AMINO_ACIDS, BLOSUM62
from ve_simple.predictors.physchem import physchem_scores, saturation_table


def test_saturation_table_size():
    seq = "MKTAYIAKQR"
    df = saturation_table(seq)
    assert len(df) == len(seq) * 19
    assert df["position"].min() == 1 and df["position"].max() == len(seq)
    assert (df["wt"] != df["mut"]).all()


def test_saturation_skips_nonstandard():
    df = saturation_table("MXK")
    assert set(df["position"]) == {1, 3}


def test_physchem_values():
    df = physchem_scores(saturation_table("K"))
    row = df[df["mut"] == "E"].iloc[0]
    assert row["blosum62"] == BLOSUM62[("K", "E")] == 1
    assert row["delta_charge"] == -2.0
    assert row["abs_delta_charge"] == 2.0
    assert np.isclose(row["delta_hydrophobicity"], -3.5 - (-3.9))
    assert np.isclose(row["delta_volume"], 138.4 - 168.6)


def test_blosum_symmetric():
    for a in AMINO_ACIDS:
        for b in AMINO_ACIDS:
            assert BLOSUM62[(a, b)] == BLOSUM62[(b, a)]
