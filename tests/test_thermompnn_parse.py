import pandas as pd

from ve_simple.predictors.thermompnn import _ACC_RE, detect_position_offset, parse_prediction_file


def test_accession_regex():
    name = "ThermoMPNN_inference_AF-A0A0B4J1V6-F1-model_v4.csv"
    assert _ACC_RE.findall(name) == ["A0A0B4J1V6"]
    assert _ACC_RE.findall("ThermoMPNN_inference_AF-P04637-F1-model_v4.csv") == ["P04637"]


def test_parse_inference_csv(tmp_path):
    p = tmp_path / "ThermoMPNN_inference_AF-P00000-F1-model_v4.csv"
    pd.DataFrame({
        "Model": "ThermoMPNN", "Dataset": "x",
        "ddG_pred": [0.0, 1.5, -0.2, 2.0], "position": [0, 0, 1, 1],
        "wildtype": ["M", "M", "K", "K"], "mutation": ["M", "A", "K", "E"], "pdb": "x", "chain": "A",
    }).to_csv(p)
    df = parse_prediction_file(p)
    assert list(df.columns) == ["position", "wt", "mut", "thermompnn"]
    assert len(df) == 2  # wt->wt rows dropped
    assert detect_position_offset(df, "MKT") == 1
    assert detect_position_offset(df.assign(position=df["position"] + 1), "MKT") == 0


def _member_csv(rows):
    return pd.DataFrame(rows, columns=["position", "wildtype", "mutation", "ddG_pred"]).to_csv().encode()


def test_stream_fragments_into_sink(tmp_path):
    import io
    import json
    import tarfile

    from ve_simple.predictors.thermompnn import ZenodoThermoMPNN

    members = {
        # P11111: two fragments; F2 starts at residue 201, so its position 0 is residue 201
        "thermompnn_outs/ThermoMPNN_inference_AF-P11111-F1-model_v4.csv": _member_csv(
            [(0, "M", "A", 1.0), (200, "K", "E", 2.0)]),
        "thermompnn_outs/ThermoMPNN_inference_AF-P11111-F2-model_v4.csv": _member_csv(
            [(0, "K", "E", 4.0), (1, "L", "P", 3.0)]),
        "thermompnn_outs/ThermoMPNN_inference_AF-P22222-F1-model_v4.csv": _member_csv([(4, "G", "D", 0.5)]),
        "thermompnn_outs/ThermoMPNN_inference_AF-P22222-2-F1-model_v4.csv": _member_csv([(4, "G", "D", 9.9)]),
    }
    arc = tmp_path / "thermompnn_outs.tar.gz"
    with tarfile.open(arc, "w:gz") as tf:
        for name, data in members.items():
            ti = tarfile.TarInfo(name)
            ti.size = len(data)
            tf.addfile(ti, io.BytesIO(data))
    zen = ZenodoThermoMPNN(tmp_path)
    idx = zen.build_index()
    assert len(zen.members_for("P11111")) == 2 and len(zen.members_for("P22222")) == 1   # isoform excluded
    got = {}
    assert zen.stream(["P11111", "P22222", "P99999"], lambda a, df: got.__setitem__(a, df)) == 2
    p1 = got["P11111"].set_index(["position", "mut"])["thermompnn"]
    assert p1[(1, "A")] == 1.0
    assert p1[(201, "E")] == 3.0            # mean of F1 (2.0) and F2 (4.0) at the overlapping residue
    assert p1[(202, "P")] == 3.0
    assert list(got["P22222"]["position"]) == [5] and list(got["P22222"]["thermompnn"]) == [0.5]
    assert not (tmp_path / "thermompnn_outs").exists()   # nothing unpacked to disk
    assert json.loads(zen.index_path.read_text()) == idx
