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
