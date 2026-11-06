import json

import numpy as np
import pandas as pd

from ve_simple.cachecheck import check_accession_dir, check_cache


def _mk(tmp_path, seq="MKTAYIAKQR"):
    d = tmp_path / "P00000"
    d.mkdir()
    (d / "uniprot.json").write_text(json.dumps({"sequence": {"value": seq}}))
    rows = [(i, wt, m, 0.1) for i, wt in enumerate(seq, 1) for m in "ACDEFGHIKLMNPQRSTVWY" if m != wt]
    pd.DataFrame(rows, columns=["position", "wt", "mut", "esm1v"]).to_csv(d / "scores_esm1v.tsv", sep="\t", index=False)
    np.save(d / "esm1v_1.npy", np.zeros((len(seq), 20), dtype=np.float32))
    return d


def test_clean_dir_has_no_problems(tmp_path):
    d = _mk(tmp_path)
    assert check_accession_dir(d) == []


def test_truncated_and_garbled_files_are_found_and_deleted(tmp_path):
    d = _mk(tmp_path)
    text = (d / "scores_esm1v.tsv").read_text()
    (d / "scores_esm1v.tsv").write_text(text[: len(text) // 2])          # truncated
    (d / "esm1v_1.npy").write_bytes(b"not a numpy file")                 # garbled
    (d / "afdb.json").write_text("{broken")                              # bad json
    (d / "scores_saprot.tsv.123.tmp").write_text("x")                    # leftover temp
    probs = check_cache(tmp_path, delete=True)
    names = sorted(p.name for p, _ in probs)
    assert names == ["afdb.json", "esm1v_1.npy", "scores_esm1v.tsv", "scores_saprot.tsv.123.tmp"]
    assert not (d / "scores_esm1v.tsv").exists() and (d / "uniprot.json").exists()
