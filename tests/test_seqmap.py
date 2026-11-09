import pandas as pd

from ve_simple.seqmap import matches_sequence, remap_to_sequence, table_sequence


def _table(seq, value_offset=0.0):
    rows = [(i, a, "A" if a != "A" else "G", float(i) + value_offset) for i, a in enumerate(seq, 1)]
    return pd.DataFrame(rows, columns=["position", "wt", "mut", "score"])


def test_identical_sequence_untouched():
    df = _table("MKTAYIAKQR")
    out, frac = remap_to_sequence(df, "MKTAYIAKQR")
    assert frac is None and out is df and matches_sequence(df, "MKTAYIAKQR")


def test_insertion_in_old_sequence_is_aligned():
    old = "MKTAYIAKQRQISFVKSHFSRQ" + "GGG" + "LEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQ"
    new = "MKTAYIAKQRQISFVKSHFSRQ" + "LEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQ"
    out, frac = remap_to_sequence(_table(old), new, "test")
    assert matches_sequence(out, new)
    assert len(out) == len(new) and frac == len(new) / len(old)
    # residue 26 of the old sequence ('L' after the GGG insertion) is residue 23 now, with its value
    assert out.set_index("position").loc[23, "score"] == 26.0


def test_n_terminal_extension_dropped_and_rest_shifted():
    core = "MSTEQLRKRAEELLQKAQELARKGDHEGAIRLYEEALRLA"
    old = "MAPGSAAALLLRRSPEPPGG" + core          # longer old isoform
    out, frac = remap_to_sequence(_table(old), core, "test")
    assert matches_sequence(out, core) and len(out) == len(core)
    assert out["position"].min() == 1 and out.set_index("position").loc[1, "score"] == 21.0


def test_unrelated_sequence_keeps_only_identical_aligned_residues():
    out, _ = remap_to_sequence(_table("WWWWWWWWWW"), "MKTAYIAKQR", "test")
    assert matches_sequence(out, "MKTAYIAKQR")


def test_table_sequence_fills_gaps():
    df = pd.DataFrame({"position": [1, 3], "wt": ["M", "T"], "mut": ["A", "A"]})
    assert table_sequence(df) == "MXT"


def test_alphamissense_bulk_fill(tmp_path):
    import gzip

    from ve_simple.predictors.alphamissense import fill_from_bulk, per_protein_path, read_alphamissense

    rows = ["# Copyright", "#", "# Licence", "uniprot_id\tprotein_variant\tam_pathogenicity\tam_class",
            "P11111\tM1A\t0.1\tlikely_benign", "P11111\tK2E\t0.9\tlikely_pathogenic",
            "P22222\tG5D\t0.5\tambiguous", "P33333\tA1V\t0.2\tlikely_benign"]
    data = tmp_path / "data"
    data.mkdir()
    with gzip.open(data / "AlphaMissense_aa_substitutions.tsv.gz", "wt") as fh:
        fh.write("\n".join(rows) + "\n")
    cache = tmp_path / "cache"
    assert fill_from_bulk(["P11111", "P33333", "P99999"], cache, data, download=False) == 2
    am = read_alphamissense(per_protein_path(cache, "P11111"))
    assert list(am["position"]) == [1, 2] and list(am["alphamissense_class"]) == ["LBen", "LPath"]
    assert not per_protein_path(cache, "P22222").exists()
