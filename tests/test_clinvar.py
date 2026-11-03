from ve_simple.clinvar import class_to_label, parse_clinvar_missense, summarise


def _feature(pos, wt, mut, sigs, sources=("ClinVar",), consequence="missense", ids=("RCV1",)):
    return {
        "type": "VARIANT", "begin": str(pos), "end": str(pos), "wildType": wt, "mutatedType": mut,
        "consequenceType": consequence,
        "clinicalSignificances": [{"type": s, "sources": list(sources), "reviewStatus": "criteria provided"} for s in sigs],
        "xrefs": [{"name": "ClinVar", "id": i} for i in ids],
    }


def test_parse_basic_classes():
    data = {"features": [
        _feature(10, "R", "H", ["Pathogenic"]),
        _feature(11, "A", "T", ["Likely benign"]),
        _feature(12, "G", "D", ["Variant of uncertain significance"]),
        _feature(13, "L", "P", ["Pathogenic"], sources=("UniProt",)),   # not ClinVar -> ignored
        _feature(14, "L", "P", ["Pathogenic"], consequence="stop gained"),  # not missense
        _feature(15, "K", "E", ["Pathogenic", "Benign"]),               # conflicting
        _feature(16, "K", "E", ["Likely pathogenic", "Pathogenic"]),    # resolves to pathogenic
    ]}
    df = parse_clinvar_missense("P00000", data)
    cls = dict(zip(df["position"], df["clinvar_class"]))
    assert cls == {10: "pathogenic", 11: "likely_benign", 12: "vus", 15: "conflicting", 16: "pathogenic"}
    lab = dict(zip(df["position"], df["clinvar_label"]))
    assert lab[11] == "benign" and lab[15] == "other" and lab[12] == "vus"
    s = summarise(df)
    assert s["n_pathogenic"] == 2 and s["n_benign"] == 1 and s["n_vus"] == 1 and s["n_conflicting"] == 1


def test_strict_labels():
    assert class_to_label("likely_pathogenic", include_likely=False) == "other"
    assert class_to_label("likely_pathogenic", include_likely=True) == "pathogenic"
    assert class_to_label("benign", include_likely=False) == "benign"


def test_empty():
    assert parse_clinvar_missense("X", None).empty
    assert parse_clinvar_missense("X", {"features": []}).empty
