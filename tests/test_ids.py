from ve_simple.utils import read_id_list


def test_read_id_list_formats(tmp_path):
    p = tmp_path / "ids.tsv"
    p.write_text("Entry\tEntry Name\tOrganism\nP04637\tP53_HUMAN\tHomo sapiens\nQ9H6T3-2\tX_HUMAN\tHomo sapiens\n"
                 "A0A0B4J1V6\tY\tz\n>sp|P68871|HBB_HUMAN\nP69905, P02185 # comment\nP04637\n")
    assert read_id_list(p) == ["P04637", "Q9H6T3", "A0A0B4J1V6", "P68871", "P69905"]
