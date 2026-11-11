"""Development only: populate a ve_simple-style cache with a random candidate pool for Stage 0.

On the cluster the full ve_simple cache already exists and this script is not needed.
Usage: python scripts/fetch_dev_sample.py --cache ../cache --per-group 120
"""
import argparse
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))       # ve_simple package (repo root)
from ve_simple import alphafold, uniprot                            # noqa: E402
from ve_simple.utils import http_get                                # noqa: E402

from protshape.sanity import SANITY_IDS                             # noqa: E402

STREAM = "https://rest.uniprot.org/uniprotkb/stream"


def accessions(query: str) -> list[str]:
    r = http_get(STREAM, params={"query": query, "format": "list"}, timeout=300)
    r.raise_for_status()
    return r.text.split()


def fetch(acc: str, cache: Path) -> str:
    try:
        uniprot.fetch_entry(acc, cache)
        meta = alphafold.fetch_afdb_metadata(acc, cache)
        if meta:
            alphafold.fetch_structure(acc, cache, meta)
        return "ok" if meta else "no model"
    except Exception as e:  # noqa: BLE001
        return f"error {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=Path("../cache"))
    ap.add_argument("--per-group", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    picks = list(SANITY_IDS)
    for taxid in (9606, 559292):
        tm = accessions(f"organism_id:{taxid} AND reviewed:true AND ft_transmem:*")
        sol = accessions(f"organism_id:{taxid} AND reviewed:true AND NOT ft_transmem:* AND NOT ft_intramem:* "
                         f"AND NOT ft_lipid:*")
        print(taxid, "candidates with TM:", len(tm), "without:", len(sol))
        picks += rng.sample(tm, a.per_group) + rng.sample(sol, a.per_group)
    with ThreadPoolExecutor(8) as ex:
        res = list(ex.map(lambda x: (x, fetch(x, a.cache)), picks))
    print({s: sum(1 for _, r in res if r == s) for s in {r for _, r in res}})
    for acc, r in res:
        if r.startswith("error"):
            print(acc, r)


if __name__ == "__main__":
    main()
