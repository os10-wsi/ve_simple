#!/usr/bin/env python
"""Representative proteins for large membrane Pfam families (input for ve_simple, protshape and mvsa).

For each family in the families table:
  1. download its UniProtKB members as FASTA: Pfam match, not a fragment, with an AlphaFold DB model,
     length MIN_LEN-MAX_LEN (the download is checked against UniProt's count);
  2. drop sequences with non-standard residues (B, J, O, U, X, Z);
  3. cluster with MMseqs2 easy-cluster (--min-seq-id ID -c COV --cov-mode 0). If there are fewer clusters
     than N, cluster again at the next identity of the ladder (default 0.3, 0.5, 0.7, 0.9);
  4. keep the representatives of the N largest clusters (--pick largest: covers the family in proportion to
     how much sequence space each subgroup occupies) or N clusters at random (--pick random: even coverage);
  5. fetch reviewed status, organism and lineage of the representatives.

Outputs (in --out, default sets/families):
  ids/<PFAM>_<short>.txt    representatives of each family (one accession per line)
  all_family_ids.txt        all representatives, deduplicated: the ID list for the ve_simple run
  family_membership.tsv     accession, pfam, short, name, clan, family_members, cluster_size, cluster_rank,
                            min_seq_id, reviewed, organism, taxid, domain, kingdom, length
  work/<PFAM>/              downloads and MMseqs2 files (large; delete once the ID lists look right)

Usage (needs network for UniProt and an `mmseqs` binary; see docs/new_datasets.md):
  python scripts/family_sets.py --families sets/families/families.tsv --n 500 --threads 16
Families already done are skipped (delete ids/<PFAM>_*.txt or pass --force to redo one).
"""
from __future__ import annotations

import argparse
import csv
import gzip
import io
import logging
import random
import re
import shutil
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

log = logging.getLogger("family_sets")
UNIPROT = "https://rest.uniprot.org/uniprotkb"
STANDARD = set("ACDEFGHIKLMNPQRSTVWY")
ACC = re.compile(r"^>(?:sp|tr)\|([A-Z0-9]+)\|")


def query_for(pfam: str, min_len: int, max_len: int) -> str:
    return (f"xref:pfam-{pfam} AND fragment:false AND database:alphafolddb "
            f"AND length:[{min_len} TO {max_len}]")


def _get(url: str, tries: int = 5, timeout: int = 600):
    for k in range(tries):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "family-sets/1.0"}),
                                          timeout=timeout)
        except Exception as e:  # noqa: BLE001
            if k == tries - 1:
                raise
            wait = 10 * 2 ** k
            log.warning("%s: %s; retrying in %d s", url[:90], e, wait)
            time.sleep(wait)


def count(query: str) -> int:
    r = _get(f"{UNIPROT}/search?size=1&query={urllib.parse.quote(query)}")
    return int(r.headers["X-Total-Results"])


def download_fasta(query: str, dest: Path, expected: int) -> int:
    """Stream the members as gzipped FASTA; retry the whole download if it comes back short."""
    url = f"{UNIPROT}/stream?format=fasta&compressed=true&query={urllib.parse.quote(query)}"
    for attempt in range(3):
        tmp = dest.with_suffix(".part")
        with _get(url, timeout=3600) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f, length=1 << 20)
        n = sum(1 for line in gzip.open(tmp, "rt") if line.startswith(">"))
        if n >= expected:
            tmp.replace(dest)
            return n
        log.warning("%s: got %d of %d sequences (attempt %d)", dest.name, n, expected, attempt + 1)
    raise SystemExit(f"download of {dest} incomplete after 3 attempts")


def clean_fasta(src: Path, dest: Path) -> int:
    """Accession-only headers; drop sequences with non-standard residues."""
    kept = 0
    with gzip.open(src, "rt") as fi, open(dest, "w") as fo:
        acc, seq = None, []

        def flush():
            nonlocal kept
            if acc and seq:
                s = "".join(seq)
                if set(s) <= STANDARD:
                    fo.write(f">{acc}\n{s}\n")
                    kept += 1
        for line in fi:
            if line.startswith(">"):
                flush()
                m = ACC.match(line)
                acc, seq = (m.group(1) if m else line[1:].split()[0]), []
            else:
                seq.append(line.strip())
        flush()
    return kept


def cluster(fasta: Path, work: Path, min_id: float, cov: float, threads: int, mmseqs: str) -> dict[str, int]:
    """MMseqs2 easy-cluster; returns {representative: cluster size}."""
    prefix = work / f"clu{int(min_id * 100)}"
    tsv = Path(f"{prefix}_cluster.tsv")
    if not tsv.exists():
        tmp = work / "tmp"
        cmd = [mmseqs, "easy-cluster", str(fasta), str(prefix), str(tmp), "--min-seq-id", str(min_id),
               "-c", str(cov), "--cov-mode", "0", "--threads", str(threads)]
        log.info("%s", " ".join(cmd))
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
        shutil.rmtree(tmp, ignore_errors=True)
    sizes: Counter = Counter()
    with open(tsv) as f:
        for line in f:
            sizes[line.split("\t", 1)[0]] += 1
    return dict(sizes)


def metadata(accs: list[str]) -> dict[str, dict]:
    out = {}
    fields = "accession,reviewed,organism_name,organism_id,lineage,length"
    for i in range(0, len(accs), 100):
        q = " OR ".join(f"accession:{a}" for a in accs[i:i + 100])
        r = _get(f"{UNIPROT}/search?format=tsv&size=500&fields={fields}&query={urllib.parse.quote(q)}")
        for row in csv.DictReader(io.TextIOWrapper(r, encoding="utf-8"), delimiter="\t"):
            lin = row.get("Taxonomic lineage", "")
            rank = lambda r_: next((t.rsplit(" (", 1)[0] for t in lin.split(", ") if t.endswith(f"({r_})")), "")
            out[row["Entry"]] = {"reviewed": row["Reviewed"] == "reviewed", "organism": row["Organism"],
                                 "taxid": row["Organism (ID)"], "domain": rank("domain") or rank("superkingdom"),
                                 "kingdom": rank("kingdom"), "length": row["Length"]}
    return out


def select(fam: dict, args, out: Path) -> list[dict]:
    pf, short = fam["pfam"], fam["short"]
    ids_file = out / "ids" / f"{pf}_{short}.txt"
    rows_file = out / "work" / pf / "selected.tsv"
    if ids_file.exists() and rows_file.exists() and not (args.force and pf in args.force):
        log.info("%s %s: done already", pf, short)
        return list(csv.DictReader(open(rows_file), delimiter="\t"))
    work = out / "work" / pf
    work.mkdir(parents=True, exist_ok=True)
    q = query_for(pf, args.min_len, args.max_len)
    n_total = count(q)
    log.info("%s %s: %d UniProtKB members with an AlphaFold model, %d-%d aa", pf, short, n_total, args.min_len, args.max_len)
    raw, fasta = work / "members.fasta.gz", work / "members.fasta"
    if not raw.exists():
        download_fasta(q, raw, n_total)
    if not fasta.exists():
        log.info("%s: %d sequences after removing non-standard residues", pf, clean_fasta(raw, fasta))
    sizes, used = {}, None
    for min_id in args.ladder:
        sizes = cluster(fasta, work, min_id, args.coverage, args.threads, args.mmseqs)
        used = min_id
        log.info("%s: %d clusters at %.0f%% identity", pf, len(sizes), 100 * min_id)
        if len(sizes) >= args.n:
            break
    ranked = sorted(sizes.items(), key=lambda kv: (-kv[1], kv[0]))
    if args.pick == "random" and len(ranked) > args.n:
        pick = sorted(random.Random(args.seed).sample(ranked, args.n), key=lambda kv: (-kv[1], kv[0]))
    else:
        pick = ranked[:args.n]
    meta = metadata([a for a, _ in pick])
    rows = []
    for rank_, (acc, size) in enumerate(pick, 1):
        m = meta.get(acc, {})
        rows.append({"accession": acc, "pfam": pf, "short": short, "name": fam.get("name", ""), "clan": fam.get("clan", ""),
                     "family_members": n_total, "cluster_size": size, "cluster_rank": rank_, "min_seq_id": used,
                     **{k: m.get(k, "") for k in ("reviewed", "organism", "taxid", "domain", "kingdom", "length")}})
    with open(rows_file, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    ids_file.parent.mkdir(parents=True, exist_ok=True)
    ids_file.write_text("\n".join(r["accession"] for r in rows) + "\n")
    dom = Counter(r["domain"] for r in rows)
    log.info("%s %s: %d representatives (%s); covering %d of %d clustered sequences", pf, short, len(rows),
             ", ".join(f"{k or '?'} {v}" for k, v in dom.most_common()), sum(s for _, s in pick), sum(sizes.values()))
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--families", type=Path, default=Path("sets/families/families.tsv"))
    ap.add_argument("--out", type=Path, default=Path("sets/families"))
    ap.add_argument("--n", type=int, default=500, help="representatives per family")
    ap.add_argument("--ladder", default="0.3,0.5,0.7,0.9", help="identity thresholds tried in turn")
    ap.add_argument("--coverage", type=float, default=0.8)
    ap.add_argument("--min-len", type=int, default=100)
    ap.add_argument("--max-len", type=int, default=1500)
    ap.add_argument("--pick", choices=["largest", "random"], default="largest")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--mmseqs", default="mmseqs", help="path to the MMseqs2 binary")
    ap.add_argument("--only", nargs="*", help="process only these Pfam accessions")
    ap.add_argument("--force", nargs="*", default=[], help="redo these Pfam accessions")
    args = ap.parse_args(argv)
    args.ladder = [float(x) for x in args.ladder.split(",")]
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    if shutil.which(args.mmseqs) is None and not Path(args.mmseqs).exists():
        raise SystemExit(f"MMseqs2 not found ({args.mmseqs}); install it or pass --mmseqs /path/to/mmseqs")
    fams = [r for r in csv.DictReader(open(args.families), delimiter="\t") if not r["pfam"].startswith("#")]
    if args.only:
        fams = [f for f in fams if f["pfam"] in args.only]
    rows = []
    for fam in fams:
        rows += select(fam, args, args.out)
    # union over ALL families done so far (not only this invocation)
    all_rows = []
    for f in sorted((args.out / "work").glob("*/selected.tsv")):
        all_rows += list(csv.DictReader(open(f), delimiter="\t"))
    with open(args.out / "family_membership.tsv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0]), delimiter="\t")
        w.writeheader()
        w.writerows(all_rows)
    ids = list(dict.fromkeys(r["accession"] for r in all_rows))
    (args.out / "all_family_ids.txt").write_text("\n".join(ids) + "\n")
    dup = len(all_rows) - len(ids)
    log.info("%d families, %d representatives, %d unique accessions%s -> %s", len({r['pfam'] for r in all_rows}),
             len(all_rows), len(ids), f" ({dup} selected by more than one family)" if dup else "", args.out / "all_family_ids.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
