"""SQLite persistence for proteins, per-variant scores, ClinVar labels and AUC statistics."""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable

import pandas as pd


class ResultsDB:
    """SQLite store. With fresh=True (a full assembly) the file is rebuilt from scratch and rows are only
    appended: deleting a protein's previous rows would scan the whole unindexed table for every protein,
    which grows quadratically with the proteome. Call finish() to add accession indexes for lookups."""

    def __init__(self, path: Path, fresh: bool = False):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fresh = fresh
        if fresh:
            for suffix in ("", "-wal", "-shm", "-journal"):
                Path(str(self.path) + suffix).unlink(missing_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.execute("PRAGMA journal_mode=WAL")
        if fresh:
            self.conn.execute("PRAGMA synchronous=OFF")     # rebuilt from scratch: a crash just means rerun

    def close(self):
        self.conn.close()

    def finish(self) -> None:
        """Index every table on accession (fast per-protein queries and replacements later)."""
        for (t,) in self.conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall():
            cols = {r[1] for r in self.conn.execute(f"PRAGMA table_info({t})")}
            if "accession" in cols:
                self.conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{t}_accession ON {t}(accession)")
        self.conn.commit()

    def _replace_rows(self, table: str, df: pd.DataFrame, key: str = "accession") -> None:
        """Replace all rows of `table` belonging to the accession(s) in `df`."""
        if df is None or len(df) == 0:
            return
        exists = self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone()
        if exists:
            # add any new columns the table lacks (schema evolves as predictors are added)
            cols = {r[1] for r in self.conn.execute(f"PRAGMA table_info({table})")}
            for c in df.columns:
                if c not in cols:
                    self.conn.execute(f'ALTER TABLE {table} ADD COLUMN "{c}"')
            if not self.fresh:
                accs = tuple(df[key].unique())
                self.conn.executemany(f"DELETE FROM {table} WHERE {key}=?", [(a,) for a in accs])
                self.conn.commit()
        df.to_sql(table, self.conn, if_exists="append", index=False)
        self.conn.commit()

    def write_protein(self, protein_row: dict) -> None:
        self._replace_rows("proteins", pd.DataFrame([protein_row]))

    def write_variants(self, df: pd.DataFrame) -> None:
        self._replace_rows("variants", df)

    def write_clinvar(self, df: pd.DataFrame) -> None:
        self._replace_rows("clinvar", df)

    def write_auc(self, df: pd.DataFrame) -> None:
        self._replace_rows("auc", df)

    def read(self, table: str, accessions: Iterable[str] | None = None) -> pd.DataFrame:
        exists = self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone()
        if not exists:
            return pd.DataFrame()
        if accessions is None:
            return pd.read_sql(f"SELECT * FROM {table}", self.conn)
        accs = list(accessions)
        q = f"SELECT * FROM {table} WHERE accession IN ({','.join('?' * len(accs))})"
        return pd.read_sql(q, self.conn, params=accs)

    def has_variants(self, accession: str) -> bool:
        exists = self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='variants'").fetchone()
        if not exists:
            return False
        r = self.conn.execute("SELECT 1 FROM variants WHERE accession=? LIMIT 1", (accession,)).fetchone()
        return r is not None
