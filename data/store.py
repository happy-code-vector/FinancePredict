"""DuckDB store for miner data."""

import logging
from pathlib import Path

import duckdb

import miner.config as mcfg

logger = logging.getLogger(__name__)

SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"


class Store:
    """Thin wrapper around a DuckDB connection for miner data."""

    def __init__(self, db_path: Path | str | None = None) -> None:
        self.db_path = str(db_path or mcfg.DB_PATH)
        self.con = duckdb.connect(self.db_path)
        self._init_schema()

    def _init_schema(self) -> None:
        if SCHEMA_PATH.exists():
            sql = SCHEMA_PATH.read_text()
            # Execute each statement separately (DuckDB doesn't like batch with PRIMARY KEY)
            for stmt in sql.split(";"):
                stmt = stmt.strip()
                if stmt:
                    self.con.execute(stmt)
            logger.info("DuckDB schema initialized at %s", self.db_path)

    def execute(self, sql: str, params: tuple | None = None):
        if params:
            return self.con.execute(sql, params)
        return self.con.execute(sql)

    def query_df(self, sql: str, params: tuple | None = None):
        """Execute query and return a pandas DataFrame."""
        if params:
            return self.con.execute(sql, params).df()
        return self.con.execute(sql).df()

    def insert_prices(self, rows: list[tuple]) -> None:
        """Bulk insert price rows. rows = [(asset, ts, o, h, l, c, v), ...]"""
        self.con.executemany(
            "INSERT OR REPLACE INTO prices VALUES (?, ?, ?, ?, ?, ?, ?)", rows
        )

    def insert_funding(self, rows: list[tuple]) -> None:
        self.con.executemany(
            "INSERT OR REPLACE INTO funding_rates VALUES (?, ?, ?, ?, ?)", rows
        )

    def insert_sentiment(self, rows: list[tuple]) -> None:
        self.con.executemany(
            "INSERT OR REPLACE INTO sentiment VALUES (?, ?, ?, ?, ?, ?)", rows
        )

    def insert_fear_greed(self, rows: list[tuple]) -> None:
        self.con.executemany(
            "INSERT OR REPLACE INTO fear_greed VALUES (?, ?, ?)", rows
        )

    def insert_onchain(self, rows: list[tuple]) -> None:
        self.con.executemany(
            "INSERT OR REPLACE INTO onchain_flows VALUES (?, ?, ?, ?, ?, ?)", rows
        )

    def get_prices(self, asset: str, limit: int = 100000) -> list[tuple]:
        return self.con.execute(
            "SELECT ts, close FROM prices WHERE asset = ? ORDER BY ts DESC LIMIT ?",
            [asset, limit],
        ).fetchall()

    def get_latest_price(self, asset: str) -> float | None:
        row = self.con.execute(
            "SELECT close FROM prices WHERE asset = ? ORDER BY ts DESC LIMIT 1",
            [asset],
        ).fetchone()
        return row[0] if row else None

    def export_parquet(self, table: str, path: Path | None = None) -> None:
        """Export a table to Parquet."""
        out = str(path or mcfg.PARQUET_DIR / f"{table}.parquet")
        self.con.execute(f"COPY {table} TO '{out}' (FORMAT PARQUET)")
        logger.info("Exported %s to %s", table, out)

    def import_parquet(self, table: str, path: Path) -> None:
        """Import a Parquet file into a table."""
        self.con.execute(f"INSERT INTO {table} SELECT * FROM read_parquet('{path}')")

    def close(self) -> None:
        self.con.close()
