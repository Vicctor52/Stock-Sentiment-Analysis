from pathlib import Path

import pandas as pd

from finsent.logging import get_logger
from finsent.storage.schemas import PRICES_COLUMNS

logger = get_logger(__name__)

# Mapping of table names to their primary key column(s) for deduplication.
TABLE_PRIMARY_KEYS = {
    "prices": ["ticker", "session_date"],
}


class ParquetStore:
    """Read and write named Parquet tables under a root directory.

    Attributes
    ----------
    root : Path
        Base directory containing the Parquet tables.
    """

    def __init__(self, root: Path) -> None:
        """Initialize the store.

        Parameters
        ----------
        root : Path
            Base directory for Parquet tables.
        """
        self.root = root

    def _table_path(self, table: str) -> Path:
        return self.root / f"{table}.parquet"

    def _primary_keys(self, table: str) -> list[str]:
        """Return the primary key column(s) for *table*.

        Parameters
        ----------
        table : str
            Name of the table (must be a key in :data:`TABLE_PRIMARY_KEYS`).

        Returns
        -------
        list[str]
            Column names that form the primary key.
        """
        keys = TABLE_PRIMARY_KEYS.get(table)
        if keys is None:
            raise KeyError(f"Unknown table '{table}'; no primary keys defined.")
        return keys

    def write(self, table: str, frame: pd.DataFrame) -> None:
        """Write *frame* to a named Parquet table, deduplicating by primary key.

        If the table file already exists, the new data is concatenated with the
        existing data and duplicates (according to the table's primary key) are
        removed, keeping the *last* occurrence.

        Parameters
        ----------
        table : str
            Name of the table (e.g. ``"prices"``).
        frame : pd.DataFrame
            Data to persist. Must contain the columns required by the table's
            schema (validated silently; missing columns are filled with ``NaN``).
        """
        # Ensure required columns exist; fill missing ones with NaN so that
        # deduplication on the key columns still works.
        required = PRICES_COLUMNS if table == "prices" else []
        for col in required:
            if col not in frame.columns:
                frame[col] = pd.NA

        path = self._table_path(table)

        # Load existing data if present.
        if path.exists():
            existing = pd.read_parquet(path)
            # Combine existing + new.
            combined = pd.concat([existing, frame], ignore_index=True)
        else:
            combined = frame

        keys = self._primary_keys(table)
        # Drop duplicates keeping the last row for each key combination.
        combined = combined.drop_duplicates(subset=keys, keep="last")

        combined.to_parquet(path, index=False)
        logger.info("Wrote %d rows to %s", len(combined), path)

    def read(self, table: str) -> pd.DataFrame:
        """Read an entire Parquet table.

        Parameters
        ----------
        table : str
            Name of the table (e.g. ``"prices"``).

        Returns
        -------
        pd.DataFrame
            The table contents.
        """
        path = self._table_path(table)
        if not path.exists():
            raise FileNotFoundError(f"Table '{table}' not found at {path}")
        return pd.read_parquet(path)
