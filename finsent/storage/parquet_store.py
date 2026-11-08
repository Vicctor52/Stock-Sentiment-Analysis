from pathlib import Path

import pandas as pd

from finsent.logging import get_logger
from finsent.storage.schemas import (
    DAILY_SENTIMENT_COLUMNS,
    NEWS_ARTICLES_COLUMNS,
    PRICES_COLUMNS,
    SENTIMENT_COLUMNS,
)

logger = get_logger(__name__)

# Mapping of table names to their expected column order.
TABLE_COLUMNS = {
    "prices": PRICES_COLUMNS,
    "news": NEWS_ARTICLES_COLUMNS,
    "news_sentiment": SENTIMENT_COLUMNS,
    "daily_sentiment": DAILY_SENTIMENT_COLUMNS,
}

# Mapping of table names to their primary key column(s) for deduplication.
TABLE_PRIMARY_KEYS = {
    "prices": ["ticker", "session_date"],
    "news": ["article_id", "ticker"],
    "news_sentiment": ["doc_id", "ticker"],
    "daily_sentiment": ["ticker", "session_date", "source"],
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

    def _columns(self, table: str) -> list[str]:
        """Return the expected column order for *table*.

        Parameters
        ----------
        table : str
            Name of the table (must be a key in :data:`TABLE_COLUMNS`).

        Returns
        -------
        list[str]
            Column names in schema order.
        """
        columns = TABLE_COLUMNS.get(table)
        if columns is None:
            raise KeyError(f"Unknown table '{table}'; no columns defined.")
        return columns

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

    def write(
        self,
        table: str,
        frame: pd.DataFrame,
        overwrite: bool = False,
    ) -> None:
        """Write *frame* to a named Parquet table, deduplicating by primary key.

        If the table file already exists and ``overwrite`` is false, the new
        data is concatenated with the existing data and duplicates (according
        to the table's primary key) are removed, keeping the *last* occurrence.
        If ``overwrite`` is true, the existing table is replaced.

        Parameters
        ----------
        table : str
            Name of the table (e.g. ``"prices"`` or ``"news"``).
        frame : pd.DataFrame
            Data to persist. Missing schema columns are filled with ``NaN``.
        overwrite : bool
            Whether to replace the existing table instead of merging into it.
        """
        columns = self._columns(table)
        keys = self._primary_keys(table)
        path = self._table_path(table)

        new_data = frame.copy()
        for col in columns:
            if col not in new_data.columns:
                new_data[col] = pd.NA
        new_data = new_data.reindex(columns=columns)

        if path.exists() and not overwrite:
            existing = pd.read_parquet(path)
            for col in columns:
                if col not in existing.columns:
                    existing[col] = pd.NA
            existing = existing.reindex(columns=columns)
            combined = pd.concat([existing, new_data], ignore_index=True)
        else:
            combined = new_data

        combined = combined.reindex(columns=columns)
        combined = combined.drop_duplicates(subset=keys, keep="last")

        path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_parquet(path, index=False)
        logger.info("Wrote %d rows to %s", len(combined), path)

    def read(self, table: str) -> pd.DataFrame:
        """Read an entire Parquet table.

        Parameters
        ----------
        table : str
            Name of the table (e.g. ``"prices"`` or ``"news"``).

        Returns
        -------
        pd.DataFrame
            The table contents.
        """
        path = self._table_path(table)
        if not path.exists():
            raise FileNotFoundError(f"Table '{table}' not found at {path}")

        frame = pd.read_parquet(path)
        columns = TABLE_COLUMNS.get(table)
        if columns is not None:
            frame = frame.reindex(columns=columns)
        return frame
