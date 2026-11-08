from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from finsent.storage.schemas import (
    DAILY_SENTIMENT_COLUMNS,
    GRANGER_COLUMNS,
    LEAD_LAG_COLUMNS,
    RETURNS_COLUMNS,
)


@dataclass(frozen=True)
class DashboardData:
    daily_sentiment: pd.DataFrame
    returns: pd.DataFrame
    lead_lag: pd.DataFrame
    granger: pd.DataFrame
    missing_tables: tuple[str, ...]


_TABLE_COLUMNS = {
    "daily_sentiment": DAILY_SENTIMENT_COLUMNS,
    "returns": RETURNS_COLUMNS,
    "lead_lag": LEAD_LAG_COLUMNS,
    "granger": GRANGER_COLUMNS,
}


def load_parquet(path: Path) -> pd.DataFrame:
    """Load a Parquet file into a DataFrame.

    Args:
        path: Path to the Parquet file.

    Returns:
        The loaded data. Raises ``FileNotFoundError`` when the table is absent.
    """
    return pd.read_parquet(path)


def load_dashboard_data(processed_dir: Path) -> DashboardData:
    """Load available analysis tables, retaining which results are missing."""
    tables: dict[str, pd.DataFrame] = {}
    missing_tables = []

    for table, columns in _TABLE_COLUMNS.items():
        try:
            tables[table] = load_parquet(processed_dir / f"{table}.parquet")
        except FileNotFoundError:
            tables[table] = pd.DataFrame(columns=columns)
            missing_tables.append(table)

    return DashboardData(
        daily_sentiment=tables["daily_sentiment"],
        returns=tables["returns"],
        lead_lag=tables["lead_lag"],
        granger=tables["granger"],
        missing_tables=tuple(missing_tables),
    )
