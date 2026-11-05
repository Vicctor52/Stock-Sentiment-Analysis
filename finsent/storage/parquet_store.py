from pathlib import Path

import pandas as pd

from finsent.logging import get_logger

logger = get_logger(__name__)


class ParquetStore:
    """Read and write named Parquet tables under a root directory.

    Attributes:
        root: Base directory containing the Parquet tables.
    """

    def __init__(self, root: Path) -> None:
        """Initialize the store.

        Args:
            root: Base directory for Parquet tables.
        """
        self.root = root

    def write(self, table: str, frame: pd.DataFrame) -> None:
        """Write a DataFrame to a named Parquet table.

        Args:
            table: Logical table name.
            frame: Data to persist.
        """
        raise NotImplementedError

    def read(self, table: str) -> pd.DataFrame:
        """Read a named Parquet table.

        Args:
            table: Logical table name.

        Returns:
            The table contents.
        """
        raise NotImplementedError
