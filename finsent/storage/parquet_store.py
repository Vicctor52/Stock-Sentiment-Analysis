from pathlib import Path

import pandas as pd

from finsent.logging import get_logger

logger = get_logger(__name__)


class ParquetStore:
    def __init__(self, root: Path) -> None:
        self.root = root

    def write(self, table: str, frame: pd.DataFrame) -> None:
        raise NotImplementedError

    def read(self, table: str) -> pd.DataFrame:
        raise NotImplementedError
