import pandas as pd

from finsent.logging import get_logger

logger = get_logger(__name__)


class FinBertScorer:
    def __init__(
        self,
        model_name: str = "ProsusAI/finbert",
        revision: str = "main",
        device: str = "cpu",
        batch_size: int = 16,
    ) -> None:
        self.model_name = model_name
        self.revision = revision
        self.device = device
        self.batch_size = batch_size

    def score(self, texts: list[str]) -> pd.DataFrame:
        raise NotImplementedError
