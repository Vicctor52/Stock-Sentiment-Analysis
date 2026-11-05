import pandas as pd

from finsent.logging import get_logger

logger = get_logger(__name__)


class FinBertScorer:
    """Score text with a FinBERT sentiment model.

    Attributes:
        model_name: Hugging Face model identifier.
        revision: Model revision to load.
        device: Torch device used for inference.
        batch_size: Number of texts processed per batch.
    """

    def __init__(
        self,
        model_name: str = "ProsusAI/finbert",
        revision: str = "main",
        device: str = "cpu",
        batch_size: int = 16,
    ) -> None:
        """Initialize the scorer.

        Args:
            model_name: Hugging Face model identifier.
            revision: Model revision to load.
            device: Torch device used for inference.
            batch_size: Number of texts processed per batch.
        """
        self.model_name = model_name
        self.revision = revision
        self.device = device
        self.batch_size = batch_size

    def score(self, texts: list[str]) -> pd.DataFrame:
        """Score texts with FinBERT.

        Args:
            texts: Texts to classify.

        Returns:
            A frame with positive, negative, neutral, and compound scores.
        """
        raise NotImplementedError
