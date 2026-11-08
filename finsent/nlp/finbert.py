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
        self.batch_size = max(1, batch_size)
        self._tokenizer = None
        self._model = None
        self._labels: dict[int, str] = {}

    def _load_model(self) -> None:
        """Load the tokenizer and model on first use."""
        if self._model is not None and self._tokenizer is not None:
            return

        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        logger.info(
            "Loading FinBERT model %s (revision=%s, device=%s)",
            self.model_name,
            self.revision,
            self.device,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, revision=self.revision)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, revision=self.revision
        )
        self._model.to(self.device)
        self._model.eval()

        self._labels = {
            int(index): str(label).lower() for index, label in self._model.config.id2label.items()
        }
        missing = {"positive", "negative", "neutral"} - set(self._labels.values())
        if missing:
            raise RuntimeError(f"Model labels are missing expected classes: {sorted(missing)}")

    def score(self, texts: list[str]) -> pd.DataFrame:
        """Score texts with FinBERT.

        Args:
            texts: Texts to classify.

        Returns:
            A frame with label, positive, negative, neutral, and compound scores.
        """
        columns = ["label", "p_pos", "p_neg", "p_neu", "score"]
        if not texts:
            return pd.DataFrame(columns=columns)

        self._load_model()
        assert self._tokenizer is not None
        assert self._model is not None

        import torch

        rows: list[dict[str, object]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            inputs = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            inputs = {key: value.to(self.device) for key, value in inputs.items()}

            with torch.inference_mode():
                logits = self._model(**inputs).logits
                probabilities = torch.softmax(logits, dim=-1).cpu()

            for probability in probabilities:
                by_label = {
                    label: float(probability[index]) for index, label in self._labels.items()
                }
                p_pos = by_label["positive"]
                p_neg = by_label["negative"]
                p_neu = by_label["neutral"]
                rows.append(
                    {
                        "label": max(by_label, key=by_label.get),
                        "p_pos": p_pos,
                        "p_neg": p_neg,
                        "p_neu": p_neu,
                        "score": p_pos - p_neg,
                    }
                )

        return pd.DataFrame(rows, columns=columns)
