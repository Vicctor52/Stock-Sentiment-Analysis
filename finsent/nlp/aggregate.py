import pandas as pd


def sentiment_index(positive: int, negative: int, neutral: int) -> float:
    """Compute a normalized sentiment index from label counts.

    Args:
        positive: Number of positive documents.
        negative: Number of negative documents.
        neutral: Number of neutral documents.

    Returns:
        The sentiment index in the range ``[-1, 1]``, or ``0.0`` when there
        are no documents.
    """
    total = positive + negative + neutral
    if total == 0:
        return 0.0
    return (positive - negative) / total


def aggregate_daily(sentiment: pd.DataFrame) -> pd.DataFrame:
    """Aggregate document-level sentiment into daily ticker metrics.

    Args:
        sentiment: Frame with ticker, session date, and sentiment labels.

    Returns:
        A frame with daily sentiment counts and indices per ticker.
    """
    raise NotImplementedError
