import pandas as pd

from finsent.storage.schemas import DAILY_SENTIMENT_COLUMNS


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
        sentiment: Frame with ticker, session date, label, and score columns.

    Returns:
        A frame with daily sentiment counts, indices, and expected sentiment
        per ticker.
    """
    if sentiment.empty:
        return pd.DataFrame(columns=DAILY_SENTIMENT_COLUMNS)

    required = {"ticker", "session_date", "label", "score"}
    missing = required - set(sentiment.columns)
    if missing:
        raise KeyError(f"Missing columns for daily aggregation: {sorted(missing)}")

    frame = sentiment.dropna(subset=["ticker", "session_date"]).copy()
    frame["label"] = frame["label"].astype(str).str.lower()

    daily = (
        frame.groupby(["ticker", "session_date"], as_index=False)
        .agg(
            n_pos=("label", lambda values: int((values == "positive").sum())),
            n_neg=("label", lambda values: int((values == "negative").sum())),
            n_neu=("label", lambda values: int((values == "neutral").sum())),
            n_docs=("label", "size"),
            sentiment_expected=("score", "mean"),
        )
        .reset_index(drop=True)
    )

    daily["source"] = "news"
    daily["sentiment_index"] = [
        sentiment_index(int(row.n_pos), int(row.n_neg), int(row.n_neu))
        for row in daily.itertuples(index=False)
    ]

    daily = daily.reindex(columns=DAILY_SENTIMENT_COLUMNS)
    return daily.sort_values(["ticker", "session_date"]).reset_index(drop=True)
