import pandas as pd


def sentiment_index(positive: int, negative: int, neutral: int) -> float:
    total = positive + negative + neutral
    if total == 0:
        return 0.0
    return (positive - negative) / total


def aggregate_daily(sentiment: pd.DataFrame) -> pd.DataFrame:
    raise NotImplementedError
