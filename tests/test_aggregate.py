import pandas as pd
import pytest

from finsent.nlp.aggregate import aggregate_daily, sentiment_index
from finsent.storage.schemas import DAILY_SENTIMENT_COLUMNS


def _sentiment_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": ["NVDA", "NVDA", "NVDA", "AAPL", "AAPL"],
            "session_date": pd.to_datetime(
                [
                    "2024-01-02",
                    "2024-01-02",
                    "2024-01-02",
                    "2024-01-02",
                    "2024-01-03",
                ]
            ),
            "label": ["Positive", "negative", "neutral", "positive", "positive"],
            "score": [0.6, -0.4, 0.0, 0.5, 0.3],
        }
    )


def test_sentiment_index() -> None:
    assert sentiment_index(2, 1, 1) == 0.25
    assert sentiment_index(0, 0, 0) == 0.0
    assert sentiment_index(3, 0, 0) == 1.0
    assert sentiment_index(0, 3, 0) == -1.0


def test_aggregate_daily_counts_and_scores() -> None:
    daily = aggregate_daily(_sentiment_frame())

    assert list(daily.columns) == DAILY_SENTIMENT_COLUMNS
    assert len(daily) == 3

    nvda = daily[(daily["ticker"] == "NVDA")].iloc[0]
    assert nvda["n_pos"] == 1
    assert nvda["n_neg"] == 1
    assert nvda["n_neu"] == 1
    assert nvda["n_docs"] == 3
    assert nvda["sentiment_index"] == 0.0
    assert nvda["sentiment_expected"] == pytest.approx(0.0666667)
    assert nvda["source"] == "news"


def test_aggregate_daily_empty() -> None:
    daily = aggregate_daily(pd.DataFrame())

    assert list(daily.columns) == DAILY_SENTIMENT_COLUMNS
    assert daily.empty


def test_aggregate_daily_requires_columns() -> None:
    with pytest.raises(KeyError):
        aggregate_daily(pd.DataFrame({"ticker": ["NVDA"]}))
