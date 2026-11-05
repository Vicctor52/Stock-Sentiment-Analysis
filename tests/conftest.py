import pandas as pd
import pytest


@pytest.fixture
def sample_sentiment() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": ["NVDA", "NVDA", "NVDA"],
            "session_date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
            "sentiment_index": [0.2, -0.1, 0.4],
        }
    )
