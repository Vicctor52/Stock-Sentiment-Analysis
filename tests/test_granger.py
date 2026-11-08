import numpy as np
import pandas as pd
import pytest

from finsent.analysis.granger import granger_causality
from finsent.storage.schemas import GRANGER_COLUMNS


def _features(n_observations: int = 80) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    sentiment = rng.normal(size=n_observations)
    returns = np.zeros(n_observations)
    for index in range(1, n_observations):
        returns[index] = 0.6 * sentiment[index - 1] + rng.normal(scale=0.4)

    return pd.DataFrame(
        {
            "ticker": ["NVDA"] * n_observations,
            "source": ["news"] * n_observations,
            "session_date": pd.date_range(
                "2024-01-02", periods=n_observations, freq="B"
            ),
            "sentiment_index": sentiment,
            "log_return": returns,
        }
    )


def test_granger_detects_sentiment_causality() -> None:
    results = granger_causality(_features(), maxlag=2, min_observations=10)

    assert list(results.columns) == GRANGER_COLUMNS
    assert len(results) == 2
    assert results["lag"].tolist() == [1, 2]
    assert results["maxlag"].unique().tolist() == [2]
    assert results["test"].unique().tolist() == ["ssr_ftest"]
    assert results["n_obs"].unique().tolist() == [80]
    assert results["p_value"].between(0.0, 1.0).all()
    assert results.loc[results["lag"] == 1, "p_value"].iloc[0] < 0.05
    assert results["adjusted_p"].between(0.0, 1.0).all()


def test_granger_insufficient_observations() -> None:
    results = granger_causality(_features(10), maxlag=5, min_observations=5)

    assert list(results.columns) == GRANGER_COLUMNS
    assert results.empty


def test_granger_empty_input() -> None:
    results = granger_causality(pd.DataFrame())

    assert list(results.columns) == GRANGER_COLUMNS
    assert results.empty


def test_granger_rejects_invalid_maxlag() -> None:
    with pytest.raises(ValueError):
        granger_causality(_features(), maxlag=0)
