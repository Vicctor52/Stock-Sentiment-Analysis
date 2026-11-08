import numpy as np
import pandas as pd
import pytest

from finsent.analysis.lead_lag import lead_lag_correlation
from finsent.storage.schemas import LEAD_LAG_COLUMNS


def _features() -> pd.DataFrame:
    sentiment = np.arange(12, dtype=float)
    returns = np.full(12, np.nan)
    returns[1:] = sentiment[:-1]
    return pd.DataFrame(
        {
            "ticker": ["NVDA"] * 12,
            "source": ["news"] * 12,
            "session_date": pd.date_range("2024-01-02", periods=12, freq="B"),
            "sentiment_index": sentiment,
            "log_return": returns,
        }
    )


def test_lead_lag_detects_one_day_lead() -> None:
    results = lead_lag_correlation(
        _features(),
        lags=(0, 1),
        methods=("pearson", "spearman"),
        min_observations=3,
    )

    assert list(results.columns) == LEAD_LAG_COLUMNS
    lag_one = results[results["lag"] == 1]
    assert len(lag_one) == 2
    assert set(lag_one["method"]) == {"pearson", "spearman"}
    assert lag_one["r"].tolist() == pytest.approx([1.0, 1.0])
    assert lag_one["n_obs"].tolist() == [11, 11]
    assert lag_one["adjusted_p"].between(0.0, 1.0).all()
    assert lag_one["ci_low"].gt(0.9).all()


def test_lead_lag_respects_min_observations() -> None:
    results = lead_lag_correlation(_features(), min_observations=20)

    assert list(results.columns) == LEAD_LAG_COLUMNS
    assert results.empty


def test_lead_lag_empty_input() -> None:
    results = lead_lag_correlation(pd.DataFrame())

    assert list(results.columns) == LEAD_LAG_COLUMNS
    assert results.empty


def test_lead_lag_rejects_unknown_method() -> None:
    with pytest.raises(ValueError):
        lead_lag_correlation(_features(), methods=("kendall",))
