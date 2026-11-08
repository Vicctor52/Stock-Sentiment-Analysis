import numpy as np
import pandas as pd
import pytest

from finsent.analysis.returns import compute_log_returns
from finsent.storage.schemas import RETURNS_COLUMNS


def _prices() -> pd.DataFrame:
    dates = pd.to_datetime(
        ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
    )
    return pd.DataFrame(
        {
            "ticker": ["AAPL"] * 4 + ["MSFT"] * 4,
            "session_date": dates.tolist() * 2,
            "adj_close": [100.0, 110.0, 121.0, 133.1, 200.0, 180.0, 198.0, 217.8],
        }
    )


def test_compute_log_returns_per_ticker() -> None:
    returns = compute_log_returns(_prices())

    assert list(returns.columns) == RETURNS_COLUMNS
    assert len(returns) == 8

    aapl = returns[returns["ticker"] == "AAPL"].reset_index(drop=True)
    assert pd.isna(aapl.loc[0, "log_return"])
    assert aapl.loc[1, "log_return"] == pytest.approx(np.log(1.1))
    assert aapl.loc[3, "log_return"] == pytest.approx(np.log(1.1))
    assert aapl.loc[0, "fwd_return_1"] == pytest.approx(np.log(1.1))
    assert aapl.loc[3, "fwd_return_1"] is np.nan or pd.isna(
        aapl.loc[3, "fwd_return_1"]
    )
    assert aapl.loc[0, "fwd_return_2"] == pytest.approx(np.log(1.21))
    assert pd.isna(aapl.loc[2, "fwd_return_2"])

    msft = returns[returns["ticker"] == "MSFT"].reset_index(drop=True)
    assert pd.isna(msft.loc[0, "log_return"])
    assert msft.loc[1, "log_return"] == pytest.approx(np.log(0.9))


def test_compute_log_returns_accepts_date_and_close() -> None:
    prices = _prices().rename(columns={"session_date": "date", "adj_close": "close"})

    returns = compute_log_returns(prices)

    assert len(returns) == 8
    assert "session_date" in returns.columns
    assert "log_return" in returns.columns


def test_compute_log_returns_empty() -> None:
    returns = compute_log_returns(pd.DataFrame())

    assert list(returns.columns) == RETURNS_COLUMNS
    assert returns.empty


def test_compute_log_returns_requires_columns() -> None:
    with pytest.raises(KeyError):
        compute_log_returns(pd.DataFrame({"ticker": ["AAPL"], "adj_close": [100.0]}))
