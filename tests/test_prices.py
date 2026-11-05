from __future__ import annotations

import pandas as pd
import pytest

from finsent.ingestion import prices
from finsent.storage.schemas import PRICES_COLUMNS


def _fake_history(periods: int = 3) -> pd.DataFrame:
    index = pd.date_range(
        "2024-01-02",
        periods=periods,
        freq="B",
        tz="America/New_York",
        name="Date",
    )
    return pd.DataFrame(
        {
            "Open": [100.0] * periods,
            "High": [110.0] * periods,
            "Low": [90.0] * periods,
            "Close": [105.0] * periods,
            "Adj Close": [104.0] * periods,
            "Volume": [1_000_000] * periods,
            "Dividends": [0.0] * periods,
            "Stock Splits": [0.0] * periods,
        },
        index=index,
    )


def test_fetch_prices_normalizes_and_sorts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        prices,
        "_download_ticker",
        lambda ticker, period, interval: _fake_history(),
    )

    result = prices.fetch_prices(["NVDA", "AAPL"], period="1mo")

    assert list(result.columns) == PRICES_COLUMNS
    assert result["ticker"].tolist() == ["AAPL"] * 3 + ["NVDA"] * 3
    assert pd.api.types.is_datetime64_any_dtype(result["session_date"])
    assert result["session_date"].dt.tz is None
    assert (result["session_date"].dt.hour == 0).all()
    assert result["ingested_at"].dt.tz is not None
    assert str(result["volume"].dtype) == "Int64"
    assert (result["source"] == "yfinance").all()
    assert result["adj_close"].tolist() == [104.0] * 6


def test_fetch_prices_continues_after_ticker_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_download(ticker: str, period: str, interval: str) -> pd.DataFrame:
        if ticker == "BAD":
            raise RuntimeError("boom")
        return _fake_history()

    monkeypatch.setattr(prices, "_download_ticker", fake_download)

    result = prices.fetch_prices(["BAD", "NVDA"])

    assert result["ticker"].unique().tolist() == ["NVDA"]


def test_fetch_prices_raises_when_all_tickers_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_download(ticker: str, period: str, interval: str) -> pd.DataFrame:
        raise RuntimeError("boom")

    monkeypatch.setattr(prices, "_download_ticker", fake_download)

    with pytest.raises(prices.PriceFetchError):
        prices.fetch_prices(["NVDA", "AAPL"])


def test_fetch_prices_requires_at_least_one_ticker() -> None:
    with pytest.raises(ValueError):
        prices.fetch_prices([])


def test_fetch_prices_raises_for_empty_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        prices,
        "_download_ticker",
        lambda ticker, period, interval: pd.DataFrame(),
    )

    with pytest.raises(prices.PriceFetchError):
        prices.fetch_prices(["NVDA"])


def test_fetch_prices_raises_for_missing_columns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        prices,
        "_download_ticker",
        lambda ticker, period, interval: _fake_history().drop(columns=["Adj Close"]),
    )

    with pytest.raises(prices.PriceFetchError):
        prices.fetch_prices(["NVDA"])


def test_fetch_prices_deduplicates_same_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_download(ticker: str, period: str, interval: str) -> pd.DataFrame:
        frame = _fake_history(periods=2)
        duplicate = frame.iloc[[0]].copy()
        duplicate["Close"] = 999.0
        duplicate["Adj Close"] = 999.0
        return pd.concat([frame, duplicate])

    monkeypatch.setattr(prices, "_download_ticker", fake_download)

    result = prices.fetch_prices(["NVDA"])

    assert len(result) == 2
    first_session = pd.Timestamp("2024-01-02")
    first_row = result.loc[result["session_date"] == first_session]
    assert first_row["close"].item() == 999.0


def test_fetch_prices_drops_invalid_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_download(ticker: str, period: str, interval: str) -> pd.DataFrame:
        frame = _fake_history(periods=2)
        frame.loc[frame.index[0], "High"] = 1.0
        return frame

    monkeypatch.setattr(prices, "_download_ticker", fake_download)

    result = prices.fetch_prices(["NVDA"])

    assert len(result) == 1


def test_fetch_prices_handles_unnamed_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_download(ticker: str, period: str, interval: str) -> pd.DataFrame:
        frame = _fake_history()
        frame.index.name = None
        return frame

    monkeypatch.setattr(prices, "_download_ticker", fake_download)

    result = prices.fetch_prices(["NVDA"])

    assert len(result) == 3


def test_normalize_session_dates_removes_timezone() -> None:
    values = pd.Series(
        pd.to_datetime(["2024-01-02 00:00", "2024-01-03 00:00"]).tz_localize("America/New_York")
    )

    result = prices._normalize_session_dates(values)

    assert result.dt.tz is None
    assert result.tolist() == [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")]


def test_normalize_session_dates_handles_naive_values() -> None:
    values = pd.Series(pd.to_datetime(["2024-01-02 15:30"]))

    result = prices._normalize_session_dates(values)

    assert result.tolist() == [pd.Timestamp("2024-01-02")]
