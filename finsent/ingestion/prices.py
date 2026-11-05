from __future__ import annotations

import pandas as pd
import yfinance as yf
from tenacity import retry, stop_after_attempt, wait_exponential

from finsent.logging import get_logger
from finsent.storage.schemas import PRICES_COLUMNS

logger = get_logger(__name__)

_PRICE_VALUE_COLUMNS = [
    "session_date",
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume",
]

_COLUMN_MAP = {
    "Date": "session_date",
    "Datetime": "session_date",
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Adj Close": "adj_close",
    "Volume": "volume",
}


class PriceFetchError(RuntimeError):
    pass


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True,
)
def _download_ticker(ticker: str, period: str, interval: str) -> pd.DataFrame:
    return yf.Ticker(ticker).history(
        period=period,
        interval=interval,
        auto_adjust=False,
    )


def _normalize_session_dates(values: pd.Series) -> pd.Series:
    dates = pd.to_datetime(values)
    if dates.dt.tz is not None:
        dates = dates.dt.tz_localize(None)
    return dates.dt.normalize()


def _fetch_single_ticker(ticker: str, period: str, interval: str) -> pd.DataFrame:
    try:
        raw = _download_ticker(ticker, period, interval)
    except Exception as exc:
        raise PriceFetchError(f"Failed to download prices for {ticker}: {exc}") from exc

    if raw is None or raw.empty:
        raise PriceFetchError(f"No price data returned for {ticker}")

    frame = raw.reset_index().rename(columns={"index": "session_date"})
    frame = frame.rename(columns=_COLUMN_MAP)
    missing = [column for column in _PRICE_VALUE_COLUMNS if column not in frame.columns]
    if missing:
        raise PriceFetchError(f"Missing columns for {ticker}: {', '.join(missing)}")

    frame["session_date"] = _normalize_session_dates(frame["session_date"])

    for column in ["open", "high", "low", "close", "adj_close", "volume"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame = frame.dropna(subset=["close", "adj_close"]).copy()
    if frame.empty:
        raise PriceFetchError(f"No valid close prices for {ticker}")

    frame["volume"] = frame["volume"].astype("Int64")
    return frame[_PRICE_VALUE_COLUMNS]


def _validate(frame: pd.DataFrame) -> pd.DataFrame:
    invalid_prices = (
        (frame["open"] < 0)
        | (frame["high"] < 0)
        | (frame["low"] < 0)
        | (frame["close"] < 0)
        | (frame["adj_close"] < 0)
    )
    invalid_range = frame["high"] < frame["low"]
    invalid_volume = frame["volume"] < 0
    invalid = (invalid_prices | invalid_range | invalid_volume).fillna(False)

    if invalid.any():
        logger.warning("Dropping %d invalid price rows", int(invalid.sum()))
        frame = frame.loc[~invalid].copy()

    if frame.empty:
        raise PriceFetchError("No valid price rows after validation")

    return frame


def fetch_prices(
    tickers: list[str],
    period: str = "12mo",
    interval: str = "1d",
) -> pd.DataFrame:
    if not tickers:
        raise ValueError("At least one ticker is required")

    ingested_at = pd.Timestamp.now(tz="UTC")
    frames: list[pd.DataFrame] = []
    failures: list[str] = []

    for ticker in tickers:
        try:
            frame = _fetch_single_ticker(ticker, period, interval)
        except PriceFetchError as exc:
            failures.append(ticker)
            logger.warning("%s", exc)
            continue

        frame = frame.assign(
            ticker=ticker,
            source="yfinance",
            ingested_at=ingested_at,
        )
        frames.append(frame)
        logger.info("Fetched %d rows for %s", len(frame), ticker)

    if not frames:
        raise PriceFetchError(f"No price data fetched for any ticker: {', '.join(tickers)}")

    result = pd.concat(frames, ignore_index=True)
    result = result.drop_duplicates(
        subset=["ticker", "session_date"],
        keep="last",
    )
    result = _validate(result)
    result = result.sort_values(["ticker", "session_date"], ignore_index=True)

    result["ticker"] = result["ticker"].astype("string")
    result["source"] = result["source"].astype("string")
    result["ingested_at"] = pd.to_datetime(result["ingested_at"], utc=True)
    result = result[PRICES_COLUMNS]

    if failures:
        logger.warning(
            "Price ingestion completed with failures: %s",
            ", ".join(failures),
        )

    logger.info(
        "Fetched %d rows for %d/%d tickers",
        len(result),
        len(frames),
        len(tickers),
    )
    return result
