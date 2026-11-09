from __future__ import annotations

import hashlib
from collections.abc import Callable
from urllib.parse import urlparse

import pandas as pd

from finsent.logging import get_logger
from finsent.storage.schemas import NEWS_ARTICLES_COLUMNS

logger = get_logger(__name__)


class NewsRateLimitError(RuntimeError):
    """Raised when a news provider quota is exhausted."""


class NewsAPIError(RuntimeError):
    """Raised for non-retryable news provider errors."""


def format_timestamp(value: object) -> str | None:
    """Format a timestamp as an RFC 3339 string in UTC."""
    if value is None:
        return None
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")


def utc_timestamp(value: object) -> pd.Timestamp:
    """Convert a value to a timezone-aware UTC timestamp."""
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def date_windows(
    start: pd.Timestamp,
    end: pd.Timestamp,
    chunk_days: int,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Split an inclusive date range into windows of at most ``chunk_days``."""
    if end < start:
        raise ValueError("The news end date must not be before the start date.")
    if chunk_days < 2:
        raise ValueError("chunk_days must be at least 2 to advance the window.")

    step = pd.Timedelta(days=chunk_days - 1)
    windows: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    cursor = start
    while True:
        window_end = min(cursor + step, end)
        windows.append((cursor, window_end))
        if window_end >= end:
            break
        cursor = window_end
    return windows


def domain_from_url(url: object) -> str:
    """Return a normalized source domain from an article URL."""
    if not isinstance(url, str):
        return ""
    return urlparse(url).netloc.removeprefix("www.")


def stable_article_id(url: object, title: object, published: object) -> str:
    """Return a provider-independent article identifier."""
    raw_id = url or f"{title}|{published}"
    return hashlib.sha256(str(raw_id).encode("utf-8")).hexdigest()


def empty_news_frame() -> pd.DataFrame:
    """Return an empty frame with the internal news schema."""
    return pd.DataFrame(columns=NEWS_ARTICLES_COLUMNS)


def collect_news(
    tickers: list[str],
    fetch_ticker: Callable[[str], list[dict[str, object]]],
    *,
    provider_label: str,
    strict: bool,
    to_date: str | None,
) -> pd.DataFrame:
    """Collect per-ticker rows, deduplicate, and enforce the exclusive end bound."""
    rows: list[dict[str, object]] = []
    failures: list[str] = []

    for ticker in tickers:
        try:
            ticker_rows = fetch_ticker(ticker)
        except NewsRateLimitError:
            raise
        except Exception as exc:
            failures.append(ticker)
            logger.error(
                "Error fetching news for %s from %s: %s",
                ticker,
                provider_label,
                exc,
            )
            if strict:
                raise RuntimeError(f"Failed to fetch news for {ticker}") from exc
            continue

        rows.extend(ticker_rows)
        logger.info("Fetched %d articles for %s", len(ticker_rows), ticker)

    frame = pd.DataFrame(rows, columns=NEWS_ARTICLES_COLUMNS)
    if not frame.empty:
        frame = frame.drop_duplicates(subset=["article_id", "ticker"], keep="last")
        if to_date is not None:
            cutoff = utc_timestamp(to_date)
            published = pd.to_datetime(frame["published_utc"], utc=True, errors="coerce")
            before = len(frame)
            frame = frame[published < cutoff]
            dropped = before - len(frame)
            if dropped:
                logger.info(
                    "Excluded %d articles published at or after %s.",
                    dropped,
                    to_date,
                )

    if failures:
        logger.warning(
            "News ingestion completed with failures: %s",
            ", ".join(failures),
        )
    logger.info(
        "Fetched %d news articles for %d tickers via %s",
        len(frame),
        len(tickers),
        provider_label,
    )
    return frame
