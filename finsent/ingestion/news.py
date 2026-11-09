import hashlib
import os
import time
from urllib.parse import urlparse

import pandas as pd
import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from finsent.logging import get_logger
from finsent.storage.schemas import NEWS_ARTICLES_COLUMNS

logger = get_logger(__name__)

_CURRENTS_SEARCH_URL = "https://api.currentsapi.services/v1/search"
_DATE_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
_MIN_REQUEST_INTERVAL = 1.1
_MAX_BURST_RETRIES = 3


class CurrentsRateLimitError(RuntimeError):
    """Raised when the Currents API daily or burst quota is exhausted."""


class CurrentsAPIError(RuntimeError):
    """Raised for non-retryable Currents API errors."""


class CurrentsNewsClient:
    """Minimal client for the Currents API search endpoint."""

    def __init__(
        self,
        api_key: str,
        timeout: float = 20.0,
        min_interval: float = _MIN_REQUEST_INTERVAL,
    ) -> None:
        self.api_key = api_key
        self.timeout = timeout
        self.min_interval = min_interval
        self._last_request_at: float | None = None

    def _throttle(self) -> None:
        if self.min_interval <= 0:
            return
        now = time.monotonic()
        if self._last_request_at is not None:
            wait = self.min_interval - (now - self._last_request_at)
            if wait > 0:
                time.sleep(wait)
        self._last_request_at = time.monotonic()

    @staticmethod
    def _is_burst_limit(response: requests.Response) -> bool:
        daily_remaining = response.headers.get("X-RateLimit-Remaining")
        burst_remaining = response.headers.get("X-RateLimit-Burst-Remaining")
        return burst_remaining == "0" and daily_remaining not in (None, "0")

    @staticmethod
    def _retry_after_seconds(response: requests.Response) -> float:
        try:
            return min(float(response.headers.get("Retry-After", "30")), 120.0)
        except ValueError:
            return 30.0

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(
            (requests.ConnectionError, requests.Timeout, requests.HTTPError)
        ),
        reraise=True,
    )
    def search(
        self,
        keywords: str,
        start_date: str | None = None,
        end_date: str | None = None,
        page_number: int = 1,
        page_size: int = 20,
    ) -> dict[str, object]:
        """Search Currents API articles for a keyword and optional date range."""
        params: dict[str, object] = {
            "keywords": keywords,
            "page_number": page_number,
            "page_size": page_size,
        }
        if start_date is not None:
            params["start_date"] = start_date
        if end_date is not None:
            params["end_date"] = end_date

        for attempt in range(_MAX_BURST_RETRIES + 1):
            self._throttle()
            response = requests.get(
                _CURRENTS_SEARCH_URL,
                params=params,
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=self.timeout,
            )

            if response.status_code == 429:
                if self._is_burst_limit(response) and attempt < _MAX_BURST_RETRIES:
                    time.sleep(self._retry_after_seconds(response))
                    continue
                retry_after = response.headers.get("Retry-After", "unknown")
                raise CurrentsRateLimitError(
                    "Currents API quota exhausted; retry after "
                    f"{retry_after} seconds or at 00:00 UTC."
                )
            if response.status_code >= 500:
                response.raise_for_status()
            if response.status_code >= 400:
                raise CurrentsAPIError(
                    f"Currents API request failed with status {response.status_code}: "
                    f"{response.text[:200]}"
                )

            try:
                payload = response.json()
            except ValueError as exc:
                raise CurrentsAPIError("Currents API returned invalid JSON.") from exc

            if not isinstance(payload, dict) or payload.get("status") != "ok":
                raise CurrentsAPIError(f"Currents API returned an error payload: {payload}")

            return payload

        raise CurrentsRateLimitError("Currents API burst limit did not clear.")


def _format_timestamp(value: object) -> str | None:
    """Format a timestamp as the RFC 3339 string expected by Currents."""
    if value is None:
        return None
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC").strftime(_DATE_FORMAT)


def _utc_timestamp(value: object) -> pd.Timestamp:
    """Convert a value to a timezone-aware UTC timestamp."""
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _date_windows(
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


def _source_name(article: dict[str, object]) -> str:
    """Return a human-readable source label for an article."""
    author = article.get("author")
    if isinstance(author, str) and author.strip():
        return author.strip()

    url = article.get("url")
    if isinstance(url, str):
        domain = urlparse(url).netloc.removeprefix("www.")
        if domain:
            return domain
    return ""


def _normalize_article(
    article: dict[str, object],
    ticker: str,
    ingested_at: pd.Timestamp,
) -> dict[str, object] | None:
    """Normalize a Currents article to the internal news schema."""
    url = article.get("url") or ""
    title = article.get("title") or ""
    published = article.get("published")
    if not url and not title:
        return None

    raw_id = url or f"{title}|{published}"
    return {
        "article_id": hashlib.sha256(str(raw_id).encode("utf-8")).hexdigest(),
        "ticker": ticker,
        "source_name": _source_name(article),
        "published_utc": pd.to_datetime(published, utc=True, errors="coerce"),
        "title": title,
        "description": article.get("description") or "",
        "content": "",
        "url": url,
        "ingested_at": ingested_at,
    }


def _fetch_ticker(
    client: CurrentsNewsClient,
    ticker: str,
    windows: list[tuple[pd.Timestamp | None, pd.Timestamp | None]],
    page_size: int,
    max_pages: int,
    ingested_at: pd.Timestamp,
) -> list[dict[str, object]]:
    """Fetch and normalize articles for one ticker across date windows."""
    rows: list[dict[str, object]] = []
    for window_start, window_end in windows:
        start_date = _format_timestamp(window_start)
        end_date = _format_timestamp(window_end)

        for page in range(1, max_pages + 1):
            payload = client.search(
                keywords=ticker,
                start_date=start_date,
                end_date=end_date,
                page_number=page,
                page_size=page_size,
            )
            articles = payload.get("news") or []
            if not isinstance(articles, list):
                raise CurrentsAPIError("Currents API returned a non-list 'news' field.")

            for article in articles:
                if isinstance(article, dict):
                    normalized = _normalize_article(article, ticker, ingested_at)
                    if normalized is not None:
                        rows.append(normalized)

            if len(articles) < page_size:
                break
            if payload.get("page_has_results") is False:
                break

    return rows


def fetch_news(
    tickers: list[str],
    page_size: int = 20,
    from_date: str | None = None,
    to_date: str | None = None,
    chunk_days: int = 7,
    max_pages: int = 3,
    strict: bool = False,
) -> pd.DataFrame:
    """Fetch recent news articles for each ticker from Currents API.

    Args:
        tickers: Ticker symbols or search terms to query individually.
        page_size: Maximum number of articles per request. Currents free plan
            allows at most 20.
        from_date: Optional ISO timestamp lower bound.
        to_date: Optional exclusive ISO timestamp upper bound. Articles
            published exactly at this instant are excluded so news at a
            session close is assigned to the next session.
        chunk_days: Maximum inclusive calendar days per search request. The
            Currents free plan allows at most 7.
        max_pages: Maximum pages fetched per ticker and date window.
        strict: Re-raise ticker-level errors so scheduled runs do not mark a
            partially collected interval as complete.

    Returns:
        A normalized DataFrame with one row per article and ticker. Columns:
        article_id, ticker, source_name, published_utc, title, description,
        content, url, ingested_at.
    """
    api_key = os.getenv("CURRENTS_API_KEY")
    if not api_key:
        raise ValueError("CURRENTS_API_KEY environment variable not set")

    if not tickers:
        logger.warning("No tickers provided; returning empty DataFrame.")
        return pd.DataFrame(columns=NEWS_ARTICLES_COLUMNS)

    if page_size < 1:
        raise ValueError("page_size must be positive.")
    if chunk_days < 2:
        raise ValueError("chunk_days must be at least 2.")
    if max_pages < 1:
        raise ValueError("max_pages must be positive.")
    if (from_date is None) != (to_date is None):
        raise ValueError("Both from_date and to_date are required for a date range.")

    if from_date is not None and to_date is not None:
        windows = _date_windows(pd.Timestamp(from_date), pd.Timestamp(to_date), chunk_days)
    else:
        windows = [(None, None)]

    client = CurrentsNewsClient(api_key)
    ingested_at = pd.Timestamp.now(tz="UTC")
    rows: list[dict[str, object]] = []
    failures: list[str] = []

    for ticker in tickers:
        try:
            ticker_rows = _fetch_ticker(
                client,
                ticker,
                windows,
                page_size,
                max_pages,
                ingested_at,
            )
        except CurrentsRateLimitError:
            raise
        except Exception as exc:
            failures.append(ticker)
            logger.error("Error fetching news for %s from Currents: %s", ticker, exc)
            if strict:
                raise RuntimeError(f"Failed to fetch news for {ticker}") from exc
            continue

        rows.extend(ticker_rows)
        logger.info("Fetched %d articles for %s", len(ticker_rows), ticker)

    frame = pd.DataFrame(rows, columns=NEWS_ARTICLES_COLUMNS)
    if not frame.empty:
        frame = frame.drop_duplicates(subset=["article_id", "ticker"], keep="last")
        if to_date is not None:
            cutoff = _utc_timestamp(to_date)
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
        "Fetched %d news articles for %d tickers via Currents API",
        len(frame),
        len(tickers),
    )
    return frame
