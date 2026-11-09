from __future__ import annotations

import os
import time

import pandas as pd
import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from finsent.ingestion.providers.base import (
    NewsAPIError,
    NewsRateLimitError,
    collect_news,
    date_windows,
    domain_from_url,
    format_timestamp,
    stable_article_id,
    utc_timestamp,
)
from finsent.logging import get_logger

logger = get_logger(__name__)

_CURRENTS_SEARCH_URL = "https://api.currentsapi.services/v1/search"
_MIN_REQUEST_INTERVAL = 1.1
_MAX_BURST_RETRIES = 3


class CurrentsRateLimitError(NewsRateLimitError):
    """Raised when the Currents API daily or burst quota is exhausted."""


class CurrentsAPIError(NewsAPIError):
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


def _source_name(article: dict[str, object]) -> str:
    """Return a human-readable source label for an article."""
    author = article.get("author")
    if isinstance(author, str) and author.strip():
        return author.strip()
    return domain_from_url(article.get("url"))


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

    return {
        "article_id": stable_article_id(url, title, published),
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
        start_date = format_timestamp(window_start)
        end_date = format_timestamp(window_end)

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


def fetch_currents(
    tickers: list[str],
    page_size: int = 20,
    from_date: str | None = None,
    to_date: str | None = None,
    chunk_days: int = 7,
    max_pages: int = 3,
    strict: bool = False,
) -> pd.DataFrame:
    """Fetch and normalize news from the Currents API."""
    api_key = os.getenv("CURRENTS_API_KEY")
    if not api_key:
        raise ValueError("CURRENTS_API_KEY environment variable not set")

    if page_size < 1:
        raise ValueError("page_size must be positive.")
    if chunk_days < 2:
        raise ValueError("chunk_days must be at least 2.")
    if max_pages < 1:
        raise ValueError("max_pages must be positive.")

    if from_date is not None and to_date is not None:
        windows: list[tuple[pd.Timestamp | None, pd.Timestamp | None]] = date_windows(
            utc_timestamp(from_date),
            utc_timestamp(to_date),
            chunk_days,
        )
    else:
        windows = [(None, None)]

    client = CurrentsNewsClient(api_key)
    ingested_at = pd.Timestamp.now(tz="UTC")
    return collect_news(
        tickers,
        lambda ticker: _fetch_ticker(
            client,
            ticker,
            windows,
            page_size,
            max_pages,
            ingested_at,
        ),
        provider_label="Currents API",
        strict=strict,
        to_date=to_date,
    )
