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

_GNEWS_SEARCH_URL = "https://gnews.io/api/v4/search"
_MIN_REQUEST_INTERVAL = 1.0
_MAX_ARTICLES = 100


class GNewsRateLimitError(NewsRateLimitError):
    """Raised when the GNews API quota is exhausted."""


class GNewsAPIError(NewsAPIError):
    """Raised for non-retryable GNews API errors."""


class GNewsClient:
    """Minimal client for the GNews search endpoint."""

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
        query: str,
        from_date: str | None = None,
        to_date: str | None = None,
        page_number: int = 1,
        max_articles: int = 10,
        language: str = "en",
        country: str | None = None,
    ) -> dict[str, object]:
        """Search GNews articles for a query and optional date range."""
        params: dict[str, object] = {
            "q": query,
            "lang": language,
            "max": max_articles,
            "page": page_number,
            "nullable": "description,content,image",
            "sortby": "publishedAt",
            "apikey": self.api_key,
        }
        if from_date is not None:
            params["from"] = from_date
        if to_date is not None:
            params["to"] = to_date
        if country is not None:
            params["country"] = country

        self._throttle()
        response = requests.get(
            _GNEWS_SEARCH_URL,
            params=params,
            timeout=self.timeout,
        )

        if response.status_code == 429:
            retry_after = response.headers.get("Retry-After", "unknown")
            raise GNewsRateLimitError(
                f"GNews API quota exhausted; retry after {retry_after} seconds or at 00:00 UTC."
            )
        if response.status_code >= 500:
            response.raise_for_status()
        if response.status_code >= 400:
            raise GNewsAPIError(
                f"GNews API request failed with status {response.status_code}: "
                f"{response.text[:200]}"
            )

        try:
            payload = response.json()
        except ValueError as exc:
            raise GNewsAPIError("GNews API returned invalid JSON.") from exc

        if not isinstance(payload, dict) or not isinstance(payload.get("articles"), list):
            raise GNewsAPIError(f"GNews API returned an error payload: {payload}")

        return payload


def _source_name(article: dict[str, object]) -> str:
    """Return a human-readable source label for a GNews article."""
    source = article.get("source")
    if isinstance(source, dict):
        name = source.get("name")
        if isinstance(name, str) and name.strip():
            return name.strip()
    return domain_from_url(article.get("url"))


def _normalize_article(
    article: dict[str, object],
    ticker: str,
    ingested_at: pd.Timestamp,
) -> dict[str, object] | None:
    """Normalize a GNews article to the internal news schema."""
    url = article.get("url") or ""
    title = article.get("title") or ""
    published = article.get("publishedAt")
    if not url and not title:
        return None

    return {
        "article_id": stable_article_id(url, title, published),
        "ticker": ticker,
        "source_name": _source_name(article),
        "published_utc": pd.to_datetime(published, utc=True, errors="coerce"),
        "title": title,
        "description": article.get("description") or "",
        "content": article.get("content") or "",
        "url": url,
        "ingested_at": ingested_at,
    }


def _fetch_ticker(
    client: GNewsClient,
    ticker: str,
    windows: list[tuple[pd.Timestamp | None, pd.Timestamp | None]],
    max_articles: int,
    max_pages: int,
    language: str,
    country: str | None,
    ingested_at: pd.Timestamp,
) -> list[dict[str, object]]:
    """Fetch and normalize articles for one ticker across date windows."""
    rows: list[dict[str, object]] = []
    for window_start, window_end in windows:
        start_date = format_timestamp(window_start)
        end_date = format_timestamp(window_end)

        for page in range(1, max_pages + 1):
            payload = client.search(
                query=ticker,
                from_date=start_date,
                to_date=end_date,
                page_number=page,
                max_articles=max_articles,
                language=language,
                country=country,
            )
            articles = payload.get("articles") or []
            if not isinstance(articles, list):
                raise GNewsAPIError("GNews API returned a non-list 'articles' field.")

            for article in articles:
                if isinstance(article, dict):
                    normalized = _normalize_article(article, ticker, ingested_at)
                    if normalized is not None:
                        rows.append(normalized)

            if len(articles) < max_articles:
                break
            total_articles = payload.get("totalArticles")
            if isinstance(total_articles, int) and page * max_articles >= total_articles:
                break

    return rows


def _warn_if_recent(to_date: str | None) -> None:
    """Warn when the GNews free-plan real-time delay may drop articles."""
    if to_date is None:
        return
    cutoff = utc_timestamp(to_date)
    lag = pd.Timestamp.now(tz="UTC") - cutoff
    if lag < pd.Timedelta(hours=12):
        logger.warning(
            "GNews free plan limits the 'to' parameter to at least 12 hours ago; "
            "articles after %s may be unavailable without a paid plan.",
            cutoff,
        )


def fetch_gnews(
    tickers: list[str],
    max_articles: int = 10,
    from_date: str | None = None,
    to_date: str | None = None,
    chunk_days: int = 7,
    max_pages: int = 1,
    language: str = "en",
    country: str | None = None,
    strict: bool = False,
) -> pd.DataFrame:
    """Fetch and normalize news from the GNews API."""
    api_key = os.getenv("GNEWS_API_KEY")
    if not api_key:
        raise ValueError("GNEWS_API_KEY environment variable not set")

    if not 1 <= max_articles <= _MAX_ARTICLES:
        raise ValueError(f"max_articles must be between 1 and {_MAX_ARTICLES}.")
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

    _warn_if_recent(to_date)

    client = GNewsClient(api_key)
    ingested_at = pd.Timestamp.now(tz="UTC")
    return collect_news(
        tickers,
        lambda ticker: _fetch_ticker(
            client,
            ticker,
            windows,
            max_articles,
            max_pages,
            language,
            country,
            ingested_at,
        ),
        provider_label="GNews API",
        strict=strict,
        to_date=to_date,
    )
