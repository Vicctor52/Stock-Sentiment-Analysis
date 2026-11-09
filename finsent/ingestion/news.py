from __future__ import annotations

import pandas as pd

from finsent.ingestion.providers.base import empty_news_frame
from finsent.ingestion.providers.currents import fetch_currents
from finsent.ingestion.providers.gnews import fetch_gnews
from finsent.logging import get_logger

logger = get_logger(__name__)

_SUPPORTED_PROVIDERS = ("currents", "gnews")


def fetch_news(
    tickers: list[str],
    page_size: int = 20,
    from_date: str | None = None,
    to_date: str | None = None,
    chunk_days: int = 7,
    max_pages: int = 3,
    strict: bool = False,
    provider: str = "currents",
    gnews_max_articles: int = 10,
    gnews_chunk_days: int = 7,
    gnews_max_pages: int = 1,
    gnews_language: str = "en",
    gnews_country: str | None = None,
    keywords: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Fetch recent news articles for each ticker from a news provider.

    Args:
        tickers: Ticker symbols or search terms to query individually.
        page_size: Maximum number of articles per Currents request.
        from_date: Optional ISO timestamp lower bound.
        to_date: Optional exclusive ISO timestamp upper bound. Articles
            published exactly at this instant are excluded so news at a
            session close is assigned to the next session.
        chunk_days: Maximum inclusive calendar days per Currents request.
        max_pages: Maximum Currents pages fetched per ticker and date window.
        strict: Re-raise ticker-level errors so scheduled runs do not mark a
            partially collected interval as complete.
        provider: News provider to use. Supported values are ``currents``
            (default) and ``gnews``.
        gnews_max_articles: Articles per GNews request. The free plan caps this
            at 10 and paid plans allow up to 100.
        gnews_chunk_days: Maximum inclusive calendar days per GNews request.
        gnews_max_pages: Maximum GNews pages fetched per ticker and date window.
        gnews_language: GNews language code. Defaults to English.
        gnews_country: Optional GNews country code.
        keywords: Per‑ticker search query strings.  If provided, each ticker's
            query will be taken from this dict instead of being built from the
            ticker symbol alone.  This is useful when you want exact‑phrase matches
            or to exclude confounding terms such as ``metadata`` for the ticker ``META``.

    Returns:
        A normalized DataFrame with one row per article and ticker. Columns:
        article_id, ticker, source_name, published_utc, title, description,
        content, url, ingested_at.
    """
    provider_name = provider.strip().lower()
    if provider_name not in _SUPPORTED_PROVIDERS:
        supported = ", ".join(_SUPPORTED_PROVIDERS)
        raise ValueError(f"Unsupported news provider {provider!r}; choose {supported}.")

    if not tickers:
        logger.warning("No tickers provided; returning empty DataFrame.")
        return empty_news_frame()

    if (from_date is None) != (to_date is None):
        raise ValueError("Both from_date and to_date are required for a date range.")

    if provider_name == "gnews":
        return fetch_gnews(
            tickers,
            max_articles=gnews_max_articles,
            from_date=from_date,
            to_date=to_date,
            chunk_days=gnews_chunk_days,
            max_pages=gnews_max_pages,
            language=gnews_language,
            country=gnews_country,
            strict=strict,
            keywords=keywords or {},
        )

    return fetch_currents(
        tickers,
        page_size=page_size,
        from_date=from_date,
        to_date=to_date,
        chunk_days=chunk_days,
        max_pages=max_pages,
        strict=strict,
        keywords=keywords or {},
    )
