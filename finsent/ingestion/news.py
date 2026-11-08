import hashlib
import os

import pandas as pd

from finsent.logging import get_logger
from finsent.storage.schemas import NEWS_ARTICLES_COLUMNS

logger = get_logger(__name__)


def fetch_news(
    tickers: list[str],
    page_size: int = 100,
    from_date: str | None = None,
    to_date: str | None = None,
) -> pd.DataFrame:
    """Fetch recent news articles for each ticker.

    Args:
        tickers: Ticker symbols or search terms to query individually.
        page_size: Maximum number of articles to request per ticker.
        from_date: Optional ISO date lower bound accepted by NewsAPI.
        to_date: Optional ISO date upper bound accepted by NewsAPI.

    Returns:
        A normalized DataFrame with one row per article and ticker. Columns:
        article_id, ticker, source_name, published_utc, title, description,
        content, url, ingested_at.
    """
    api_key = os.getenv("NEWSAPI_KEY")
    if not api_key:
        raise ValueError("NEWSAPI_KEY environment variable not set")

    if not tickers:
        logger.warning("No tickers provided; returning empty DataFrame.")
        return pd.DataFrame(columns=NEWS_ARTICLES_COLUMNS)

    from newsapi import NewsApiClient

    newsapi = NewsApiClient(api_key=api_key)
    ingested_at = pd.Timestamp.now(tz="UTC")
    rows: list[dict[str, object]] = []

    for ticker in tickers:
        query_kwargs: dict[str, object] = {
            "q": ticker,
            "language": "en",
            "page_size": min(page_size, 100),
            "sort_by": "publishedAt",
        }
        if from_date:
            query_kwargs["from_param"] = from_date
        if to_date:
            query_kwargs["to"] = to_date

        try:
            response = newsapi.get_everything(**query_kwargs)
        except Exception as exc:
            logger.error("Error fetching news for %s from NewsAPI: %s", ticker, exc)
            continue

        for article in response.get("articles", []):
            url = article.get("url") or ""
            title = article.get("title") or ""
            published_at = article.get("publishedAt")
            if not url and not title:
                continue

            raw_id = url or f"{title}|{published_at}"
            rows.append(
                {
                    "article_id": hashlib.sha256(raw_id.encode("utf-8")).hexdigest(),
                    "ticker": ticker,
                    "source_name": (article.get("source") or {}).get("name"),
                    "published_utc": pd.to_datetime(published_at, utc=True, errors="coerce"),
                    "title": title,
                    "description": article.get("description"),
                    "content": article.get("content"),
                    "url": url,
                    "ingested_at": ingested_at,
                }
            )

    frame = pd.DataFrame(rows, columns=NEWS_ARTICLES_COLUMNS)
    if not frame.empty:
        frame = frame.drop_duplicates(subset=["article_id", "ticker"], keep="last")

    logger.info("Fetched %d news articles for %d tickers", len(frame), len(tickers))
    return frame
