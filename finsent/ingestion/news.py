import os

import pandas as pd

from finsent.logging import get_logger

logger = get_logger(__name__)


def fetch_news(tickers: list[str], page_size: int = 100) -> pd.DataFrame:
    """Fetch recent news articles for the given tickers.

    Args:
        tickers: Ticker symbols or search terms to query.
        page_size: Maximum number of articles to request per query.

    Returns:
        A normalized DataFrame with columns:
        article_id, source_name, published_utc, title, description,
        content, url, tickers, ingested_at.
    """
    api_key = os.getenv("NEWSAPI_KEY")
    if not api_key:
        raise ValueError("NEWSAPI_KEY environment variable not set")

    from newsapi import NewsApiClient

    newsapi = NewsApiClient(api_key=api_key)

    if not tickers:
        logger.warning("No tickers provided; returning empty DataFrame.")
        return pd.DataFrame(
            columns=[
                "article_id",
                "source_name",
                "published_utc",
                "title",
                "description",
                "content",
                "url",
                "tickers",
                "ingested_at",
            ]
        )

    query = " OR ".join(tickers)

    try:
        response = newsapi.get_everything(
            q=query,
            language="en",
            page_size=min(page_size, 100),
            sort_by="publishedAt",
            from_param=None,
        )
        articles = response.get("articles", [])
    except Exception as exc:
        logger.error("Error fetching news from NewsAPI: %s", exc)
        articles = []

    rows = []
    for art in articles:
        rows.append(
            {
                "article_id": art.get("id"),
                "source_name": art.get("source", {}).get("name"),
                "published_utc": art.get("publishedAt"),
                "title": art.get("title"),
                "description": art.get("description"),
                "content": art.get("content"),
                "url": art.get("url"),
                "tickers": tickers,
                "ingested_at": pd.Timestamp.now(tz="UTC"),
            }
        )

    df = pd.DataFrame(
        rows,
        columns=[  # type: ignore[list-item]
            "article_id",
            "source_name",
            "published_utc",
            "title",
            "description",
            "content",
            "url",
            "tickers",
            "ingested_at",
        ],
    )

    logger.info("Fetched %d news articles for tickers %s", len(df), tickers)
    return df
