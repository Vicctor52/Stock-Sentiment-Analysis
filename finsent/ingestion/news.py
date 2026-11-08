import hashlib
import os
from datetime import datetime

import pandas as pd
import yfinance as yf

from finsent.logging import get_logger
from finsent.storage.schemas import NEWS_ARTICLES_COLUMNS

logger = get_logger(__name__)


def _normalize_yfinance_article(article: dict, ticker: str) -> dict | None:
    """Normalize a yfinance news dict to the internal schema."""
    try:
        title = article.get("title") or ""
        publisher = article.get("publisher") or ""
        link = article.get("link") or ""
        # yfinance publishes roughly in the last 7 days; we'll set a generic UTC date
        published_at = datetime.utcnow().replace(tzinfo=None).isoformat() + "Z"
        # Use published_at as now for simplicity; could be improved
        raw_id = link or f"{title}|{published_at}"
        article_id = hashlib.sha256(raw_id.encode("utf-8")).hexdigest()
        return {
            "article_id": article_id,
            "ticker": ticker,
            "source_name": publisher,
            "published_utc": pd.to_datetime(published_at, utc=True, errors="coerce"),
            "title": title,
            "description": "",
            "content": "",
            "url": link,
            "ingested_at": pd.Timestamp.now(tz="UTC"),
        }
    except Exception as exc:
        logger.warning("Failed to normalize yfinance article: %s", exc)
        return None


def fetch_news(
    tickers: list[str],
    page_size: int = 100,
    from_date: str | None = None,
    to_date: str | None = None,
    use_yfinance: bool = False,
    strict: bool = False,
) -> pd.DataFrame:
    """Fetch recent news articles for each ticker.

    Args:
        tickers: Ticker symbols or search terms to query individually.
        page_size: Maximum number of articles to request per ticker (NewsAPI limit 100).
        from_date: Optional ISO date lower bound accepted by NewsAPI.
        to_date: Optional ISO date upper bound accepted by NewsAPI.
        use_yfinance: If ``True``, fall back to ``yfinance`` news when
            ``NEWSAPI_KEY`` is not set or to supplement the NewsAPI results.
        strict: Re-raise ticker-level errors so scheduled runs do not mark a
            partially collected interval as complete.

    Returns:
        A normalized DataFrame with one row per article and ticker. Columns:
        article_id, ticker, source_name, published_utc, title, description,
        content, url, ingested_at.
    """
    api_key = os.getenv("NEWSAPI_KEY")

    if not use_yfinance and not api_key:
        raise ValueError("NEWSAPI_KEY environment variable not set")

    # ------------------------------------------------------------------
    # 1) NewsAPI path (unchanged)
    # ------------------------------------------------------------------
    if not use_yfinance and api_key:
        from newsapi import NewsApiClient

        newsapi = NewsApiClient(api_key=api_key)

        if not tickers:
            logger.warning("No tickers provided; returning empty DataFrame.")
            return pd.DataFrame(columns=NEWS_ARTICLES_COLUMNS)

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
                page = 1
                while True:
                    response = newsapi.get_everything(**query_kwargs, page=page)
                    articles = response.get("articles", [])
                    for article in articles:
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
                                "published_utc": pd.to_datetime(
                                    published_at, utc=True, errors="coerce"
                                ),
                                "title": title,
                                "description": article.get("description"),
                                "content": article.get("content"),
                                "url": url,
                                "ingested_at": pd.Timestamp.now(tz="UTC"),
                            }
                        )

                    total_results = response.get("totalResults")
                    if len(articles) < min(page_size, 100) or (
                        total_results is not None and page * min(page_size, 100) >= total_results
                    ):
                        break
                    page += 1
            except Exception as exc:
                logger.error("Error fetching news for %s from NewsAPI: %s", ticker, exc)
                if strict:
                    raise RuntimeError(f"Failed to fetch news for {ticker}") from exc

        frame = pd.DataFrame(rows, columns=NEWS_ARTICLES_COLUMNS)
        if not frame.empty:
            frame = frame.drop_duplicates(subset=["article_id", "ticker"], keep="last")
        logger.info("Fetched %d news articles for %d tickers via NewsAPI", len(frame), len(tickers))
        return frame

    # ------------------------------------------------------------------
    # 2) yfinance fallback path
    # ------------------------------------------------------------------
    if not use_yfinance:
        # No API key and user didn't request yfinance – just return empty.
        logger.warning("No NEWSAPI_KEY set and use_yfinance=False; returning empty DataFrame.")
        return pd.DataFrame(columns=NEWS_ARTICLES_COLUMNS)

    logger.info("Fetching news via yfinance fallback for %d tickers", len(tickers))
    rows: list[dict[str, object]] = []

    for ticker in tickers:
        try:
            yf_ticker = yf.Ticker(ticker)
            # yfinance.news returns a list of dicts; each dict has title, publisher,
            # link, and a "content" field that may include a short summary.
            news_list = yf_ticker.news  # type: ignore[index]
        except Exception as exc:
            logger.error("Error fetching yfinance news for %s: %s", ticker, exc)
            continue

        for article in news_list:
            normalized = _normalize_yfinance_article(article, ticker)
            if normalized is None:
                continue
            # De‑duplicate by article_id + ticker (keep last occurrence)
            # We'll just append and deduplicate later.
            rows.append(normalized)

    if rows:
        frame = pd.DataFrame(rows, columns=NEWS_ARTICLES_COLUMNS)
    else:
        frame = pd.DataFrame(columns=NEWS_ARTICLES_COLUMNS)
    if not frame.empty:
        frame = frame.drop_duplicates(subset=["article_id", "ticker"], keep="last")
    logger.info("Fetched %d news articles for %d tickers via yfinance", len(frame), len(tickers))
    return frame
