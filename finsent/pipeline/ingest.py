from pathlib import Path

from finsent.config import load_settings, load_tickers
from finsent.ingestion.news import fetch_news
from finsent.ingestion.prices import fetch_prices
from finsent.logging import get_logger
from finsent.storage.parquet_store import ParquetStore

logger = get_logger(__name__)


def run_ingestion() -> None:
    """Run the ingestion stage for prices and news.

    1. Loads pipeline settings and the ticker list.
    2. Fetches historical prices for the configured tickers.
    3. Persists prices to ``data/processed/prices.parquet``.
    4. Fetches recent news for each ticker.
    5. Persists news to ``data/processed/news.parquet``.

    News ingestion is skipped when ``NEWSAPI_KEY`` is not configured. Both
    tables are deduplicated by their primary keys via ``ParquetStore``.
    """
    settings = load_settings()
    tickers_config = load_tickers()
    tickers = tickers_config["tickers"]

    if not tickers:
        logger.warning("No tickers configured; skipping ingestion.")
        return

    store = ParquetStore(root=Path("data/processed"))

    period: str = settings.ingestion.price_period
    interval: str = settings.ingestion.price_interval

    logger.info(
        "Fetching prices for %d tickers (period=%s, interval=%s)",
        len(tickers),
        period,
        interval,
    )
    prices = fetch_prices(tickers, period=period, interval=interval)
    store.write("prices", prices)
    logger.info("Price ingestion complete. %d rows saved.", len(prices))

    try:
        news = fetch_news(tickers, page_size=settings.ingestion.news_page_size)
    except ValueError as exc:
        logger.warning("Skipping news ingestion: %s", exc)
        return

    if news.empty:
        logger.warning("No news articles fetched; skipping news write.")
        return

    store.write("news", news)
    logger.info("News ingestion complete. %d rows saved.", len(news))
