from pathlib import Path

from finsent.config import load_settings, load_tickers
from finsent.ingestion.news import fetch_news
from finsent.ingestion.prices import fetch_prices
from finsent.logging import get_logger
from finsent.storage.parquet_store import ParquetStore

logger = get_logger(__name__)


def run_ingestion(
    news_from: str | None = None,
    news_to: str | None = None,
    price_start: str | None = None,
    price_end: str | None = None,
    strict: bool = False,
) -> None:
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

    store = ParquetStore(root=Path(settings.storage.processed_dir))

    period: str = settings.ingestion.price_period
    interval: str = settings.ingestion.price_interval

    logger.info(
        "Fetching prices for %d tickers (period=%s, interval=%s)",
        len(tickers),
        period,
        interval,
    )
    if price_start is None:
        if strict:
            prices = fetch_prices(tickers, period=period, interval=interval, strict=True)
        else:
            prices = fetch_prices(tickers, period=period, interval=interval)
    else:
        prices = fetch_prices(
            tickers,
            period=period,
            interval=interval,
            start=price_start,
            end=price_end,
            strict=strict,
        )
    store.write("prices", prices)
    logger.info("Price ingestion complete. %d rows saved.", len(prices))

    try:
        if strict or news_from is not None:
            news = fetch_news(
                tickers,
                page_size=settings.ingestion.news_page_size,
                from_date=news_from,
                to_date=news_to,
                strict=strict,
            )
        else:
            news = fetch_news(tickers, page_size=settings.ingestion.news_page_size)
    except ValueError as exc:
        if strict:
            raise
        logger.warning("Skipping news ingestion: %s", exc)
        return

    if news.empty:
        logger.warning("No news articles fetched; skipping news write.")
        if strict:
            store.write("news", news)
        return

    store.write("news", news)
    logger.info("News ingestion complete. %d rows saved.", len(news))
