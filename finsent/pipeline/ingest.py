from pathlib import Path

from finsent.config import load_settings, load_tickers
from finsent.ingestion.prices import fetch_prices
from finsent.storage.parquet_store import ParquetStore

logger = __import__("logging").getLogger(__name__)


def run_ingestion() -> None:
    """Run the ingestion stage for prices, news, and Reddit data.

    This step currently only handles price ingestion:
    1. Loads pipeline settings and ticker list.
    2. Fetches historical prices for the configured tickers.
    3. Persists the resulting DataFrame to ``data/processed/prices.parquet``
       via the ``ParquetStore`` (deduplication by ``["ticker", "session_date"]``).
    """
    settings = load_settings()
    tickers_config = load_tickers()
    tickers = tickers_config["tickers"]  # extract the list of ticker symbols

    if not tickers:
        logger.warning("No tickers configured; skipping price ingestion.")
        return

    period: str = settings.ingestion.price_period  # e.g. "12mo"
    interval: str = settings.ingestion.price_interval  # e.g. "1d"

    logger.info(
        "Fetching prices for %d tickers (period=%s, interval=%s)",
        len(tickers),
        period,
        interval,
    )

    frame = fetch_prices(tickers, period=period, interval=interval)

    store = ParquetStore(root=Path("data/processed"))
    store.write("prices", frame)

    logger.info("Ingestion complete. %d rows saved.", len(frame))
