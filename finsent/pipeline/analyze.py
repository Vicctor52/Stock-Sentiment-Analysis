from pathlib import Path

from finsent.analysis.granger import granger_causality
from finsent.analysis.lead_lag import lead_lag_correlation
from finsent.analysis.returns import compute_log_returns
from finsent.config import load_settings
from finsent.logging import get_logger
from finsent.storage.parquet_store import ParquetStore

logger = get_logger(__name__)


def run_analysis() -> None:
    """Run the analysis stage over processed sentiment and prices.

    The stage writes ``returns.parquet``, ``lead_lag.parquet``, and
    ``granger.parquet`` under the processed data directory.
    """
    settings = load_settings()
    store = ParquetStore(root=Path(settings.storage.processed_dir))

    try:
        prices = store.read("prices")
        sentiment = store.read("daily_sentiment")
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "prices.parquet and daily_sentiment.parquet are required; run "
            "`python -m finsent ingest` and `python -m finsent score` first."
        ) from exc

    returns = compute_log_returns(prices)
    store.write("returns", returns, overwrite=True)

    features = sentiment.merge(
        returns,
        on=["ticker", "session_date"],
        how="inner",
        validate="many_to_one",
    )
    features = features.sort_values(["ticker", "source", "session_date"]).reset_index(drop=True)

    lead_lag = lead_lag_correlation(
        features,
        lags=tuple(settings.analysis.lags),
        min_observations=settings.analysis.min_observations,
        fdr_alpha=settings.analysis.fdr_alpha,
    )
    store.write("lead_lag", lead_lag, overwrite=True)

    granger = granger_causality(
        features,
        maxlag=settings.analysis.max_granger_lag,
        min_observations=settings.analysis.min_observations,
        fdr_alpha=settings.analysis.fdr_alpha,
    )
    store.write("granger", granger, overwrite=True)

    logger.info(
        "Analysis complete: %d returns, %d lead-lag rows, %d Granger rows.",
        len(returns),
        len(lead_lag),
        len(granger),
    )
