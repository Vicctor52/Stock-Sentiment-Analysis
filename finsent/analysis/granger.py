import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests

from finsent.analysis.stats import adjust_pvalues
from finsent.logging import get_logger
from finsent.storage.schemas import GRANGER_COLUMNS

logger = get_logger(__name__)


def granger_causality(
    features: pd.DataFrame,
    maxlag: int = 5,
    min_observations: int = 5,
    fdr_alpha: float = 0.05,
) -> pd.DataFrame:
    """Run Granger causality tests for sentiment causing returns.

    Args:
        features: Aligned frame with ticker, session date, sentiment, and return
            columns.
        maxlag: Maximum lag order to test.
        min_observations: Minimum observations required per ticker.
        fdr_alpha: False discovery rate level for p-value adjustment.

    Returns:
        A frame with test statistics and p-values for each relationship.
    """
    if maxlag < 1:
        raise ValueError("maxlag must be a positive integer.")
    if features.empty:
        return pd.DataFrame(columns=GRANGER_COLUMNS)

    required = {"ticker", "session_date", "sentiment_index", "log_return"}
    missing = required - set(features.columns)
    if missing:
        raise KeyError(f"Missing columns for Granger analysis: {sorted(missing)}")

    frame = features.copy()
    if "source" not in frame.columns:
        frame["source"] = "news"
    frame["session_date"] = pd.to_datetime(frame["session_date"])
    frame = frame.sort_values(["ticker", "source", "session_date"]).reset_index(drop=True)

    rows: list[dict[str, object]] = []
    for (ticker, source), group in frame.groupby(["ticker", "source"], sort=False):
        data = group[["log_return", "sentiment_index"]].dropna()
        if len(data) < min_observations:
            continue
        if len(data) <= 3 * maxlag + 1:
            logger.warning(
                "Skipping Granger test for %s/%s: %d observations are insufficient "
                "for maxlag=%d.",
                ticker,
                source,
                len(data),
                maxlag,
            )
            continue

        try:
            tests = grangercausalitytests(data.to_numpy(), maxlag=maxlag)
        except Exception as exc:
            logger.warning("Skipping Granger test for %s/%s: %s", ticker, source, exc)
            continue

        for lag in range(1, maxlag + 1):
            statistic, p_value, _, _ = tests[lag][0]["ssr_ftest"]
            rows.append(
                {
                    "ticker": ticker,
                    "source": source,
                    "maxlag": maxlag,
                    "lag": lag,
                    "test": "ssr_ftest",
                    "statistic": float(statistic),
                    "p_value": float(p_value),
                    "adjusted_p": float("nan"),
                    "n_obs": len(data),
                }
            )

    if not rows:
        return pd.DataFrame(columns=GRANGER_COLUMNS)

    results = pd.DataFrame(rows, columns=GRANGER_COLUMNS)
    results["adjusted_p"] = adjust_pvalues(
        results["p_value"].tolist(), alpha=fdr_alpha
    )
    return results.sort_values(
        ["ticker", "source", "maxlag", "lag", "test"]
    ).reset_index(drop=True)
