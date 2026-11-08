import numpy as np
import pandas as pd
from scipy import stats

from finsent.analysis.stats import adjust_pvalues, fisher_confidence_interval
from finsent.logging import get_logger
from finsent.storage.schemas import LEAD_LAG_COLUMNS

logger = get_logger(__name__)

_SUPPORTED_METHODS = {"pearson", "spearman"}


def _correlation(x: pd.Series, y: pd.Series, method: str) -> tuple[float, float]:
    """Calculate a correlation coefficient and p-value for two series."""
    if len(x) < 3 or x.nunique() < 2 or y.nunique() < 2:
        return float("nan"), float("nan")

    if method == "pearson":
        result = stats.pearsonr(x, y)
    elif method == "spearman":
        result = stats.spearmanr(x, y)
    else:
        raise ValueError(f"Unsupported correlation method: {method}")
    return float(result.statistic), float(result.pvalue)


def lead_lag_correlation(
    features: pd.DataFrame,
    lags: tuple[int, ...] = (0, 1, 2),
    methods: tuple[str, ...] = ("pearson", "spearman"),
    min_observations: int = 3,
    fdr_alpha: float = 0.05,
) -> pd.DataFrame:
    """Compute lead-lag correlations between sentiment and returns.

    Positive lags mean sentiment leads returns: ``lag=1`` correlates sentiment
    on session ``t`` with the return on session ``t+1``. Negative lags mean
    returns lead sentiment.

    Args:
        features: Aligned frame with ticker, session date, sentiment, and return
            columns.
        lags: Lag offsets to evaluate, in trading days.
        methods: Correlation methods to apply.
        min_observations: Minimum paired observations required per correlation.
        fdr_alpha: False discovery rate level for p-value adjustment.

    Returns:
        A frame with correlation coefficients for each lag and method.
    """
    invalid_methods = set(methods) - _SUPPORTED_METHODS
    if invalid_methods:
        raise ValueError(f"Unsupported correlation methods: {sorted(invalid_methods)}")
    if features.empty:
        return pd.DataFrame(columns=LEAD_LAG_COLUMNS)

    required = {"ticker", "session_date", "sentiment_index", "log_return"}
    missing = required - set(features.columns)
    if missing:
        raise KeyError(f"Missing columns for lead-lag analysis: {sorted(missing)}")

    frame = features.copy()
    if "source" not in frame.columns:
        frame["source"] = "news"
    frame["session_date"] = pd.to_datetime(frame["session_date"])
    frame = frame.sort_values(["ticker", "source", "session_date"]).reset_index(drop=True)

    rows: list[dict[str, object]] = []
    for (ticker, source), group in frame.groupby(["ticker", "source"], sort=False):
        for lag in lags:
            shifted_returns = group["log_return"].shift(-lag)
            paired = pd.DataFrame(
                {
                    "sentiment": group["sentiment_index"],
                    "returns": shifted_returns,
                }
            ).dropna()

            if len(paired) < min_observations:
                continue

            for method in methods:
                correlation, p_value = _correlation(paired["sentiment"], paired["returns"], method)
                if not np.isfinite(correlation):
                    continue
                ci_low, ci_high = fisher_confidence_interval(
                    correlation, len(paired), alpha=fdr_alpha
                )
                rows.append(
                    {
                        "ticker": ticker,
                        "source": source,
                        "method": method,
                        "lag": int(lag),
                        "r": correlation,
                        "p_value": p_value,
                        "adjusted_p": float("nan"),
                        "n_obs": len(paired),
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                    }
                )

    if not rows:
        return pd.DataFrame(columns=LEAD_LAG_COLUMNS)

    results = pd.DataFrame(rows, columns=LEAD_LAG_COLUMNS)
    results["adjusted_p"] = adjust_pvalues(results["p_value"].tolist(), alpha=fdr_alpha)
    return results.sort_values(["ticker", "source", "method", "lag"]).reset_index(drop=True)
