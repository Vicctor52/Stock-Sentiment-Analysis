import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests


def adjust_pvalues(pvalues: list[float], alpha: float = 0.05) -> list[float]:
    """Adjust p-values with the Benjamini-Hochberg procedure.

    Args:
        pvalues: Raw p-values to adjust.
        alpha: False discovery rate level.

    Returns:
        Adjusted p-values in the same order as the input. Non-finite values
        remain ``NaN``.
    """
    values = np.asarray(pvalues, dtype=float)
    adjusted = np.full(values.shape, np.nan, dtype=float)
    mask = np.isfinite(values)
    if mask.any():
        adjusted[mask] = multipletests(values[mask], alpha=alpha, method="fdr_bh")[1]
    return adjusted.tolist()


def fisher_confidence_interval(
    correlation: float,
    n_observations: int,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Compute a Fisher z confidence interval for a correlation.

    Args:
        correlation: Correlation coefficient.
        n_observations: Number of paired observations.
        alpha: Significance level.

    Returns:
        Lower and upper confidence bounds. Bounds are ``NaN`` when they cannot
        be estimated.
    """
    if n_observations <= 3 or not np.isfinite(correlation):
        return float("nan"), float("nan")

    clipped = float(np.clip(correlation, -0.999999, 0.999999))
    standard_error = 1.0 / np.sqrt(n_observations - 3)
    critical = stats.norm.ppf(1.0 - alpha / 2.0)
    center = np.arctanh(clipped)
    return (
        float(np.tanh(center - critical * standard_error)),
        float(np.tanh(center + critical * standard_error)),
    )
