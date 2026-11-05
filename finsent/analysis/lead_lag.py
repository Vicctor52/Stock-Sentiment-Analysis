import pandas as pd


def lead_lag_correlation(
    features: pd.DataFrame,
    lags: tuple[int, ...] = (0, 1, 2),
    methods: tuple[str, ...] = ("pearson", "spearman"),
) -> pd.DataFrame:
    """Compute lead-lag correlations between sentiment and returns.

    Args:
        features: Aligned frame of sentiment and return series.
        lags: Lag offsets to evaluate, in trading days.
        methods: Correlation methods to apply.

    Returns:
        A frame with correlation coefficients for each lag and method.
    """
    raise NotImplementedError
