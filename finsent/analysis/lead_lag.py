import pandas as pd


def lead_lag_correlation(
    features: pd.DataFrame,
    lags: tuple[int, ...] = (0, 1, 2),
    methods: tuple[str, ...] = ("pearson", "spearman"),
) -> pd.DataFrame:
    raise NotImplementedError
