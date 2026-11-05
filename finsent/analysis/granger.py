import pandas as pd


def granger_causality(
    features: pd.DataFrame,
    maxlag: int = 5,
) -> pd.DataFrame:
    """Run Granger causality tests between sentiment and returns.

    Args:
        features: Aligned frame of sentiment and return series.
        maxlag: Maximum lag order to test.

    Returns:
        A frame with test statistics and p-values for each relationship.
    """
    raise NotImplementedError
