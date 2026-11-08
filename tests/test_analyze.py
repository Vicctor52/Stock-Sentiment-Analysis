from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from finsent.pipeline import analyze
from finsent.storage.parquet_store import ParquetStore
from finsent.storage.schemas import GRANGER_COLUMNS, LEAD_LAG_COLUMNS, RETURNS_COLUMNS


def _settings(tmp_path):
    return SimpleNamespace(
        storage=SimpleNamespace(processed_dir=tmp_path),
        analysis=SimpleNamespace(
            lags=[0, 1],
            min_observations=5,
            max_granger_lag=2,
            fdr_alpha=0.05,
        ),
    )


def _prices() -> pd.DataFrame:
    rng = np.random.default_rng(3)
    rows = []
    for ticker in ("NVDA", "AAPL"):
        returns = rng.normal(scale=0.01, size=30)
        prices = 100.0 * np.exp(np.cumsum(returns))
        for date, price in zip(
            pd.date_range("2024-01-02", periods=30, freq="B"), prices
        ):
            rows.append(
                {"ticker": ticker, "session_date": date, "adj_close": float(price)}
            )
    return pd.DataFrame(rows)


def _sentiment() -> pd.DataFrame:
    rng = np.random.default_rng(4)
    rows = []
    for ticker in ("NVDA", "AAPL"):
        for date, value in zip(
            pd.date_range("2024-01-02", periods=30, freq="B"),
            rng.normal(size=30),
        ):
            rows.append(
                {
                    "ticker": ticker,
                    "source": "news",
                    "session_date": date,
                    "sentiment_index": float(value),
                }
            )
    return pd.DataFrame(rows)


def test_run_analysis_writes_all_tables(monkeypatch, tmp_path):
    store = ParquetStore(root=tmp_path)
    store.write("prices", _prices(), overwrite=True)
    store.write("daily_sentiment", _sentiment(), overwrite=True)

    monkeypatch.setattr(analyze, "load_settings", lambda: _settings(tmp_path))

    analyze.run_analysis()

    store = ParquetStore(root=tmp_path)
    returns = store.read("returns")
    lead_lag = store.read("lead_lag")
    granger = store.read("granger")

    assert list(returns.columns) == RETURNS_COLUMNS
    assert len(returns) == 60
    assert list(lead_lag.columns) == LEAD_LAG_COLUMNS
    assert not lead_lag.empty
    assert list(granger.columns) == GRANGER_COLUMNS
    assert not granger.empty


def test_run_analysis_requires_inputs(monkeypatch, tmp_path):
    monkeypatch.setattr(analyze, "load_settings", lambda: _settings(tmp_path))

    with pytest.raises(FileNotFoundError):
        analyze.run_analysis()
