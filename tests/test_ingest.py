from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from finsent.pipeline import ingest
from finsent.storage.schemas import NEWS_ARTICLES_COLUMNS


class FakeStore:
    """Capture writes instead of touching Parquet files."""

    instances: list[FakeStore] = []

    def __init__(self, root) -> None:
        self.root = root
        self.writes: dict[str, pd.DataFrame] = {}
        FakeStore.instances.append(self)

    def write(self, table: str, frame: pd.DataFrame) -> None:
        self.writes[table] = frame


def _settings() -> SimpleNamespace:
    return SimpleNamespace(
        storage=SimpleNamespace(processed_dir="data/processed"),
        ingestion=SimpleNamespace(
            price_period="12mo",
            price_interval="1d",
            news_page_size=50,
        )
    )


def _prices() -> pd.DataFrame:
    return pd.DataFrame({"ticker": ["NVDA"], "session_date": ["2024-01-02"]})


def _news() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "article_id": ["a1"],
            "ticker": ["NVDA"],
        }
    )


def test_run_ingestion_writes_prices_and_news(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    FakeStore.instances.clear()
    price_calls = []
    news_calls = []

    def fake_prices(tickers, period, interval):
        price_calls.append((tickers, period, interval))
        return _prices()

    def fake_news(tickers, page_size):
        news_calls.append((tickers, page_size))
        return _news()

    monkeypatch.setattr(ingest, "load_settings", _settings)
    monkeypatch.setattr(ingest, "load_tickers", lambda: {"tickers": ["NVDA", "AAPL"]})
    monkeypatch.setattr(ingest, "ParquetStore", FakeStore)
    monkeypatch.setattr(ingest, "fetch_prices", fake_prices)
    monkeypatch.setattr(ingest, "fetch_news", fake_news)

    ingest.run_ingestion()

    assert price_calls == [(["NVDA", "AAPL"], "12mo", "1d")]
    assert news_calls == [(["NVDA", "AAPL"], 50)]
    store = FakeStore.instances[0]
    assert set(store.writes) == {"prices", "news"}


def test_run_ingestion_skips_news_without_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    FakeStore.instances.clear()

    def fake_news(tickers, page_size):
        raise ValueError("NEWSAPI_KEY environment variable not set")

    monkeypatch.setattr(ingest, "load_settings", _settings)
    monkeypatch.setattr(ingest, "load_tickers", lambda: {"tickers": ["NVDA"]})
    monkeypatch.setattr(ingest, "ParquetStore", FakeStore)
    monkeypatch.setattr(ingest, "fetch_prices", lambda *args, **kwargs: _prices())
    monkeypatch.setattr(ingest, "fetch_news", fake_news)

    ingest.run_ingestion()

    store = FakeStore.instances[0]
    assert set(store.writes) == {"prices"}


def test_run_ingestion_skips_empty_news(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    FakeStore.instances.clear()

    monkeypatch.setattr(ingest, "load_settings", _settings)
    monkeypatch.setattr(ingest, "load_tickers", lambda: {"tickers": ["NVDA"]})
    monkeypatch.setattr(ingest, "ParquetStore", FakeStore)
    monkeypatch.setattr(ingest, "fetch_prices", lambda *args, **kwargs: _prices())
    monkeypatch.setattr(
        ingest,
        "fetch_news",
        lambda *args, **kwargs: pd.DataFrame(columns=NEWS_ARTICLES_COLUMNS),
    )

    ingest.run_ingestion()

    store = FakeStore.instances[0]
    assert set(store.writes) == {"prices"}
