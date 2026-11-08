from __future__ import annotations

import pandas as pd
import pytest

from finsent.ingestion import news
from finsent.storage.schemas import NEWS_ARTICLES_COLUMNS


class FakeNewsApiClient:
    """Minimal NewsAPI client stub used by the tests."""

    calls: list[str] = []

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    def get_everything(self, **kwargs: object) -> dict[str, list[dict[str, object]]]:
        query = str(kwargs["q"])
        self.calls.append(query)
        if query == "BAD":
            raise RuntimeError("boom")
        return {
            "articles": [
                {
                    "source": {"name": "Example"},
                    "publishedAt": "2024-01-02T21:30:00Z",
                    "title": f"{query} headline",
                    "description": "Description",
                    "content": "Content",
                    "url": f"https://example.com/{query}",
                },
                {
                    "source": {"name": "Example"},
                    "publishedAt": "2024-01-02T21:30:00Z",
                    "title": f"{query} duplicate",
                    "description": "Description",
                    "content": "Content",
                    "url": f"https://example.com/{query}",
                },
            ]
        }


def test_fetch_news_queries_each_ticker_and_deduplicates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    FakeNewsApiClient.calls.clear()
    monkeypatch.setenv("NEWSAPI_KEY", "test-key")
    monkeypatch.setattr("newsapi.NewsApiClient", FakeNewsApiClient)

    result = news.fetch_news(["NVDA", "AAPL"], page_size=10)

    assert FakeNewsApiClient.calls == ["NVDA", "AAPL"]
    assert list(result.columns) == NEWS_ARTICLES_COLUMNS
    assert set(result["ticker"]) == {"NVDA", "AAPL"}
    assert len(result) == 2
    assert result["article_id"].nunique() == 2
    assert pd.api.types.is_datetime64_any_dtype(result["published_utc"])
    assert result["published_utc"].dt.tz is not None


def test_fetch_news_article_id_is_stable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NEWSAPI_KEY", "test-key")
    monkeypatch.setattr("newsapi.NewsApiClient", FakeNewsApiClient)

    first = news.fetch_news(["NVDA"])
    second = news.fetch_news(["NVDA"])

    assert first["article_id"].iloc[0] == second["article_id"].iloc[0]


def test_fetch_news_continues_after_ticker_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NEWSAPI_KEY", "test-key")
    monkeypatch.setattr("newsapi.NewsApiClient", FakeNewsApiClient)

    result = news.fetch_news(["BAD", "NVDA"])

    assert result["ticker"].unique().tolist() == ["NVDA"]


def test_fetch_news_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NEWSAPI_KEY", raising=False)

    with pytest.raises(ValueError):
        news.fetch_news(["NVDA"])


def test_fetch_news_without_tickers_returns_empty_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NEWSAPI_KEY", "test-key")

    result = news.fetch_news([])

    assert result.empty
    assert list(result.columns) == NEWS_ARTICLES_COLUMNS
