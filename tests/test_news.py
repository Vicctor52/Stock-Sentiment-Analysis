from __future__ import annotations

import pandas as pd
import pytest
import requests

from finsent.ingestion import news
from finsent.ingestion.providers import currents, gnews
from finsent.storage.schemas import NEWS_ARTICLES_COLUMNS


def _article(
    title: str = "Headline",
    url: str = "https://example.com/article",
    published: str = "2024-01-02 21:30:00 +0000",
    **extra: object,
) -> dict[str, object]:
    article: dict[str, object] = {
        "title": title,
        "description": "Description",
        "url": url,
        "author": "Example",
        "published": published,
    }
    article.update(extra)
    return article


def _install_fake_client(monkeypatch: pytest.MonkeyPatch, handler) -> list[dict[str, object]]:
    calls: list[dict[str, object]] = []

    class FakeCurrentsClient:
        def __init__(self, api_key: str) -> None:
            self.api_key = api_key

        def search(self, **kwargs: object) -> dict[str, object]:
            calls.append(kwargs)
            return handler(kwargs, calls)

    monkeypatch.setattr(currents, "CurrentsNewsClient", FakeCurrentsClient)
    return calls


def _gnews_article(
    title: str = "GNews headline",
    url: str = "https://news.example.com/article",
    published: str = "2024-01-02T20:30:00Z",
    **extra: object,
) -> dict[str, object]:
    article: dict[str, object] = {
        "title": title,
        "description": "GNews description",
        "content": "GNews content",
        "url": url,
        "publishedAt": published,
        "source": {"name": "Example News", "url": "https://news.example.com"},
    }
    article.update(extra)
    return article


def _install_fake_gnews_client(
    monkeypatch: pytest.MonkeyPatch,
    handler,
) -> list[dict[str, object]]:
    calls: list[dict[str, object]] = []

    class FakeGNewsClient:
        def __init__(self, api_key: str) -> None:
            self.api_key = api_key

        def search(self, **kwargs: object) -> dict[str, object]:
            calls.append(kwargs)
            return handler(kwargs, calls)

    monkeypatch.setattr(gnews, "GNewsClient", FakeGNewsClient)
    return calls


def test_fetch_news_queries_each_ticker_and_deduplicates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CURRENTS_API_KEY", "test-key")

    def handler(kwargs, calls):
        query = str(kwargs["keywords"])
        return {
            "status": "ok",
            "news": [
                _article(title=f"{query} headline", url=f"https://example.com/{query}"),
                _article(title=f"{query} duplicate", url=f"https://example.com/{query}"),
            ],
            "page": 1,
        }

    calls = _install_fake_client(monkeypatch, handler)

    result = news.fetch_news(["NVDA", "AAPL"], page_size=10)

    assert [call["keywords"] for call in calls] == ["NVDA", "AAPL"]
    assert list(result.columns) == NEWS_ARTICLES_COLUMNS
    assert set(result["ticker"]) == {"NVDA", "AAPL"}
    assert len(result) == 2
    assert result["article_id"].nunique() == 2
    assert result["content"].eq("").all()
    assert pd.api.types.is_datetime64_any_dtype(result["published_utc"])
    assert result["published_utc"].dt.tz is not None


def test_fetch_news_article_id_is_stable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CURRENTS_API_KEY", "test-key")
    _install_fake_client(
        monkeypatch,
        lambda kwargs, calls: {"status": "ok", "news": [_article()], "page": 1},
    )

    first = news.fetch_news(["NVDA"])
    second = news.fetch_news(["NVDA"])

    assert first["article_id"].iloc[0] == second["article_id"].iloc[0]


def test_fetch_news_excludes_articles_at_or_after_to_date(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CURRENTS_API_KEY", "test-key")
    _install_fake_client(
        monkeypatch,
        lambda kwargs, calls: {
            "status": "ok",
            "news": [
                _article(
                    title="Before close",
                    url="https://example.com/before",
                    published="2024-01-02 20:59:59 +0000",
                ),
                _article(
                    title="At close",
                    url="https://example.com/at-close",
                    published="2024-01-02 21:00:00 +0000",
                ),
            ],
            "page": 1,
        },
    )

    result = news.fetch_news(
        ["NVDA"],
        from_date="2024-01-02T13:00:00+00:00",
        to_date="2024-01-02T21:00:00+00:00",
        page_size=10,
    )

    assert result["title"].tolist() == ["Before close"]
    assert result["published_utc"].iloc[0] < pd.Timestamp("2024-01-02T21:00:00Z")


def test_fetch_news_uses_domain_when_author_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CURRENTS_API_KEY", "test-key")
    article = _article(author=None, url="https://www.example.com/story")
    _install_fake_client(
        monkeypatch,
        lambda kwargs, calls: {"status": "ok", "news": [article], "page": 1},
    )

    result = news.fetch_news(["NVDA"])

    assert result["source_name"].iloc[0] == "example.com"


def test_fetch_news_continues_after_ticker_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CURRENTS_API_KEY", "test-key")

    def handler(kwargs, calls):
        if kwargs["keywords"] == "BAD":
            raise RuntimeError("boom")
        return {"status": "ok", "news": [_article()], "page": 1}

    _install_fake_client(monkeypatch, handler)

    result = news.fetch_news(["BAD", "NVDA"])

    assert result["ticker"].unique().tolist() == ["NVDA"]


def test_fetch_news_strict_raises_on_ticker_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CURRENTS_API_KEY", "test-key")

    def handler(kwargs, calls):
        raise RuntimeError("boom")

    _install_fake_client(monkeypatch, handler)

    with pytest.raises(RuntimeError, match="Failed to fetch news for BAD"):
        news.fetch_news(["BAD"], strict=True)


def test_fetch_news_rate_limit_always_aborts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CURRENTS_API_KEY", "test-key")

    def handler(kwargs, calls):
        raise currents.CurrentsRateLimitError("quota exhausted")

    _install_fake_client(monkeypatch, handler)

    with pytest.raises(currents.CurrentsRateLimitError):
        news.fetch_news(["NVDA"])


def test_fetch_news_chunks_date_range_into_week_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CURRENTS_API_KEY", "test-key")
    calls = _install_fake_client(
        monkeypatch,
        lambda kwargs, calls: {"status": "ok", "news": [], "page": 1},
    )

    result = news.fetch_news(
        ["NVDA"],
        page_size=20,
        from_date="2024-01-01T21:00:00+00:00",
        to_date="2024-01-21T21:00:00+00:00",
        chunk_days=7,
        max_pages=3,
    )

    assert result.empty
    assert len(calls) == 4
    assert calls[0]["start_date"] == "2024-01-01T21:00:00Z"
    assert calls[-1]["end_date"] == "2024-01-21T21:00:00Z"
    for call in calls:
        start = pd.Timestamp(call["start_date"])
        end = pd.Timestamp(call["end_date"])
        assert end >= start
        assert end - start <= pd.Timedelta(days=6)


def test_fetch_news_paginates_until_short_page(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CURRENTS_API_KEY", "test-key")

    def handler(kwargs, calls):
        page = int(kwargs["page_number"])
        if page == 1:
            articles = [
                _article(url="https://example.com/1"),
                _article(url="https://example.com/2"),
            ]
        else:
            articles = [_article(url="https://example.com/3")]
        return {"status": "ok", "news": articles, "page": page}

    calls = _install_fake_client(monkeypatch, handler)

    result = news.fetch_news(["NVDA"], page_size=2, max_pages=3)

    assert [call["page_number"] for call in calls] == [1, 2]
    assert len(result) == 3


def test_fetch_news_respects_max_pages(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CURRENTS_API_KEY", "test-key")

    def handler(kwargs, calls):
        page = int(kwargs["page_number"])
        articles = [_article(url=f"https://example.com/{page}-{index}") for index in range(2)]
        return {"status": "ok", "news": articles, "page": page}

    calls = _install_fake_client(monkeypatch, handler)

    result = news.fetch_news(["NVDA"], page_size=2, max_pages=2)

    assert [call["page_number"] for call in calls] == [1, 2]
    assert len(result) == 4


def test_fetch_news_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CURRENTS_API_KEY", raising=False)

    with pytest.raises(ValueError, match="CURRENTS_API_KEY"):
        news.fetch_news(["NVDA"])


def test_fetch_news_without_tickers_returns_empty_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CURRENTS_API_KEY", "test-key")

    result = news.fetch_news([])

    assert result.empty
    assert list(result.columns) == NEWS_ARTICLES_COLUMNS


def test_fetch_news_uses_gnews_when_requested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GNEWS_API_KEY", "test-key")
    calls = _install_fake_gnews_client(
        monkeypatch,
        lambda kwargs, calls: {
            "totalArticles": 1,
            "articles": [_gnews_article()],
        },
    )

    result = news.fetch_news(["NVDA"], provider="gnews")

    assert [call["query"] for call in calls] == ["NVDA"]
    assert calls[0]["language"] == "en"
    assert calls[0]["max_articles"] == 10
    assert calls[0]["country"] is None
    assert list(result.columns) == NEWS_ARTICLES_COLUMNS
    assert result["ticker"].tolist() == ["NVDA"]
    assert result["source_name"].tolist() == ["Example News"]
    assert result["content"].tolist() == ["GNews content"]
    assert result["description"].tolist() == ["GNews description"]
    assert pd.api.types.is_datetime64_any_dtype(result["published_utc"])
    assert result["published_utc"].dt.tz is not None


def test_fetch_news_gnews_excludes_articles_at_or_after_to_date(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GNEWS_API_KEY", "test-key")
    _install_fake_gnews_client(
        monkeypatch,
        lambda kwargs, calls: {
            "totalArticles": 2,
            "articles": [
                _gnews_article(
                    title="Before close",
                    url="https://news.example.com/before",
                    published="2024-01-02T20:59:59Z",
                ),
                _gnews_article(
                    title="At close",
                    url="https://news.example.com/at-close",
                    published="2024-01-02T21:00:00Z",
                ),
            ],
        },
    )

    result = news.fetch_news(
        ["NVDA"],
        provider="gnews",
        from_date="2024-01-02T13:00:00+00:00",
        to_date="2024-01-02T21:00:00+00:00",
    )

    assert result["title"].tolist() == ["Before close"]


def test_fetch_news_gnews_paginates_until_short_page(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GNEWS_API_KEY", "test-key")

    def handler(kwargs, calls):
        page = int(kwargs["page_number"])
        if page == 1:
            articles = [
                _gnews_article(url="https://news.example.com/1"),
                _gnews_article(url="https://news.example.com/2"),
            ]
        else:
            articles = [_gnews_article(url="https://news.example.com/3")]
        return {"totalArticles": 3, "articles": articles}

    calls = _install_fake_gnews_client(monkeypatch, handler)

    result = news.fetch_news(
        ["NVDA"],
        provider="gnews",
        gnews_max_articles=2,
        gnews_max_pages=2,
    )

    assert [call["page_number"] for call in calls] == [1, 2]
    assert len(result) == 3


def test_fetch_news_gnews_requires_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GNEWS_API_KEY", raising=False)

    with pytest.raises(ValueError, match="GNEWS_API_KEY"):
        news.fetch_news(["NVDA"], provider="gnews")


def test_fetch_news_rejects_unknown_provider() -> None:
    with pytest.raises(ValueError, match="Unsupported news provider"):
        news.fetch_news(["NVDA"], provider="unknown")


class FakeResponse:
    def __init__(
        self,
        status_code: int,
        payload: dict[str, object] | None = None,
        headers: dict[str, str] | None = None,
        text: str = "",
    ) -> None:
        self.status_code = status_code
        self.payload = payload or {}
        self.headers = headers or {}
        self.text = text

    def json(self) -> dict[str, object]:
        return self.payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(response=self)


def test_currents_client_sends_bearer_token_and_params(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_get(url, **kwargs):
        captured["url"] = url
        captured.update(kwargs)
        return FakeResponse(200, {"status": "ok", "news": [], "page": 1})

    monkeypatch.setattr(currents.requests, "get", fake_get)

    client = currents.CurrentsNewsClient("secret")
    payload = client.search(
        keywords="NVDA",
        start_date="2024-01-01T00:00:00Z",
        end_date="2024-01-07T00:00:00Z",
        page_number=2,
        page_size=20,
    )

    assert payload["status"] == "ok"
    assert captured["url"] == currents._CURRENTS_SEARCH_URL
    assert captured["headers"] == {"Authorization": "Bearer secret"}
    assert captured["params"] == {
        "keywords": "NVDA",
        "page_number": 2,
        "page_size": 20,
        "start_date": "2024-01-01T00:00:00Z",
        "end_date": "2024-01-07T00:00:00Z",
    }


def test_currents_client_reports_rate_limit(monkeypatch) -> None:
    monkeypatch.setattr(
        currents.requests,
        "get",
        lambda url, **kwargs: FakeResponse(429, headers={"Retry-After": "30"}),
    )

    with pytest.raises(currents.CurrentsRateLimitError, match="30"):
        currents.CurrentsNewsClient("secret").search(keywords="NVDA")


def test_currents_client_retries_burst_rate_limit(monkeypatch) -> None:
    responses = [
        FakeResponse(
            429,
            headers={
                "Retry-After": "1",
                "X-RateLimit-Remaining": "100",
                "X-RateLimit-Burst-Remaining": "0",
            },
        ),
        FakeResponse(200, {"status": "ok", "news": [], "page": 1}),
    ]

    sleeps: list[float] = []
    monkeypatch.setattr(currents.time, "sleep", sleeps.append)
    monkeypatch.setattr(currents.requests, "get", lambda url, **kwargs: responses.pop(0))

    payload = currents.CurrentsNewsClient("secret", min_interval=0).search(keywords="NVDA")

    assert payload["status"] == "ok"
    assert sleeps == [1.0]


def test_currents_client_reports_bad_request(monkeypatch) -> None:
    monkeypatch.setattr(
        currents.requests,
        "get",
        lambda url, **kwargs: FakeResponse(400, text="bad date range"),
    )

    with pytest.raises(currents.CurrentsAPIError, match="bad date range"):
        currents.CurrentsNewsClient("secret").search(keywords="NVDA")


def test_gnews_client_sends_api_key_and_params(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_get(url, **kwargs):
        captured["url"] = url
        captured.update(kwargs)
        return FakeResponse(200, {"totalArticles": 0, "articles": []})

    monkeypatch.setattr(gnews.requests, "get", fake_get)

    client = gnews.GNewsClient("secret")
    payload = client.search(
        query="NVDA",
        from_date="2024-01-01T00:00:00Z",
        to_date="2024-01-07T00:00:00Z",
        page_number=2,
        max_articles=5,
        language="en",
        country="us",
    )

    assert payload["articles"] == []
    assert captured["url"] == gnews._GNEWS_SEARCH_URL
    assert captured["params"] == {
        "q": "NVDA",
        "lang": "en",
        "max": 5,
        "page": 2,
        "nullable": "description,content,image",
        "sortby": "publishedAt",
        "apikey": "secret",
        "from": "2024-01-01T00:00:00Z",
        "to": "2024-01-07T00:00:00Z",
        "country": "us",
    }


def test_gnews_client_reports_rate_limit(monkeypatch) -> None:
    monkeypatch.setattr(
        gnews.requests,
        "get",
        lambda url, **kwargs: FakeResponse(429, headers={"Retry-After": "30"}),
    )

    with pytest.raises(gnews.GNewsRateLimitError, match="30"):
        gnews.GNewsClient("secret").search(query="NVDA")


def test_gnews_client_reports_bad_request(monkeypatch) -> None:
    monkeypatch.setattr(
        gnews.requests,
        "get",
        lambda url, **kwargs: FakeResponse(400, text="invalid api key"),
    )

    with pytest.raises(gnews.GNewsAPIError, match="invalid api key"):
        gnews.GNewsClient("secret").search(query="NVDA")
