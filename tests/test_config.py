import pytest
from pydantic import ValidationError

from finsent.config import IngestionConfig


def _ingestion_config(**overrides: object) -> IngestionConfig:
    values: dict[str, object] = {
        "price_period": "12mo",
        "price_interval": "1d",
        "news_provider": "currents",
        "news_page_size": 20,
        "news_chunk_days": 7,
        "news_max_pages": 3,
        "bootstrap_days": 20,
        "news_overlap_sessions": 2,
    }
    values.update(overrides)
    return IngestionConfig(**values)


def test_ingestion_config_accepts_bootstrap_window() -> None:
    config = _ingestion_config()

    assert config.bootstrap_days == 20
    assert config.news_overlap_sessions == 2
    assert config.news_provider == "currents"
    assert config.news_page_size == 20
    assert config.news_chunk_days == 7
    assert config.news_max_pages == 3
    assert config.news_keyword_whitelist == {}
    assert config.news_keyword_exclude == {}
    assert config.gnews_max_articles == 10
    assert config.gnews_chunk_days == 7
    assert config.gnews_max_pages == 1
    assert config.gnews_language == "en"
    assert config.gnews_country is None


@pytest.mark.parametrize("bootstrap_days", [0, -1])
def test_ingestion_config_rejects_non_positive_bootstrap_days(bootstrap_days: int) -> None:
    with pytest.raises(ValidationError):
        _ingestion_config(bootstrap_days=bootstrap_days)


def test_ingestion_config_rejects_negative_news_overlap() -> None:
    with pytest.raises(ValidationError):
        _ingestion_config(news_overlap_sessions=-1)


@pytest.mark.parametrize("chunk_days", [0, 1])
def test_ingestion_config_rejects_too_small_news_chunk(chunk_days: int) -> None:
    with pytest.raises(ValidationError):
        _ingestion_config(news_chunk_days=chunk_days)


@pytest.mark.parametrize("max_pages", [0, -1])
def test_ingestion_config_rejects_non_positive_news_max_pages(max_pages: int) -> None:
    with pytest.raises(ValidationError):
        _ingestion_config(news_max_pages=max_pages)


def test_ingestion_config_accepts_gnews_provider() -> None:
    config = _ingestion_config(news_provider="gnews", gnews_country="us")

    assert config.news_provider == "gnews"
    assert config.gnews_country == "us"


def test_ingestion_config_accepts_news_keyword_filters() -> None:
    config = _ingestion_config(
        news_keyword_whitelist={"META": ["Meta Platforms", "Facebook"]},
        news_keyword_exclude={"META": ["metadata"]},
    )

    assert config.news_keyword_whitelist == {"META": ["Meta Platforms", "Facebook"]}
    assert config.news_keyword_exclude == {"META": ["metadata"]}


def test_ingestion_config_rejects_unknown_news_provider() -> None:
    with pytest.raises(ValidationError):
        _ingestion_config(news_provider="newsapi")


@pytest.mark.parametrize("max_articles", [0, 101])
def test_ingestion_config_rejects_invalid_gnews_article_limit(max_articles: int) -> None:
    with pytest.raises(ValidationError):
        _ingestion_config(gnews_max_articles=max_articles)
