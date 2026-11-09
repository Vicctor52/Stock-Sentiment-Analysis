import pytest
from pydantic import ValidationError

from finsent.config import IngestionConfig


def _ingestion_config(**overrides: object) -> IngestionConfig:
    values: dict[str, object] = {
        "price_period": "12mo",
        "price_interval": "1d",
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
    assert config.news_page_size == 20
    assert config.news_chunk_days == 7
    assert config.news_max_pages == 3


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
