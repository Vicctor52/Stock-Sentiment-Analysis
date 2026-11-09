import pytest
from pydantic import ValidationError

from finsent.config import IngestionConfig


def _ingestion_config(**overrides: object) -> IngestionConfig:
    values: dict[str, object] = {
        "price_period": "12mo",
        "price_interval": "1d",
        "news_page_size": 100,
        "bootstrap_days": 20,
        "news_overlap_sessions": 2,
    }
    values.update(overrides)
    return IngestionConfig(**values)


def test_ingestion_config_accepts_bootstrap_window() -> None:
    config = _ingestion_config()

    assert config.bootstrap_days == 20
    assert config.news_overlap_sessions == 2


@pytest.mark.parametrize("bootstrap_days", [0, -1])
def test_ingestion_config_rejects_non_positive_bootstrap_days(bootstrap_days: int) -> None:
    with pytest.raises(ValidationError):
        _ingestion_config(bootstrap_days=bootstrap_days)


def test_ingestion_config_rejects_negative_news_overlap() -> None:
    with pytest.raises(ValidationError):
        _ingestion_config(news_overlap_sessions=-1)
