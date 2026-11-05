from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel


class ModelConfig(BaseModel):
    name: str
    revision: str
    batch_size: int
    device: str


class IngestionConfig(BaseModel):
    price_period: str
    price_interval: str
    reddit_comment_limit: int
    news_page_size: int


class AnalysisConfig(BaseModel):
    lags: list[int]
    min_observations: int
    max_granger_lag: int
    fdr_alpha: float


class StorageConfig(BaseModel):
    raw_dir: Path
    processed_dir: Path
    cache_dir: Path


class Settings(BaseModel):
    timezone: str
    market_calendar: str
    model: ModelConfig
    ingestion: IngestionConfig
    analysis: AnalysisConfig
    storage: StorageConfig


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


@lru_cache
def load_settings(path: str = "config/settings.yaml") -> Settings:
    return Settings.model_validate(load_yaml(Path(path)))


def load_tickers(path: str = "config/tickers.yaml") -> dict[str, list[str]]:
    return load_yaml(Path(path))
