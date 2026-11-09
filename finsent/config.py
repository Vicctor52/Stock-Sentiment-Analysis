from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    """Configuration for the sentiment model and inference settings."""

    name: str
    revision: str
    batch_size: int
    device: str


class IngestionConfig(BaseModel):
    """Configuration for external data ingestion."""

    price_period: str
    price_interval: str
    news_page_size: int
    bootstrap_days: int = Field(default=20, gt=0)
    news_overlap_sessions: int = Field(default=2, ge=0)


class AnalysisConfig(BaseModel):
    """Configuration for statistical analysis."""

    lags: list[int]
    min_observations: int
    max_granger_lag: int
    fdr_alpha: float


class StorageConfig(BaseModel):
    """Filesystem locations for pipeline data."""

    raw_dir: Path
    processed_dir: Path
    cache_dir: Path


class Settings(BaseModel):
    """Validated application settings loaded from YAML configuration."""

    timezone: str
    market_calendar: str
    model: ModelConfig
    ingestion: IngestionConfig
    analysis: AnalysisConfig
    storage: StorageConfig


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file into a dictionary.

    Args:
        path: Path to the YAML file.

    Returns:
        The parsed YAML content.
    """
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


@lru_cache
def load_settings(path: str = "config/settings.yaml") -> Settings:
    """Load and validate pipeline settings from a YAML file.

    Args:
        path: Path to the settings file.

    Returns:
        The validated pipeline settings. Results are cached by path.
    """
    return Settings.model_validate(load_yaml(Path(path)))


def load_tickers(path: str = "config/tickers.yaml") -> dict[str, list[str]]:
    """Load ticker configuration from a YAML file.

    Args:
        path: Path to the ticker configuration file.

    Returns:
        A mapping of configuration names to lists of values.
    """
    return load_yaml(Path(path))
