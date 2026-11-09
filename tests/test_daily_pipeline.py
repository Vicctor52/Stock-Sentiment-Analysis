from pathlib import Path
from types import SimpleNamespace

import exchange_calendars as xcals
import pandas as pd
import pytest

from finsent.pipeline import daily


def _settings(processed_dir: Path) -> SimpleNamespace:
    return SimpleNamespace(
        storage=SimpleNamespace(processed_dir=processed_dir),
        market_calendar="NYSE",
        ingestion=SimpleNamespace(bootstrap_days=20, news_overlap_sessions=2),
    )


def _patch_pipeline(monkeypatch, settings: SimpleNamespace, calls: list[object]) -> None:
    monkeypatch.setattr(daily, "load_settings", lambda: settings)
    monkeypatch.setattr(daily, "run_ingestion", lambda **kwargs: calls.append(kwargs))
    monkeypatch.setattr(daily, "run_scoring", lambda: calls.append("score"))
    monkeypatch.setattr(daily, "run_analysis", lambda: calls.append("analyze"))


def test_daily_pipeline_bootstraps_aligned_window(monkeypatch, tmp_path):
    calls: list[object] = []
    _patch_pipeline(monkeypatch, _settings(tmp_path), calls)

    daily.run_daily_pipeline(pd.Timestamp("2024-01-03T22:00:00Z"))

    assert calls[0] == {
        "news_from": "2023-12-14T21:00:00+00:00",
        "news_to": "2024-01-03T21:00:00+00:00",
        "price_start": "2023-12-14",
        "price_end": "2024-01-04",
        "strict": True,
    }
    assert calls[1:] == ["score", "analyze"]
    assert (Path(tmp_path) / "pipeline_state.json").read_text(encoding="utf-8") == (
        '{\n  "last_processed_session": "2024-01-03"\n}\n'
    )


def test_daily_pipeline_skips_an_already_processed_session(monkeypatch, tmp_path):
    (tmp_path / "pipeline_state.json").write_text(
        '{"last_processed_session": "2024-01-03"}',
        encoding="utf-8",
    )
    calls: list[object] = []
    _patch_pipeline(monkeypatch, _settings(tmp_path), calls)

    daily.run_daily_pipeline(pd.Timestamp("2024-01-03T22:00:00Z"))

    assert calls == []


def test_daily_pipeline_catches_up_after_weekend_with_news_overlap(monkeypatch, tmp_path):
    (tmp_path / "pipeline_state.json").write_text(
        '{"last_processed_session": "2024-01-05"}',
        encoding="utf-8",
    )
    calls: list[object] = []
    _patch_pipeline(monkeypatch, _settings(tmp_path), calls)

    daily.run_daily_pipeline(pd.Timestamp("2024-01-08T22:00:00Z"))

    assert calls[0] == {
        "news_from": "2024-01-03T21:00:00+00:00",
        "news_to": "2024-01-08T21:00:00+00:00",
        "price_start": "2024-01-06",
        "price_end": "2024-01-09",
        "strict": True,
    }
    assert (
        (tmp_path / "pipeline_state.json").read_text(encoding="utf-8").endswith('"2024-01-08"\n}\n')
    )


def test_daily_pipeline_processes_next_session_with_news_overlap(monkeypatch, tmp_path):
    (tmp_path / "pipeline_state.json").write_text(
        '{"last_processed_session": "2024-01-09"}',
        encoding="utf-8",
    )
    calls: list[object] = []
    _patch_pipeline(monkeypatch, _settings(tmp_path), calls)

    daily.run_daily_pipeline(pd.Timestamp("2024-01-10T22:00:00Z"))

    assert calls[0] == {
        "news_from": "2024-01-05T21:00:00+00:00",
        "news_to": "2024-01-10T21:00:00+00:00",
        "price_start": "2024-01-10",
        "price_end": "2024-01-11",
        "strict": True,
    }


def test_daily_pipeline_does_not_advance_checkpoint_on_failure(monkeypatch, tmp_path):
    calls: list[object] = []
    _patch_pipeline(monkeypatch, _settings(tmp_path), calls)

    def fail_analysis():
        raise RuntimeError("analysis failed")

    monkeypatch.setattr(daily, "run_analysis", fail_analysis)

    with pytest.raises(RuntimeError, match="analysis failed"):
        daily.run_daily_pipeline(pd.Timestamp("2024-01-03T22:00:00Z"))

    assert not (Path(tmp_path) / "pipeline_state.json").exists()


def test_bootstrap_window_uses_calendar_days_and_seed_session():
    calendar = xcals.get_calendar("NYSE")

    start_session, news_from, news_to, price_start, price_end = daily._bootstrap_window(
        calendar,
        pd.Timestamp("2024-01-31"),
        20,
    )

    assert start_session == pd.Timestamp("2024-01-12")
    assert news_from == "2024-01-11T21:00:00+00:00"
    assert news_to == "2024-01-31T21:00:00+00:00"
    assert price_start == "2024-01-11"
    assert price_end == "2024-02-01"


def test_overlap_session_moves_back_by_trading_sessions():
    calendar = xcals.get_calendar("NYSE")

    assert daily._overlap_session(calendar, pd.Timestamp("2024-01-09"), 0) == pd.Timestamp(
        "2024-01-09"
    )
    assert daily._overlap_session(calendar, pd.Timestamp("2024-01-09"), 2) == pd.Timestamp(
        "2024-01-05"
    )
