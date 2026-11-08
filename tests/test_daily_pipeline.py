from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from finsent.pipeline import daily


def test_daily_pipeline_processes_new_session_and_advances_checkpoint(monkeypatch, tmp_path):
    monkeypatch.setattr(
        daily,
        "load_settings",
        lambda: SimpleNamespace(
            storage=SimpleNamespace(processed_dir=tmp_path),
            market_calendar="NYSE",
        ),
    )
    calls = []
    monkeypatch.setattr(daily, "run_ingestion", lambda **kwargs: calls.append(kwargs))
    monkeypatch.setattr(daily, "run_scoring", lambda: calls.append("score"))
    monkeypatch.setattr(daily, "run_analysis", lambda: calls.append("analyze"))

    daily.run_daily_pipeline(pd.Timestamp("2024-01-03T22:00:00Z"))

    assert calls[0]["news_from"] == "2024-01-02T21:00:00+00:00"
    assert calls[0]["news_to"] == "2024-01-03T21:00:00+00:00"
    assert calls[0]["strict"] is True
    assert calls[1:] == ["score", "analyze"]
    assert (Path(tmp_path) / "pipeline_state.json").read_text(encoding="utf-8") == (
        '{\n  "last_processed_session": "2024-01-03"\n}\n'
    )


def test_daily_pipeline_skips_an_already_processed_session(monkeypatch, tmp_path):
    (tmp_path / "pipeline_state.json").write_text(
        '{"last_processed_session": "2024-01-03"}',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        daily,
        "load_settings",
        lambda: SimpleNamespace(
            storage=SimpleNamespace(processed_dir=tmp_path),
            market_calendar="NYSE",
        ),
    )
    calls = []
    monkeypatch.setattr(daily, "run_ingestion", lambda **kwargs: calls.append(kwargs))
    monkeypatch.setattr(daily, "run_scoring", lambda: calls.append("score"))
    monkeypatch.setattr(daily, "run_analysis", lambda: calls.append("analyze"))

    daily.run_daily_pipeline(pd.Timestamp("2024-01-03T22:00:00Z"))

    assert calls == []


def test_daily_pipeline_catches_up_after_weekend(monkeypatch, tmp_path):
    (tmp_path / "pipeline_state.json").write_text(
        '{"last_processed_session": "2024-01-05"}',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        daily,
        "load_settings",
        lambda: SimpleNamespace(
            storage=SimpleNamespace(processed_dir=tmp_path),
            market_calendar="NYSE",
        ),
    )
    calls = []
    monkeypatch.setattr(daily, "run_ingestion", lambda **kwargs: calls.append(kwargs))
    monkeypatch.setattr(daily, "run_scoring", lambda: None)
    monkeypatch.setattr(daily, "run_analysis", lambda: None)

    daily.run_daily_pipeline(pd.Timestamp("2024-01-08T22:00:00Z"))

    assert calls[0]["news_from"] == "2024-01-05T21:00:00+00:00"
    assert calls[0]["news_to"] == "2024-01-08T21:00:00+00:00"
    assert calls[0]["price_start"] == "2024-01-06"
    assert calls[0]["price_end"] == "2024-01-09"
    assert (
        (tmp_path / "pipeline_state.json").read_text(encoding="utf-8").endswith('"2024-01-08"\n}\n')
    )


def test_daily_pipeline_does_not_advance_checkpoint_on_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(
        daily,
        "load_settings",
        lambda: SimpleNamespace(
            storage=SimpleNamespace(processed_dir=tmp_path),
            market_calendar="NYSE",
        ),
    )
    monkeypatch.setattr(daily, "run_ingestion", lambda **kwargs: None)
    monkeypatch.setattr(daily, "run_scoring", lambda: None)

    def fail_analysis():
        raise RuntimeError("analysis failed")

    monkeypatch.setattr(daily, "run_analysis", fail_analysis)

    with pytest.raises(RuntimeError, match="analysis failed"):
        daily.run_daily_pipeline(pd.Timestamp("2024-01-03T22:00:00Z"))

    assert not (Path(tmp_path) / "pipeline_state.json").exists()
