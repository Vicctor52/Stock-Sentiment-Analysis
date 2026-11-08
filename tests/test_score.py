from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from finsent.pipeline import score
from finsent.storage.parquet_store import ParquetStore


class _FakeFinBert:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def score(self, texts):
        rows = []
        for text in texts:
            lower = text.lower()
            if "beats" in lower:
                rows.append(
                    {
                        "label": "positive",
                        "p_pos": 0.8,
                        "p_neg": 0.1,
                        "p_neu": 0.1,
                        "score": 0.7,
                    }
                )
            elif "falls" in lower or "weak" in lower:
                rows.append(
                    {
                        "label": "negative",
                        "p_pos": 0.1,
                        "p_neg": 0.8,
                        "p_neu": 0.1,
                        "score": -0.7,
                    }
                )
            else:
                rows.append(
                    {
                        "label": "neutral",
                        "p_pos": 0.1,
                        "p_neg": 0.1,
                        "p_neu": 0.8,
                        "score": 0.0,
                    }
                )
        return pd.DataFrame(rows)


def _settings(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        storage=SimpleNamespace(processed_dir=tmp_path),
        market_calendar="NYSE",
        model=SimpleNamespace(
            name="ProsusAI/finbert",
            revision="main",
            device="cpu",
            batch_size=8,
        ),
    )


def _news_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "article_id": ["a1", "a2", "a3", "a4"],
            "ticker": ["NVDA", "NVDA", "AAPL", "MSFT"],
            "published_utc": pd.to_datetime(
                [
                    "2024-01-02T15:00:00Z",
                    "2024-01-02T15:00:00Z",
                    "2024-01-02T15:00:00Z",
                    "2024-01-02T21:30:00Z",
                ],
                utc=True,
            ),
            "title": [
                "Nvidia beats",
                "Nvidia beats",
                "Apple falls",
                "Microsoft flat",
            ],
            "description": ["Strong demand", "Strong demand", "", "No change"],
            "content": ["", "", "Weak iPhone sales", ""],
        }
    )


def test_run_scoring_writes_deduplicated_sentiment(monkeypatch, tmp_path):
    store = ParquetStore(root=tmp_path)
    store.write("news", _news_frame(), overwrite=True)

    monkeypatch.setattr(score, "load_settings", lambda: _settings(tmp_path))
    monkeypatch.setattr(score, "FinBertScorer", _FakeFinBert)

    score.run_scoring()

    sentiment = ParquetStore(root=tmp_path).read("news_sentiment")
    assert len(sentiment) == 3
    assert set(sentiment["doc_id"]) == {"a1", "a3", "a4"}
    assert sentiment["model_name"].unique().tolist() == ["ProsusAI/finbert"]
    assert sentiment["model_revision"].unique().tolist() == ["main"]
    assert sentiment["label"].tolist() == ["positive", "negative", "neutral"]

    by_doc = sentiment.set_index("doc_id")
    assert by_doc.loc["a1", "session_date"] == pd.Timestamp("2024-01-02")
    assert by_doc.loc["a3", "session_date"] == pd.Timestamp("2024-01-02")
    assert by_doc.loc["a4", "session_date"] == pd.Timestamp("2024-01-03")
    assert by_doc.loc["a3", "score"] == pytest.approx(-0.7)

    daily = ParquetStore(root=tmp_path).read("daily_sentiment")
    assert len(daily) == 3
    nvda = daily[daily["ticker"] == "NVDA"].iloc[0]
    assert nvda["n_docs"] == 1
    assert nvda["sentiment_index"] == 1.0
    assert nvda["sentiment_expected"] == pytest.approx(0.7)


def test_run_scoring_replaces_previous_sentiment(monkeypatch, tmp_path):
    store = ParquetStore(root=tmp_path)
    store.write("news", _news_frame(), overwrite=True)
    store.write(
        "news_sentiment",
        pd.DataFrame(
            {
                "doc_id": ["stale"],
                "doc_type": ["news"],
                "ticker": ["TSLA"],
                "session_date": pd.to_datetime(["2024-01-02"]),
                "label": ["neutral"],
                "score": [0.0],
            }
        ),
        overwrite=True,
    )

    monkeypatch.setattr(score, "load_settings", lambda: _settings(tmp_path))
    monkeypatch.setattr(score, "FinBertScorer", _FakeFinBert)

    score.run_scoring()

    sentiment = ParquetStore(root=tmp_path).read("news_sentiment")
    assert "stale" not in set(sentiment["doc_id"])
    assert len(sentiment) == 3


def test_run_scoring_scores_only_new_articles(monkeypatch, tmp_path):
    class _TrackingFinBert(_FakeFinBert):
        scored_texts = []

        def score(self, texts):
            self.scored_texts.extend(texts)
            return super().score(texts)

    store = ParquetStore(root=tmp_path)
    store.write("news", _news_frame(), overwrite=True)
    monkeypatch.setattr(score, "load_settings", lambda: _settings(tmp_path))
    monkeypatch.setattr(score, "FinBertScorer", _TrackingFinBert)

    score.run_scoring()
    _TrackingFinBert.scored_texts.clear()
    store.write(
        "news",
        pd.DataFrame(
            {
                "article_id": ["a5"],
                "ticker": ["AMZN"],
                "published_utc": pd.to_datetime(["2024-01-03T15:00:00Z"], utc=True),
                "title": ["Amazon beats expectations"],
                "description": ["Strong results"],
                "content": [""],
            }
        ),
    )

    score.run_scoring()

    assert len(_TrackingFinBert.scored_texts) == 1
    sentiment = ParquetStore(root=tmp_path).read("news_sentiment")
    assert len(sentiment) == 4
    assert "a5" in set(sentiment["doc_id"])


def test_run_scoring_requires_news_file(monkeypatch, tmp_path):
    monkeypatch.setattr(score, "load_settings", lambda: _settings(tmp_path))

    with pytest.raises(FileNotFoundError):
        score.run_scoring()
