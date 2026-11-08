import contextlib
import sys
import types
from types import SimpleNamespace

import pytest

from finsent.nlp.finbert import FinBertScorer


class _FakeTensor:
    def __init__(self, data):
        self.data = data

    def to(self, device):
        return self

    def cpu(self):
        return self

    def __iter__(self):
        return iter(self.data)


class _FakeTokenizer:
    def __call__(self, batch, **kwargs):
        return {"input_ids": _FakeTensor([[1]] * len(batch))}


class _FakeLogits:
    def __init__(self, rows):
        self.rows = rows

    def cpu(self):
        return self

    def __iter__(self):
        return iter(self.rows)


class _FakeModel:
    def __call__(self, input_ids):
        rows = [[0.7, 0.2, 0.1] for _ in range(len(input_ids.data))]
        return SimpleNamespace(logits=_FakeLogits(rows))


def _install_fakes(monkeypatch, scorer: FinBertScorer) -> None:
    fake_torch = types.SimpleNamespace(
        softmax=lambda logits, dim: logits,
        inference_mode=lambda: contextlib.nullcontext(),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    def load_model():
        scorer._tokenizer = _FakeTokenizer()
        scorer._model = _FakeModel()
        scorer._labels = {0: "positive", 1: "negative", 2: "neutral"}

    monkeypatch.setattr(scorer, "_load_model", load_model)


def test_score_returns_expected_columns(monkeypatch) -> None:
    scorer = FinBertScorer()
    _install_fakes(monkeypatch, scorer)

    scores = scorer.score(["Nvidia beats estimates", "AMD rises"])

    assert list(scores.columns) == ["label", "p_pos", "p_neg", "p_neu", "score"]
    assert len(scores) == 2
    assert scores["label"].tolist() == ["positive", "positive"]
    assert scores["p_pos"].iloc[0] == pytest.approx(0.7)
    assert scores["p_neg"].iloc[0] == pytest.approx(0.2)
    assert scores["p_neu"].iloc[0] == pytest.approx(0.1)
    assert scores["score"].iloc[0] == pytest.approx(0.5)


def test_score_empty_input_does_not_load_model() -> None:
    scorer = FinBertScorer()

    scores = scorer.score([])

    assert scores.empty
    assert list(scores.columns) == ["label", "p_pos", "p_neg", "p_neu", "score"]
    assert scorer._model is None
