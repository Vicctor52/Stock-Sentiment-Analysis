from finsent import __version__
from finsent.nlp.aggregate import sentiment_index


def test_version_is_defined() -> None:
    assert __version__ == "0.1.0"


def test_sentiment_index_bounds() -> None:
    assert sentiment_index(10, 0, 0) == 1.0
    assert sentiment_index(0, 10, 0) == -1.0
    assert sentiment_index(5, 5, 0) == 0.0


def test_sentiment_index_empty() -> None:
    assert sentiment_index(0, 0, 0) == 0.0
