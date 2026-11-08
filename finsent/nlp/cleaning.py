import hashlib
import html
import re

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_TRUNCATION_RE = re.compile(r"\[\+\d+ chars\]")
_WHITESPACE_RE = re.compile(r"\s+")
_ELLIPSIS_RE = re.compile(r"…")


def clean_text(text: str | None) -> str:
    """Normalize raw text for sentiment scoring.

    Args:
        text: Raw text to clean.

    Returns:
        The cleaned text with HTML, URLs, NewsAPI truncation markers, and
        repeated whitespace removed.
    """
    if not isinstance(text, str):
        return ""

    cleaned = html.unescape(text)
    cleaned = _HTML_TAG_RE.sub(" ", cleaned)
    cleaned = _URL_RE.sub(" ", cleaned)
    cleaned = _TRUNCATION_RE.sub(" ", cleaned)
    cleaned = _ELLIPSIS_RE.sub(" ", cleaned)
    cleaned = _WHITESPACE_RE.sub(" ", cleaned)
    return cleaned.strip()


def extract_tickers(text: str, universe: set[str]) -> list[str]:
    """Extract ticker mentions from text.

    Args:
        text: Text to inspect.
        universe: Ticker symbols considered valid.

    Returns:
        The tickers found in the text, in order of appearance.
    """
    raise NotImplementedError


def text_hash(text: str) -> str:
    """Compute a stable hash for deduplicating text.

    Args:
        text: Text to hash.

    Returns:
        A hexadecimal hash digest.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
