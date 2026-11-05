def clean_text(text: str) -> str:
    """Normalize raw text for sentiment scoring.

    Args:
        text: Raw text to clean.

    Returns:
        The cleaned text.
    """
    raise NotImplementedError


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
    raise NotImplementedError
