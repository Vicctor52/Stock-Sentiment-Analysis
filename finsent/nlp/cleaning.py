def clean_text(text: str) -> str:
    raise NotImplementedError


def extract_tickers(text: str, universe: set[str]) -> list[str]:
    raise NotImplementedError


def text_hash(text: str) -> str:
    raise NotImplementedError
