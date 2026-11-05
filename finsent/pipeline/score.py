from finsent.logging import get_logger

logger = get_logger(__name__)


def run_scoring() -> None:
    """Run the sentiment scoring stage over ingested text."""
    raise NotImplementedError
