import json
from pathlib import Path

import exchange_calendars as xcals
import pandas as pd

from finsent.config import load_settings
from finsent.logging import get_logger
from finsent.pipeline.analyze import run_analysis
from finsent.pipeline.ingest import run_ingestion
from finsent.pipeline.score import run_scoring

logger = get_logger(__name__)
_STATE_FILENAME = "pipeline_state.json"


def _session_date(value: object) -> pd.Timestamp:
    """Convert a calendar session label to a timezone-naive midnight timestamp."""
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert("UTC").tz_localize(None)
    return timestamp.normalize()


def _latest_closed_session(calendar, now: pd.Timestamp) -> pd.Timestamp:
    """Return the latest NYSE session whose close is no later than ``now``."""
    today = now.tz_convert("UTC").tz_localize(None).normalize()
    session = calendar.date_to_session(today, direction="previous")
    if calendar.session_close(session) > now:
        session = calendar.previous_session(session)
    return _session_date(session)


def _read_checkpoint(path: Path) -> pd.Timestamp | None:
    """Read the last successfully processed session, if a checkpoint exists."""
    if not path.exists():
        return None

    state = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(state, dict) or not isinstance(state.get("last_processed_session"), str):
        raise ValueError(f"Invalid daily pipeline state in {path}")
    return _session_date(state["last_processed_session"])


def _write_checkpoint(path: Path, session: pd.Timestamp) -> None:
    """Atomically persist the last successfully processed session."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(".tmp")
    temporary_path.write_text(
        json.dumps({"last_processed_session": session.date().isoformat()}, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def run_daily_pipeline(now: pd.Timestamp | None = None) -> None:
    """Collect and analyze data through the latest completed NYSE session."""
    settings = load_settings()
    processed_dir = Path(settings.storage.processed_dir)
    checkpoint_path = processed_dir / _STATE_FILENAME
    calendar = xcals.get_calendar(settings.market_calendar)
    current_time = now if now is not None else pd.Timestamp.now(tz="UTC")
    if current_time.tzinfo is None:
        raise ValueError("The daily pipeline run time must include a timezone.")
    current_time = current_time.tz_convert("UTC")

    target_session = _latest_closed_session(calendar, current_time)
    last_processed = _read_checkpoint(checkpoint_path)
    if last_processed is not None and last_processed >= target_session:
        logger.info("No completed NYSE session to process; latest is %s.", target_session.date())
        return

    start_session = (
        last_processed
        if last_processed is not None
        else _session_date(calendar.previous_session(target_session))
    )
    news_from = calendar.session_close(start_session).isoformat()
    news_to = calendar.session_close(target_session).isoformat()

    if last_processed is None:
        logger.info(
            "Bootstrapping prices; collecting news from %s through %s.",
            news_from,
            news_to,
        )
        run_ingestion(news_from=news_from, news_to=news_to, strict=True)
    else:
        price_start = (last_processed + pd.Timedelta(days=1)).date().isoformat()
        price_end = (target_session + pd.Timedelta(days=1)).date().isoformat()
        logger.info(
            "Processing NYSE sessions after %s through %s.",
            last_processed.date(),
            target_session.date(),
        )
        run_ingestion(
            news_from=news_from,
            news_to=news_to,
            price_start=price_start,
            price_end=price_end,
            strict=True,
        )

    run_scoring()
    run_analysis()
    _write_checkpoint(checkpoint_path, target_session)
    logger.info("Daily pipeline completed through %s.", target_session.date())
