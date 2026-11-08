from pathlib import Path

import pandas as pd

from finsent.analysis.alignment import assign_sessions
from finsent.config import load_settings
from finsent.logging import get_logger
from finsent.nlp.aggregate import aggregate_daily
from finsent.nlp.cleaning import clean_text, text_hash
from finsent.nlp.finbert import FinBertScorer
from finsent.storage.parquet_store import ParquetStore

logger = get_logger(__name__)


def _combine_text(row: pd.Series) -> str:
    """Combine title and description, falling back to content."""
    title = clean_text(row.get("title"))
    description = clean_text(row.get("description"))
    content = clean_text(row.get("content"))

    if not description:
        description = content

    if title and description:
        if title.lower() in description.lower():
            return description
        return f"{title}. {description}"
    return title or description


def run_scoring() -> None:
    """Run the sentiment scoring stage over ingested news.

    News is scored with FinBERT, then aggregated by ticker and NYSE session.
    The stage writes ``news_sentiment.parquet`` and ``daily_sentiment.parquet``.
    """
    settings = load_settings()
    store = ParquetStore(root=Path(settings.storage.processed_dir))

    try:
        news = store.read("news")
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "news.parquet not found; run `python -m finsent ingest` first."
        ) from exc

    if news.empty:
        logger.warning("No news articles found; skipping scoring.")
        return

    required = {
        "article_id",
        "ticker",
        "published_utc",
        "title",
        "description",
        "content",
    }
    missing = required - set(news.columns)
    if missing:
        raise KeyError(f"Missing columns in news table: {sorted(missing)}")

    news = news.copy()
    news["text"] = news.apply(_combine_text, axis=1)
    news = news[news["text"].str.len() > 0].copy()
    if news.empty:
        logger.warning("No non-empty news text found; skipping scoring.")
        return

    news["text_hash"] = news["text"].map(text_hash)
    before_dedup = len(news)
    news = news.drop_duplicates(subset=["ticker", "text_hash"], keep="first")
    dropped = before_dedup - len(news)
    if dropped:
        logger.info("Dropped %d duplicate news texts before scoring.", dropped)

    news = assign_sessions(
        news,
        timestamp_col="published_utc",
        calendar_name=settings.market_calendar,
    )
    missing_sessions = int(news["session_date"].isna().sum())
    if missing_sessions:
        logger.warning("Dropping %d articles without a valid session date.", missing_sessions)
        news = news.dropna(subset=["session_date"])
    if news.empty:
        logger.warning("No news articles with valid session dates; skipping scoring.")
        return

    scorer = FinBertScorer(
        model_name=settings.model.name,
        revision=settings.model.revision,
        device=settings.model.device,
        batch_size=settings.model.batch_size,
    )
    scores = scorer.score(news["text"].tolist())
    if len(scores) != len(news):
        raise RuntimeError(f"FinBERT returned {len(scores)} scores for {len(news)} texts.")

    scored_at = pd.Timestamp.now(tz="UTC")
    row_count = len(news)
    sentiment = pd.DataFrame(
        {
            "doc_id": news["article_id"].to_numpy(),
            "doc_type": ["news"] * row_count,
            "ticker": news["ticker"].to_numpy(),
            "published_utc": pd.to_datetime(news["published_utc"], utc=True),
            "session_date": news["session_date"],
            "text_hash": news["text_hash"].to_numpy(),
            "model_name": [settings.model.name] * row_count,
            "model_revision": [settings.model.revision] * row_count,
            "label": scores["label"].to_numpy(),
            "p_pos": scores["p_pos"].to_numpy(),
            "p_neg": scores["p_neg"].to_numpy(),
            "p_neu": scores["p_neu"].to_numpy(),
            "score": scores["score"].to_numpy(),
            "scored_at": [scored_at] * row_count,
        }
    )

    store.write("news_sentiment", sentiment, overwrite=True)
    daily = aggregate_daily(sentiment)
    store.write("daily_sentiment", daily, overwrite=True)

    label_counts = sentiment["label"].value_counts().to_dict()
    logger.info(
        "Scored %d news articles across %d tickers: %s",
        len(sentiment),
        sentiment["ticker"].nunique(),
        label_counts,
    )
