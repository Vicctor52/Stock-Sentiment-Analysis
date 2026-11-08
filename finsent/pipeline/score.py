from pathlib import Path

import pandas as pd

from finsent.analysis.alignment import assign_sessions
from finsent.config import load_settings
from finsent.logging import get_logger
from finsent.nlp.aggregate import aggregate_daily
from finsent.nlp.cleaning import clean_text, text_hash
from finsent.nlp.finbert import FinBertScorer
from finsent.storage.parquet_store import ParquetStore
from finsent.storage.schemas import SENTIMENT_COLUMNS

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

    available_articles = set(zip(news["article_id"], news["ticker"], strict=False))
    news = news.copy()
    news["text"] = news.apply(_combine_text, axis=1)
    news = news[news["text"].str.len() > 0].copy()
    news["text_hash"] = news["text"].map(text_hash)
    if not news.empty:
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
            logger.warning(
                "Dropping %d articles without a valid session date.", missing_sessions
            )
            news = news.dropna(subset=["session_date"])

    try:
        existing = store.read("news_sentiment")
    except FileNotFoundError:
        existing = pd.DataFrame(columns=SENTIMENT_COLUMNS)
    existing = existing.loc[
        [
            (doc_id, ticker) in available_articles
            for doc_id, ticker in zip(existing["doc_id"], existing["ticker"], strict=False)
        ]
    ].copy()

    existing_keys = set(zip(existing["doc_id"], existing["ticker"], strict=False))
    existing_texts = set(zip(existing["ticker"], existing["text_hash"], strict=False))
    new_news = news.loc[
        [
            (article_id, ticker) not in existing_keys
            and (ticker, digest) not in existing_texts
            for article_id, ticker, digest in zip(
                news["article_id"], news["ticker"], news["text_hash"], strict=False
            )
        ]
    ].copy()

    new_sentiment = pd.DataFrame(columns=SENTIMENT_COLUMNS)
    if not new_news.empty:
        scorer = FinBertScorer(
            model_name=settings.model.name,
            revision=settings.model.revision,
            device=settings.model.device,
            batch_size=settings.model.batch_size,
        )
        scores = scorer.score(new_news["text"].tolist())
        if len(scores) != len(new_news):
            raise RuntimeError(
                f"FinBERT returned {len(scores)} scores for {len(new_news)} texts."
            )

        scored_at = pd.Timestamp.now(tz="UTC")
        row_count = len(new_news)
        new_sentiment = pd.DataFrame(
            {
                "doc_id": new_news["article_id"].to_numpy(),
                "doc_type": ["news"] * row_count,
                "ticker": new_news["ticker"].to_numpy(),
                "published_utc": pd.to_datetime(new_news["published_utc"], utc=True),
                "session_date": new_news["session_date"],
                "text_hash": new_news["text_hash"].to_numpy(),
                "model_name": [settings.model.name] * row_count,
                "model_revision": [settings.model.revision] * row_count,
                "label": scores["label"].to_numpy(),
                "p_pos": scores["p_pos"].to_numpy(),
                "p_neg": scores["p_neg"].to_numpy(),
                "p_neu": scores["p_neu"].to_numpy(),
                "score": scores["score"].to_numpy(),
                "scored_at": [scored_at] * row_count,
            },
            columns=SENTIMENT_COLUMNS,
        )

    sentiment = pd.concat([existing, new_sentiment], ignore_index=True)
    sentiment = sentiment.drop_duplicates(subset=["doc_id", "ticker"], keep="last")
    store.write("news_sentiment", sentiment, overwrite=True)
    daily = aggregate_daily(sentiment)
    store.write("daily_sentiment", daily, overwrite=True)

    logger.info(
        "Scored %d new news articles across %d tickers; %d total scored.",
        len(new_sentiment),
        sentiment["ticker"].nunique(),
        len(sentiment),
    )
