NEWS_ARTICLES_COLUMNS = [
    "article_id",
    "ticker",
    "source_name",
    "published_utc",
    "title",
    "description",
    "content",
    "url",
    "ingested_at",
]

PRICES_COLUMNS = [
    "ticker",
    "session_date",
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume",
    "source",
    "ingested_at",
]

SENTIMENT_COLUMNS = [
    "doc_id",
    "doc_type",
    "ticker",
    "published_utc",
    "session_date",
    "text_hash",
    "model_name",
    "model_revision",
    "label",
    "p_pos",
    "p_neg",
    "p_neu",
    "score",
    "scored_at",
]

DAILY_SENTIMENT_COLUMNS = [
    "ticker",
    "session_date",
    "source",
    "n_pos",
    "n_neg",
    "n_neu",
    "n_docs",
    "sentiment_index",
    "sentiment_expected",
]

RETURNS_COLUMNS = [
    "ticker",
    "session_date",
    "log_return",
    "fwd_return_1",
    "fwd_return_2",
]

LEAD_LAG_COLUMNS = [
    "ticker",
    "source",
    "method",
    "lag",
    "r",
    "p_value",
    "adjusted_p",
    "n_obs",
    "ci_low",
    "ci_high",
]

GRANGER_COLUMNS = [
    "ticker",
    "source",
    "maxlag",
    "lag",
    "test",
    "statistic",
    "p_value",
    "adjusted_p",
    "n_obs",
]
