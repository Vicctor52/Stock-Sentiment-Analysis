from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import praw

from finsent.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RedditClient:
    """Client for fetching Reddit posts and comments.

    Attributes:
        client_id: Reddit API client identifier.
        client_secret: Reddit API client secret.
        user_agent: User agent string sent to Reddit.
    """

    client_id: str
    client_secret: str
    user_agent: str

    def __post_init__(self) -> None:
        self.reddit = praw.Reddit(
            client_id=self.client_id,
            client_secret=self.client_secret,
            user_agent=self.user_agent,
        )

    def fetch_posts(
        self,
        subreddits: list[str],
        tickers: list[str],
        limit: int = 100,
    ) -> pd.DataFrame:
        """Fetch recent Reddit posts that mention the given tickers.

        Args:
            subreddits: Subreddit names to search.
            tickers: Ticker symbols to filter by.
            limit: Maximum number of posts to fetch per subreddit.

        Returns:
            A normalized DataFrame with columns:
            post_id, subreddit, author_hash, created_utc, title, selftext,
            url, score, num_comments, tickers, ingested_at.
        """
        if not self.reddit.read_only:
            logger.warning("Reddit instance is not in read-only mode; this may cause issues.")

        queries: list[str] = []
        for sub in subreddits:
            for ticker in tickers:
                queries.append(f"{ticker} in:{sub}")

        if not queries:
            return pd.DataFrame(
                columns=[
                    "post_id",
                    "subreddit",
                    "author_hash",
                    "created_utc",
                    "title",
                    "selftext",
                    "url",
                    "score",
                    "num_comments",
                    "tickers",
                    "ingested_at",
                ]
            )

        fetched: list[dict] = []
        seen_ids: set[str] = set()

        for query in queries:
            try:
                for submission in self.reddit.subreddit(
                    subreddits[0] if subreddits else "all"
                ).search(query, limit=limit, sort="new", filter=["stickied"]):
                    if submission.id in seen_ids:
                        continue
                    seen_ids.add(submission.id)

                    # Collect tickers mentioned in the submission title or selftext
                    text = f"{submission.title} {submission.selftext}".lower()
                    mentioned = [t for t in tickers if t.lower() in text]

                    fetched.append(
                        {
                            "post_id": submission.id,
                            "subreddit": submission.subreddit.display_name,
                            "author_hash": submission.author.name if submission.author else "",
                            "created_utc": submission.created_utc,
                            "title": submission.title,
                            "selftext": submission.selftext,
                            "url": submission.url,
                            "score": submission.score,
                            "num_comments": submission.num_comments,
                            "tickers": mentioned if mentioned else tickers,
                            "ingested_at": pd.Timestamp.now(tz="UTC"),
                        }
                    )
            except Exception as exc:
                logger.error("Error fetching Reddit posts for query '%s': %s", query, exc)

        if not fetched:
            return pd.DataFrame(
                columns=[
                    "post_id",
                    "subreddit",
                    "author_hash",
                    "created_utc",
                    "title",
                    "selftext",
                    "url",
                    "score",
                    "num_comments",
                    "tickers",
                    "ingested_at",
                ]
            )

        df = pd.DataFrame(
            fetched,
            columns=[  # type: ignore[list-item]
                "post_id",
                "subreddit",
                "author_hash",
                "created_utc",
                "title",
                "selftext",
                "url",
                "score",
                "num_comments",
                "tickers",
                "ingested_at",
            ],
        )

        logger.info("Fetched %d Reddit posts for tickers %s", len(df), tickers)
        return df

    def fetch_comments(self, post_ids: list[str], limit: int = 50) -> pd.DataFrame:
        """Fetch comments for the given Reddit posts.

        Args:
            post_ids: Reddit post identifiers.
            limit: Maximum number of comments to fetch per post.

        Returns:
            A normalized DataFrame with columns:
            comment_id, post_id, author_hash, created_utc, body, score, ingested_at.
        """
        if not post_ids:
            return pd.DataFrame(
                columns=[
                    "comment_id",
                    "post_id",
                    "author_hash",
                    "created_utc",
                    "body",
                    "score",
                    "ingested_at",
                ]
            )

        fetched: list[dict] = []

        for pid in post_ids:
            try:
                submission = self.reddit.submission(id=pid)
                submission.comments.replace_more(limit=0)

                for comment in submission.comments.list()[:limit]:
                    fetched.append(
                        {
                            "comment_id": comment.id,
                            "post_id": comment.parent_id.split("_")[-1]
                            if comment.parent_id
                            else "",
                            "author_hash": comment.author.name if comment.author else "",
                            "created_utc": comment.created_utc,
                            "body": comment.body,
                            "score": comment.score,
                            "ingested_at": pd.Timestamp.now(tz="UTC"),
                        }
                    )
            except Exception as exc:
                logger.error("Error fetching Reddit comments for post '%s': %s", pid, exc)

        if not fetched:
            return pd.DataFrame(
                columns=[
                    "comment_id",
                    "post_id",
                    "author_hash",
                    "created_utc",
                    "body",
                    "score",
                    "ingested_at",
                ]
            )

        df = pd.DataFrame(
            fetched,
            columns=[  # type: ignore[list-item]
                "comment_id",
                "post_id",
                "author_hash",
                "created_utc",
                "body",
                "score",
                "ingested_at",
            ],
        )

        logger.info("Fetched %d Reddit comments for posts %s", len(df), post_ids)
        return df
