from dataclasses import dataclass

import pandas as pd

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

    def fetch_posts(
        self,
        subreddits: list[str],
        tickers: list[str],
        limit: int = 100,
    ) -> pd.DataFrame:
        """Fetch recent posts that mention the given tickers.

        Args:
            subreddits: Subreddit names to search.
            tickers: Ticker symbols to filter by.
            limit: Maximum number of posts to fetch per subreddit.

        Returns:
            A normalized frame of Reddit posts.
        """
        raise NotImplementedError

    def fetch_comments(self, post_ids: list[str], limit: int = 50) -> pd.DataFrame:
        """Fetch comments for the given Reddit posts.

        Args:
            post_ids: Reddit post identifiers.
            limit: Maximum number of comments to fetch per post.

        Returns:
            A normalized frame of Reddit comments.
        """
        raise NotImplementedError
