from dataclasses import dataclass

import pandas as pd

from finsent.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RedditClient:
    client_id: str
    client_secret: str
    user_agent: str

    def fetch_posts(
        self,
        subreddits: list[str],
        tickers: list[str],
        limit: int = 100,
    ) -> pd.DataFrame:
        raise NotImplementedError

    def fetch_comments(self, post_ids: list[str], limit: int = 50) -> pd.DataFrame:
        raise NotImplementedError
