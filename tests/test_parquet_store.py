from pathlib import Path

import pandas as pd
import pytest

from finsent.storage.parquet_store import ParquetStore
from finsent.storage.schemas import NEWS_ARTICLES_COLUMNS, PRICES_COLUMNS


@pytest.fixture
def prices_df():
    """Sample DataFrame matching PRICES_COLUMNS."""
    return pd.DataFrame(
        {
            "ticker": ["AAPL", "MSFT", "GOOG"],
            "session_date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
            "open": [150.0, 250.0, 100.0],
            "high": [155.0, 255.0, 105.0],
            "low": [148.0, 252.0, 102.0],
            "close": [153.0, 253.0, 103.0],
            "adj_close": [152.5, 252.8, 104.0],
            "volume": [1_000_000, 2_000_000, 500_000],
            "source": ["yfinance", "yfinance", "yfinance"],
            "ingested_at": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
        }
    )


@pytest.fixture
def news_df():
    """Sample DataFrame matching NEWS_ARTICLES_COLUMNS."""
    return pd.DataFrame(
        {
            "article_id": ["a1", "a2", "a1"],
            "ticker": ["NVDA", "AAPL", "NVDA"],
            "source_name": ["Example", "Example", "Example"],
            "published_utc": pd.to_datetime(
                [
                    "2024-01-02T21:30:00Z",
                    "2024-01-02T21:30:00Z",
                    "2024-01-02T21:30:00Z",
                ]
            ),
            "title": ["NVDA headline", "AAPL headline", "NVDA duplicate"],
            "description": ["Description", "Description", "Description"],
            "content": ["Content", "Content", "Content"],
            "url": [
                "https://example.com/nvda",
                "https://example.com/aapl",
                "https://example.com/nvda",
            ],
            "ingested_at": pd.to_datetime(["2024-01-02T22:00:00Z"] * 3),
        }
    )


def _tmp_dir():
    import tempfile

    return Path(tempfile.mkdtemp())


class TestParquetStore:
    """Tests for ParquetStore class."""

    def test_write_new_table(self, prices_df):
        """Writing a new table should create the Parquet file."""
        root = _tmp_dir()
        store = ParquetStore(root=root)
        store.write("prices", prices_df)
        assert (root / "prices.parquet").is_file()

    def test_write_existing_table_deduplicates(self, prices_df):
        """Writing to an existing table should deduplicate by primary key."""
        root = _tmp_dir()
        store = ParquetStore(root=root)
        # First write
        store.write("prices", prices_df)
        # Second write with overlapping key + new row
        df2 = pd.DataFrame(
            {
                "ticker": ["AAPL", "MSFT", "AMZN"],
                "session_date": pd.to_datetime(["2024-01-02", "2024-01-05", "2024-01-06"]),
                "open": [151.0, 251.0, 110.0],
                "high": [156.0, 256.0, 115.0],
                "low": [149.0, 252.0, 112.0],
                "close": [154.0, 254.0, 113.0],
                "adj_close": [153.5, 253.5, 114.0],
                "volume": [1_100_000, 2_100_000, 550_000],
                "source": ["yfinance", "yfinance", "yfinance"],
                "ingested_at": pd.to_datetime(["2024-01-02", "2024-01-05", "2024-01-06"]),
            }
        )
        store.write("prices", df2)
        rows = store.read("prices")
        # AAPL 2024-01-02 should appear only once (last write keeps 151.0)
        aapl = rows[rows["ticker"] == "AAPL"]
        assert len(aapl) == 1
        # MSFT appears twice (no overlap in dates)
        msft = rows[rows["ticker"] == "MSFT"]
        assert len(msft) == 2
        # AMZN only from second write
        amzn = rows[rows["ticker"] == "AMZN"]
        assert len(amzn) == 1

    def test_write_missing_columns_fill_na(self):
        """If DataFrame is missing required columns, they should be filled with NaN."""
        root = _tmp_dir()
        store = ParquetStore(root=root)
        df = pd.DataFrame({"ticker": ["AAPL"], "session_date": pd.to_datetime(["2024-01-02"])})
        store.write("prices", df)
        rows = store.read("prices")
        assert len(rows) == 1
        assert pd.isna(rows["open"].iloc[0])
        assert pd.isna(rows["high"].iloc[0])

    def test_read_existing_table(self, prices_df):
        """Reading a table that was written should return the DataFrame."""
        root = _tmp_dir()
        store = ParquetStore(root=root)
        store.write("prices", prices_df)
        rows = store.read("prices")
        assert len(rows) == 3
        assert list(rows.columns) == PRICES_COLUMNS

    def test_read_nonexistent_raises(self):
        """Reading a table never written should raise FileNotFoundError."""
        root = _tmp_dir()
        store = ParquetStore(root=root)
        with pytest.raises(FileNotFoundError):
            store.read("nonexistent")

    def test_primary_keys_prices(self):
        """Primary key for 'prices' should be ['ticker', 'session_date']."""
        root = _tmp_dir()
        store = ParquetStore(root=root)
        keys = store._primary_keys("prices")
        assert keys == ["ticker", "session_date"]

    def test_primary_key_unknown_raises(self):
        """Asking for primary keys of unknown table should raise KeyError."""
        root = _tmp_dir()
        store = ParquetStore(root=root)
        with pytest.raises(KeyError):
            store._primary_keys("unknown_table")

    def test_write_and_read_news_deduplicates(self, news_df):
        """News should be deduplicated by article_id and ticker."""
        root = _tmp_dir()
        store = ParquetStore(root=root)

        store.write("news", news_df)
        rows = store.read("news")

        assert list(rows.columns) == NEWS_ARTICLES_COLUMNS
        assert len(rows) == 2
        nvda = rows[rows["ticker"] == "NVDA"]
        assert len(nvda) == 1
        assert nvda["title"].iloc[0] == "NVDA duplicate"

        update = pd.DataFrame(
            {
                "article_id": ["a1", "a3"],
                "ticker": ["NVDA", "TSLA"],
                "title": ["NVDA updated", "TSLA headline"],
            }
        )
        store.write("news", update)
        rows = store.read("news")

        assert len(rows) == 3
        nvda = rows[rows["ticker"] == "NVDA"]
        assert nvda["title"].iloc[0] == "NVDA updated"
        assert pd.isna(nvda["description"].iloc[0])

    def test_primary_keys_news(self):
        """Primary key for 'news' should be ['article_id', 'ticker']."""
        root = _tmp_dir()
        store = ParquetStore(root=root)
        keys = store._primary_keys("news")
        assert keys == ["article_id", "ticker"]

    def test_write_creates_parent_directories(self, prices_df):
        """Writing should create missing parent directories."""
        root = _tmp_dir() / "nested" / "processed"
        store = ParquetStore(root=root)

        store.write("prices", prices_df)

        assert (root / "prices.parquet").is_file()

    def test_write_does_not_mutate_input(self):
        """Writing should not add schema columns to the caller's DataFrame."""
        root = _tmp_dir()
        store = ParquetStore(root=root)
        frame = pd.DataFrame({"article_id": ["a1"], "ticker": ["NVDA"]})

        store.write("news", frame)

        assert list(frame.columns) == ["article_id", "ticker"]
