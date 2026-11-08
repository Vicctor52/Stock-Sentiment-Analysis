from pathlib import Path

import pandas as pd
import streamlit as st

from finsent.config import load_settings
from finsent.dashboard.data import DashboardData, load_dashboard_data


@st.cache_data(ttl=60, show_spinner=False)
def _cached_dashboard_data(processed_dir: str) -> DashboardData:
    """Load dashboard tables with Streamlit's short-lived data cache."""
    return load_dashboard_data(Path(processed_dir))


def _filter_ticker(frame: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Filter a results frame by ticker, or return all rows for the all-tickers option."""
    if ticker == "All":
        return frame
    return frame[frame["ticker"] == ticker]


def main() -> None:
    """Render the dashboard from the pipeline's processed analysis tables."""
    st.set_page_config(page_title="Financial Sentiment Lab", layout="wide")
    st.title("Financial Sentiment Lab")

    processed_dir = Path(load_settings().storage.processed_dir)
    if st.sidebar.button("Reload data"):
        _cached_dashboard_data.clear()
    data = _cached_dashboard_data(str(processed_dir))

    if data.missing_tables:
        missing = ", ".join(f"`{table}.parquet`" for table in data.missing_tables)
        st.warning(
            f"Missing results: {missing}. Run `python -m finsent analyze` "
            "after completing the ingestion and scoring stages."
        )

    available_tickers = sorted(
        set(data.daily_sentiment.get("ticker", pd.Series(dtype=str)).dropna())
        | set(data.returns.get("ticker", pd.Series(dtype=str)).dropna())
    )
    ticker = st.sidebar.selectbox("Ticker", ["All", *available_tickers])

    overview, lead_lag, granger = st.tabs(["Overview", "Lead-lag", "Granger"])

    with overview:
        sentiment = _filter_ticker(data.daily_sentiment, ticker).copy()
        returns = _filter_ticker(data.returns, ticker).copy()
        sentiment["session_date"] = pd.to_datetime(sentiment["session_date"])
        returns["session_date"] = pd.to_datetime(returns["session_date"])

        if not sentiment.empty:
            latest_date = sentiment["session_date"].max()
            latest_sentiment = sentiment.loc[
                sentiment["session_date"] == latest_date, "sentiment_index"
            ].mean()
            total_articles = int(sentiment["n_docs"].sum()) if "n_docs" in sentiment.columns else 0
            st.metric("Articles analyzed", f"{total_articles:,}")
            st.metric("Latest sentiment", f"{latest_sentiment:.3f}")
            st.caption(f"Latest sentiment session: {latest_date:%Y-%m-%d}")
            st.subheader("Daily sentiment index")
            sentiment_chart = sentiment.sort_values("session_date").set_index("session_date")
            st.line_chart(sentiment_chart, y="sentiment_index", color="ticker")
        elif "daily_sentiment" not in data.missing_tables:
            st.info("The daily sentiment file contains no observations.")

        if not returns.empty:
            st.subheader("Daily log returns")
            returns_chart = returns.sort_values("session_date").set_index("session_date")
            st.line_chart(returns_chart, y="log_return", color="ticker")
        elif "returns" not in data.missing_tables:
            st.info("The returns file contains no observations.")

    with lead_lag:
        results = _filter_ticker(data.lead_lag, ticker)
        if results.empty:
            st.info("No lead-lag results are available for the selected filters.")
        else:
            st.dataframe(
                results.sort_values(["adjusted_p", "p_value"], na_position="last"),
                use_container_width=True,
                hide_index=True,
            )

    with granger:
        results = _filter_ticker(data.granger, ticker)
        if results.empty:
            st.info("No Granger results are available for the selected filters.")
        else:
            st.dataframe(
                results.sort_values(["adjusted_p", "p_value"], na_position="last"),
                use_container_width=True,
                hide_index=True,
            )
