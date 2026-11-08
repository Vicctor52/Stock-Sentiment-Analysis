import pandas as pd

from finsent.dashboard.data import load_dashboard_data


def test_load_dashboard_data_reads_results_and_reports_missing(tmp_path):
    sentiment = pd.DataFrame(
        {
            "ticker": ["NVDA"],
            "session_date": ["2024-01-02"],
            "sentiment_index": [0.5],
        }
    )
    sentiment.to_parquet(tmp_path / "daily_sentiment.parquet", index=False)

    data = load_dashboard_data(tmp_path)

    assert data.daily_sentiment["sentiment_index"].tolist() == [0.5]
    assert data.returns.empty
    assert data.lead_lag.empty
    assert data.granger.empty
    assert data.missing_tables == ("returns", "lead_lag", "granger")


def test_load_dashboard_data_reports_all_missing_tables(tmp_path):
    data = load_dashboard_data(tmp_path)

    assert data.missing_tables == ("daily_sentiment", "returns", "lead_lag", "granger")
    assert data.daily_sentiment.empty
