import pandas as pd

from finsent.dashboard.charts import build_returns_chart, build_sentiment_chart


def test_build_sentiment_chart_uses_padded_fixed_y_axis() -> None:
    frame = pd.DataFrame(
        {
            "ticker": ["AAPL", "AAPL", "NVDA", "NVDA"],
            "session_date": pd.to_datetime(
                ["2026-11-05", "2026-11-06", "2026-11-05", "2026-11-06"]
            ),
            "sentiment_index": [1.0, 0.4, -1.0, 0.2],
        }
    )

    figure = build_sentiment_chart(frame)

    assert list(figure.layout.yaxis.range) == [-1.1, 1.1]
    assert list(figure.layout.yaxis.tickvals) == [-1.0, -0.5, 0.0, 0.5, 1.0]
    assert len(figure.data) == 2


def test_build_returns_chart_starts_at_first_valid_return() -> None:
    first_return = pd.Timestamp("2026-11-19")
    last_return = pd.Timestamp("2026-11-20")
    frame = pd.DataFrame(
        {
            "ticker": ["AAPL", "AAPL", "AAPL"],
            "session_date": pd.to_datetime(["2026-11-16", first_return, last_return]),
            "log_return": [None, 0.02, -0.01],
        }
    )

    figure = build_returns_chart(frame)

    assert pd.Timestamp(figure.layout.xaxis.range[0]) == first_return
    assert pd.Timestamp(figure.layout.xaxis.range[1]) == last_return
    assert pd.to_datetime(figure.data[0].x).tolist() == [first_return, last_return]
