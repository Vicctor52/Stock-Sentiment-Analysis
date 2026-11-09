from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def build_sentiment_chart(frame: pd.DataFrame) -> go.Figure:
    """Build a sentiment line chart with a fixed, padded y-axis."""
    data = frame.copy()
    data["session_date"] = pd.to_datetime(data["session_date"])
    data = data.sort_values("session_date")

    figure = px.line(
        data,
        x="session_date",
        y="sentiment_index",
        color="ticker",
        labels={
            "session_date": "Session",
            "sentiment_index": "Sentiment index",
            "ticker": "Ticker",
        },
    )
    figure.update_yaxes(
        range=[-1.1, 1.1],
        tickvals=[-1.0, -0.5, 0.0, 0.5, 1.0],
        zeroline=True,
        zerolinecolor="rgba(128, 128, 128, 0.45)",
        zerolinewidth=1,
    )
    figure.update_xaxes(title_text="Session")
    figure.update_layout(
        hovermode="x unified",
        legend_title_text="Ticker",
        margin={"l": 10, "r": 10, "t": 20, "b": 10},
    )
    return figure


def build_returns_chart(frame: pd.DataFrame) -> go.Figure:
    """Build a returns chart whose date range starts at the first valid return."""
    data = frame.dropna(subset=["log_return"]).copy()
    data["session_date"] = pd.to_datetime(data["session_date"])
    data = data.sort_values("session_date")

    figure = px.line(
        data,
        x="session_date",
        y="log_return",
        color="ticker",
        labels={
            "session_date": "Session",
            "log_return": "Log return",
            "ticker": "Ticker",
        },
    )
    if not data.empty:
        figure.update_xaxes(
            range=[data["session_date"].min(), data["session_date"].max()],
            title_text="Session",
        )
    figure.update_layout(
        hovermode="x unified",
        legend_title_text="Ticker",
        margin={"l": 10, "r": 10, "t": 20, "b": 10},
    )
    return figure
