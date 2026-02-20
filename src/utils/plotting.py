"""
Reusable plotting utilities (Plotly-based, dashboard-friendly).

All functions return a ``plotly.graph_objects.Figure`` so they work
seamlessly in both Streamlit and standalone HTML exports.
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


def plot_equity_curve(
    equity: pd.Series,
    title: str = "Equity Curve",
    benchmark: Optional[pd.Series] = None,
) -> go.Figure:
    """Plot cumulative NAV, optionally versus a benchmark."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=equity.index, y=equity.values,
        name="Strategy", line=dict(color="#2563eb", width=2),
    ))
    if benchmark is not None:
        bench_norm = benchmark / benchmark.iloc[0] * equity.iloc[0]
        fig.add_trace(go.Scatter(
            x=bench_norm.index, y=bench_norm.values,
            name="Benchmark", line=dict(color="#9ca3af", width=1.5, dash="dash"),
        ))
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        template="plotly_white",
        hovermode="x unified",
    )
    return fig


def plot_drawdown(returns: pd.Series, title: str = "Drawdown") -> go.Figure:
    """Plot the drawdown series as a filled area chart."""
    cumulative = (1 + returns).cumprod()
    rolling_peak = cumulative.cummax()
    drawdown = (cumulative - rolling_peak) / rolling_peak * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=drawdown.index, y=drawdown.values,
        fill="tozeroy",
        fillcolor="rgba(220,38,38,0.2)",
        line=dict(color="#dc2626", width=1),
        name="Drawdown",
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        template="plotly_white",
        hovermode="x unified",
    )
    return fig


def plot_spread_and_signals(
    spread: pd.Series,
    zscore: pd.Series,
    signals: pd.Series,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    title: str = "Spread Z-Score and Trading Signals",
) -> go.Figure:
    """Plot z-score with entry/exit bands and signal markers."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=("Raw Spread", "Z-Score + Signals"),
                        vertical_spacing=0.08)

    # --- Top: raw spread ---
    fig.add_trace(go.Scatter(
        x=spread.index, y=spread.values,
        name="Spread", line=dict(color="#6366f1", width=1.5),
    ), row=1, col=1)

    # --- Bottom: z-score ---
    fig.add_trace(go.Scatter(
        x=zscore.index, y=zscore.values,
        name="Z-Score", line=dict(color="#0ea5e9", width=1.5),
    ), row=2, col=1)

    # Bands
    for z, colour, label in [
        (entry_z, "rgba(220,38,38,0.15)", f"+{entry_z}σ"),
        (-entry_z, "rgba(220,38,38,0.15)", f"-{entry_z}σ"),
        (exit_z, "rgba(34,197,94,0.15)", f"+{exit_z}σ"),
        (-exit_z, "rgba(34,197,94,0.15)", f"-{exit_z}σ"),
    ]:
        fig.add_hline(y=z, line_dash="dot", line_color="grey",
                      annotation_text=label, row=2, col=1)

    # Entry / exit markers
    long_entries = signals[(signals == 1) & (signals.shift(1) != 1)].index
    short_entries = signals[(signals == -1) & (signals.shift(1) != -1)].index
    exits = signals[(signals == 0) & (signals.shift(1) != 0)].index

    for dates, symbol, colour, name in [
        (long_entries, "triangle-up", "#16a34a", "Long Entry"),
        (short_entries, "triangle-down", "#dc2626", "Short Entry"),
        (exits, "circle", "#9ca3af", "Exit"),
    ]:
        valid = [d for d in dates if d in zscore.index]
        if valid:
            fig.add_trace(go.Scatter(
                x=valid, y=zscore.reindex(valid).values,
                mode="markers",
                marker=dict(symbol=symbol, size=8, color=colour),
                name=name,
            ), row=2, col=1)

    fig.update_layout(title=title, template="plotly_white", hovermode="x unified", height=600)
    return fig


def plot_kalman_hedge_ratio(
    beta: pd.Series,
    title: str = "Kalman Filter: Time-Varying Hedge Ratio",
) -> go.Figure:
    """Plot the dynamic hedge ratio estimated by the Kalman filter."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=beta.index, y=beta.values,
        name="β (hedge ratio)",
        line=dict(color="#7c3aed", width=2),
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="β",
        template="plotly_white",
        hovermode="x unified",
    )
    return fig


def plot_correlation_heatmap(
    corr_matrix: pd.DataFrame,
    title: str = "Pair Return Correlations",
) -> go.Figure:
    """Portfolio heat map of pair-return correlations."""
    fig = px.imshow(
        corr_matrix,
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        title=title,
        text_auto=".2f",
    )
    fig.update_layout(template="plotly_white")
    return fig


def plot_return_distribution(
    returns: pd.Series,
    title: str = "Daily Return Distribution",
) -> go.Figure:
    """Histogram of daily returns with normal overlay."""
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=returns.values * 100,
        nbinsx=60,
        name="Daily Returns",
        histnorm="probability density",
        marker_color="#2563eb",
        opacity=0.7,
    ))
    # Normal fit
    mu, sigma = returns.mean() * 100, returns.std() * 100
    x_range = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 200)
    from scipy.stats import norm
    y_norm = norm.pdf(x_range, mu, sigma)
    fig.add_trace(go.Scatter(
        x=x_range, y=y_norm,
        name="Normal fit",
        line=dict(color="#dc2626", width=2, dash="dash"),
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Daily Return (%)",
        yaxis_title="Density",
        template="plotly_white",
    )
    return fig


def plot_performance_summary(metrics: dict) -> go.Figure:
    """Bar chart of key performance metrics."""
    labels = ["Sharpe", "Sortino", "Ann. Return %", "Max DD %", "Win Rate %"]
    values = [
        metrics.get("sharpe_ratio", 0),
        metrics.get("sortino_ratio", 0),
        metrics.get("annualised_return_pct", 0),
        abs(metrics.get("max_drawdown_pct", 0)),
        metrics.get("win_rate_pct", 0),
    ]
    colours = ["#2563eb", "#7c3aed", "#16a34a", "#dc2626", "#0ea5e9"]

    fig = go.Figure(go.Bar(
        x=labels, y=values,
        marker_color=colours,
        text=[f"{v:.2f}" for v in values],
        textposition="outside",
    ))
    fig.update_layout(
        title="Performance Summary",
        yaxis_title="Value",
        template="plotly_white",
    )
    return fig
