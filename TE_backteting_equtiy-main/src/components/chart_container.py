"""
chart_container.py

Interactive Plotly Financial Chart Builders for QUANTLAB.
Provides publication-grade dark-themed interactive charts with zoom, pan, hover tooltips,
and date range sliders.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

PLOTLY_DARK_LAYOUT = dict(
    paper_bgcolor="rgba(11, 15, 25, 0.0)",
    plot_bgcolor="rgba(15, 23, 42, 0.6)",
    font=dict(family="Inter, sans-serif", color="#94a3b8", size=11),
    margin=dict(l=40, r=30, t=50, b=40),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
        font=dict(color="#e2e8f0", size=11),
        bgcolor="rgba(15, 23, 42, 0.8)"
    ),
    xaxis=dict(
        gridcolor="rgba(255, 255, 255, 0.06)",
        zerolinecolor="rgba(255, 255, 255, 0.1)",
        showgrid=True,
        tickfont=dict(color="#94a3b8")
    ),
    yaxis=dict(
        gridcolor="rgba(255, 255, 255, 0.06)",
        zerolinecolor="rgba(255, 255, 255, 0.1)",
        showgrid=True,
        tickfont=dict(color="#94a3b8")
    ),
    hoverlabel=dict(
        bgcolor="#1e293b",
        font_size=12,
        font_family="Inter, sans-serif",
        font_color="#f8fafc"
    )
)


def render_equity_curve_plotly(
    portfolio_df: pd.DataFrame,
    benchmark_df: pd.DataFrame = None,
    ticker: str = "INFY",
    benchmark_symbol: str = "NIFTY50"
):
    """Render Strategy Equity Curve vs Benchmark Equity Curve interactive Plotly chart."""
    fig = go.Figure()

    dates = portfolio_df.index
    strat_val = portfolio_df["Portfolio Value"]

    fig.add_trace(go.Scatter(
        x=dates,
        y=strat_val,
        mode="lines",
        name=f"Strategy ({ticker})",
        line=dict(color="#10b981", width=2.2),
        fill="tozeroy",
        fillcolor="rgba(16, 185, 129, 0.08)",
        hovertemplate="<b>Date:</b> %{x|%Y-%m-%d}<br><b>Strategy Value:</b> ₹%{y:,.2f}<extra></extra>"
    ))

    if benchmark_df is not None:
        price_col = "Adj Close" if "Adj Close" in benchmark_df.columns else "Close"
        bench_aligned = benchmark_df.reindex(portfolio_df.index).ffill().bfill()

        init_cap = strat_val.iloc[0]
        bench_start = bench_aligned[price_col].iloc[0]
        bench_scaled = (bench_aligned[price_col] / bench_start) * init_cap

        fig.add_trace(go.Scatter(
            x=dates,
            y=bench_scaled,
            mode="lines",
            name=f"Benchmark ({benchmark_symbol})",
            line=dict(color="#3b82f6", width=1.8, dash="dash"),
            hovertemplate="<b>Date:</b> %{x|%Y-%m-%d}<br><b>Benchmark Value:</b> ₹%{y:,.2f}<extra></extra>"
        ))

    fig.add_hline(
        y=strat_val.iloc[0],
        line_dash="dot",
        line_color="#64748b",
        annotation_text="Initial Capital",
        annotation_position="bottom right"
    )

    layout = dict(PLOTLY_DARK_LAYOUT)
    layout["title"] = dict(text=f"<b>Equity Growth: Strategy ({ticker}) vs Benchmark ({benchmark_symbol})</b>", font=dict(color="#f8fafc", size=14))
    layout["yaxis"]["title"] = "Portfolio Value (INR)"
    layout["xaxis"]["rangeselector"] = dict(
        buttons=list([
            dict(count=1, label="1M", step="month", stepmode="backward"),
            dict(count=6, label="6M", step="month", stepmode="backward"),
            dict(count=1, label="1Y", step="year", stepmode="backward"),
            dict(step="all", label="All")
        ]),
        bgcolor="#1e293b",
        font=dict(color="#e2e8f0")
    )
    fig.update_layout(layout)

    st.plotly_chart(fig, use_container_width=True)


def render_drawdown_plotly(portfolio_df: pd.DataFrame):
    """Render Underwater Drawdown Plotly chart with worst drawdown highlight."""
    portfolio_values = portfolio_df["Portfolio Value"]
    running_max = portfolio_values.cummax()
    drawdowns = ((portfolio_values - running_max) / running_max) * 100.0

    worst_dd = drawdowns.min()
    worst_date = drawdowns.idxmin()

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=portfolio_df.index,
        y=drawdowns,
        mode="lines",
        name="Drawdown (%)",
        line=dict(color="#ef4444", width=1.5),
        fill="tozeroy",
        fillcolor="rgba(239, 68, 68, 0.25)",
        hovertemplate="<b>Date:</b> %{x|%Y-%m-%d}<br><b>Drawdown:</b> %{y:.2f}%<extra></extra>"
    ))

    fig.add_trace(go.Scatter(
        x=[worst_date],
        y=[worst_dd],
        mode="markers+text",
        name="Max Drawdown Peak",
        marker=dict(color="#dc2626", size=10, symbol="x"),
        text=[f"Max DD: {worst_dd:.2f}%"],
        textposition="bottom center",
        hovertemplate=f"<b>Max Drawdown:</b> {worst_dd:.2f}% on {worst_date.strftime('%Y-%m-%d')}<extra></extra>"
    ))

    layout = dict(PLOTLY_DARK_LAYOUT)
    layout["title"] = dict(text="<b>Underwater Drawdown Profile</b>", font=dict(color="#f8fafc", size=14))
    layout["yaxis"]["title"] = "Drawdown (%)"
    layout["yaxis"]["ticksuffix"] = "%"
    fig.update_layout(layout)

    st.plotly_chart(fig, use_container_width=True)


def render_signals_plotly(df_signals: pd.DataFrame, ticker: str = "INFY"):
    """Render Price chart with indicators and BUY/SELL scatter markers."""
    fig = go.Figure()

    price_col = "Adj Close" if "Adj Close" in df_signals.columns else "Close"

    fig.add_trace(go.Scatter(
        x=df_signals.index,
        y=df_signals[price_col],
        mode="lines",
        name=f"{ticker} Price",
        line=dict(color="#e2e8f0", width=1.5),
        hovertemplate="<b>Date:</b> %{x|%Y-%m-%d}<br><b>Price:</b> ₹%{y:,.2f}<extra></extra>"
    ))

    colors = ["#3b82f6", "#f59e0b", "#8b5cf6", "#ec4899"]
    c_idx = 0
    for col in df_signals.columns:
        if col.startswith("SMA") or col.startswith("EMA"):
            color = colors[c_idx % len(colors)]
            fig.add_trace(go.Scatter(
                x=df_signals.index,
                y=df_signals[col],
                mode="lines",
                name=col,
                line=dict(color=color, width=1.2, dash="dot" if "20" in col or "12" in col else "solid"),
                hovertemplate=f"<b>{col}:</b> ₹%{{y:,.2f}}<extra></extra>"
            ))
            c_idx += 1

    buy_df = df_signals[df_signals["Signal"] == 1]
    if not buy_df.empty:
        fig.add_trace(go.Scatter(
            x=buy_df.index,
            y=buy_df[price_col],
            mode="markers",
            name="BUY Signal",
            marker=dict(symbol="triangle-up", color="#10b981", size=12, line=dict(color="#047857", width=1)),
            hovertemplate="<b>BUY SIGNAL</b><br><b>Date:</b> %{x|%Y-%m-%d}<br><b>Execution Price:</b> ₹%{y:,.2f}<extra></extra>"
        ))

    sell_df = df_signals[df_signals["Signal"] == -1]
    if not sell_df.empty:
        fig.add_trace(go.Scatter(
            x=sell_df.index,
            y=sell_df[price_col],
            mode="markers",
            name="SELL Signal",
            marker=dict(symbol="triangle-down", color="#ef4444", size=12, line=dict(color="#b91c1c", width=1)),
            hovertemplate="<b>SELL SIGNAL</b><br><b>Date:</b> %{x|%Y-%m-%d}<br><b>Execution Price:</b> ₹%{y:,.2f}<extra></extra>"
        ))

    layout = dict(PLOTLY_DARK_LAYOUT)
    layout["title"] = dict(text=f"<b>Technical Strategy Signals — {ticker}</b>", font=dict(color="#f8fafc", size=14))
    layout["yaxis"]["title"] = "Price (INR)"
    fig.update_layout(layout)

    st.plotly_chart(fig, use_container_width=True)


def render_monthly_heatmap_plotly(portfolio_df: pd.DataFrame):
    """Render Monthly Return Heatmap (Years x Months Jan-Dec)."""
    monthly_val = portfolio_df["Portfolio Value"].resample("ME").last()
    monthly_ret = monthly_val.pct_change().dropna() * 100.0

    if monthly_ret.empty:
        st.info("Insufficient date range to calculate monthly heatmap.")
        return

    m_df = pd.DataFrame({"Year": monthly_ret.index.year, "Month": monthly_ret.index.month_name(), "Return": monthly_ret.values})

    month_order = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
    pivot = m_df.pivot(index="Year", columns="Month", values="Return")
    pivot = pivot.reindex(columns=[m for m in month_order if m in pivot.columns])

    fig = px.imshow(
        pivot,
        labels=dict(x="Month", y="Year", color="Return (%)"),
        x=pivot.columns,
        y=pivot.index,
        color_continuous_scale=[[0.0, "#ef4444"], [0.5, "#1e293b"], [1.0, "#10b981"]],
        color_continuous_midpoint=0.0,
        text_auto=".1f"
    )

    layout = dict(PLOTLY_DARK_LAYOUT)
    layout["title"] = dict(text="<b>Monthly Returns Breakdown (%)</b>", font=dict(color="#f8fafc", size=14))
    fig.update_layout(layout)

    st.plotly_chart(fig, use_container_width=True)


# Aliases for backward compatibility
render_monthly_returns_plotly = render_monthly_heatmap_plotly
