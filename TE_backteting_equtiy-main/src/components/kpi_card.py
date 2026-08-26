"""
kpi_card.py

Reusable KPI & Metric Card Components for Institutional Financial UI.
Supports currency formatting (₹), percentage formatting (%), ratio precision,
hover tooltips, and strict semantic color highlights (Green/Red/Amber/Blue).
"""

import textwrap
import streamlit as st

def render_kpi_card(
    label: str,
    value: str,
    delta: str = None,
    delta_color: str = "normal", # "normal", "positive", "negative", "warning", "neutral"
    tooltip: str = None,
    icon: str = None
):
    """
    Render a single sleek financial KPI card.
    """
    tooltip_attr = f'title="{tooltip}"' if tooltip else ""
    
    color_map = {
        "positive": "#10b981",
        "negative": "#ef4444",
        "warning": "#f59e0b",
        "neutral": "#3b82f6",
        "normal": "#f8fafc"
    }
    accent_color = color_map.get(delta_color, "#f8fafc")
    
    delta_html = ""
    if delta:
        delta_html = f'<div style="font-size: 0.85rem; font-weight: 600; color: {accent_color}; margin-top: 4px;">{delta}</div>'

    icon_html = f"<span style='margin-right: 6px;'>{icon}</span>" if icon else ""

    card_html = textwrap.dedent(f"""
        <div {tooltip_attr} style="
            background: rgba(22, 28, 45, 0.85);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 0.75rem;
            padding: 1rem 1.2rem;
            backdrop-filter: blur(10px);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
            margin-bottom: 12px;
        ">
            <div style="
                color: #94a3b8;
                font-size: 0.78rem;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 0.06em;
                margin-bottom: 6px;
                display: flex;
                align-items: center;
                justify-content: space-between;
            ">
                <span>{icon_html}{label}</span>
                {f'<span style="cursor: pointer; opacity: 0.7;" title="{tooltip}">ⓘ</span>' if tooltip else ''}
            </div>
            <div style="
                font-size: 1.6rem;
                font-weight: 700;
                color: #f8fafc;
                letter-spacing: -0.02em;
                font-family: 'Inter', sans-serif;
            ">
                {value}
            </div>
            {delta_html}
        </div>
    """)
    st.markdown(card_html, unsafe_allow_html=True)


def render_kpi_grid(metrics_dict: dict, benchmark_comp: dict = None):
    """
    Render standard 6-card top performance summary grid.
    """
    col1, col2, col3, col4, col5, col6 = st.columns(6)

    init_cap = metrics_dict.get("Initial Capital", 100000.0)
    final_val = metrics_dict.get("Final Portfolio Value", 0.0)
    if final_val == 0.0 and "Total Return" in metrics_dict:
        final_val = init_cap * (1.0 + metrics_dict["Total Return"])

    tot_ret = metrics_dict.get("Total Return", 0.0) * 100.0
    cagr = metrics_dict.get("CAGR", 0.0) * 100.0
    sharpe = metrics_dict.get("Sharpe Ratio", 0.0)
    max_dd = metrics_dict.get("Maximum Drawdown", 0.0) * 100.0
    win_rate = metrics_dict.get("Win Rate", 0.0) * 100.0

    with col1:
        render_kpi_card(
            label="Initial Capital",
            value=f"₹{init_cap:,.0f}",
            tooltip="Starting portfolio cash value",
            icon="💰"
        )
    with col2:
        render_kpi_card(
            label="Final Portfolio",
            value=f"₹{final_val:,.0f}",
            delta=f"{tot_ret:+.2f}% Total PnL",
            delta_color="positive" if tot_ret >= 0 else "negative",
            tooltip="End-of-period portfolio value including cash & assets",
            icon="📈"
        )
    with col3:
        excess_str = f"{benchmark_comp.get('excess_return', 0.0)*100:+.2f}% vs Bench" if benchmark_comp else None
        render_kpi_card(
            label="CAGR",
            value=f"{cagr:.2f}%",
            delta=excess_str,
            delta_color="positive" if cagr >= 0 else "negative",
            tooltip="Compound Annual Growth Rate",
            icon="🚀"
        )
    with col4:
        bench_sharpe = benchmark_comp.get("benchmark_sharpe", 0.0) if benchmark_comp else 0.0
        render_kpi_card(
            label="Sharpe Ratio",
            value=f"{sharpe:.2f}",
            delta=f"Bench: {bench_sharpe:.2f}" if benchmark_comp else None,
            delta_color="positive" if sharpe >= 1.0 else ("warning" if sharpe >= 0 else "negative"),
            tooltip="Measures risk-adjusted return over risk-free rate",
            icon="⚖️"
        )
    with col5:
        render_kpi_card(
            label="Max Drawdown",
            value=f"{max_dd:.2f}%",
            delta="Worst peak-to-trough decline",
            delta_color="negative" if abs(max_dd) > 20 else "warning",
            tooltip="Maximum percentage loss from a historical peak",
            icon="📉"
        )
    with col6:
        render_kpi_card(
            label="Win Rate",
            value=f"{win_rate:.1f}%",
            delta=f"{metrics_dict.get('Number of Trades', 0)} Trades",
            delta_color="positive" if win_rate >= 50 else "neutral",
            tooltip="Percentage of closed trades with positive net PnL",
            icon="🎯"
        )
