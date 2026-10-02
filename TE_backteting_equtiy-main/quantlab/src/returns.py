"""Monthly and Yearly Return Logs and Histogram for QuantLab Backtesting Engine."""

from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def calculate_monthly_and_yearly_returns(daily_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Computes monthly return matrix and yearly returns log from the daily equity curve.
    Uses portfolio equity values, NOT sum of trade returns.
    """
    df = daily_df.copy()
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month

    years = sorted(df["year"].unique())
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    monthly_matrix_rows = []
    yearly_log_rows = []

    for y in years:
        year_df = df[df["year"] == y]
        year_start_equity = year_df["equity"].iloc[0]
        year_end_equity = year_df["equity"].iloc[-1]
        year_pnl = year_end_equity - year_start_equity
        year_return_pct = (year_pnl / year_start_equity) * 100.0 if year_start_equity > 0 else 0.0

        yearly_log_rows.append({
            "Year": y,
            "Starting Equity": round(year_start_equity, 2),
            "Ending Equity": round(year_end_equity, 2),
            "P&L": round(year_pnl, 2),
            "Return %": round(year_return_pct, 2)
        })

        month_row = {"Year": y}
        for m in range(1, 13):
            m_name = month_names[m - 1]
            month_df = year_df[year_df["month"] == m]
            if month_df.empty:
                month_row[m_name] = np.nan
            else:
                m_start_eq = month_df["equity"].iloc[0]
                m_end_eq = month_df["equity"].iloc[-1]
                m_ret = ((m_end_eq - m_start_eq) / m_start_eq) * 100.0 if m_start_eq > 0 else 0.0
                month_row[m_name] = round(m_ret, 2)

        month_row["Year"] = y
        month_row["Annual Return %"] = round(year_return_pct, 2)
        monthly_matrix_rows.append(month_row)

    monthly_df = pd.DataFrame(monthly_matrix_rows)
    # Ensure column ordering: Year, Jan ... Dec, Annual Return %
    cols = ["Year"] + month_names + ["Annual Return %"]
    monthly_df = monthly_df[cols]

    yearly_df = pd.DataFrame(yearly_log_rows)

    return monthly_df, yearly_df


def export_returns_reports(
    monthly_df: pd.DataFrame,
    yearly_df: pd.DataFrame,
    output_dir: str
) -> None:
    """Export output/monthly_returns.xlsx and output/yearly_returns.xlsx."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    monthly_df.to_excel(out_dir / "monthly_returns.xlsx", index=False, sheet_name="Monthly Returns")
    yearly_df.to_excel(out_dir / "yearly_returns.xlsx", index=False, sheet_name="Yearly Returns")


def plot_yearly_return_histogram(yearly_df: pd.DataFrame, output_dir: str) -> None:
    """
    Generate output/yearly_return_histogram.png clearly distinguishing
    positive (green) and negative (red) years with percentage labels.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)

    years = yearly_df["Year"].astype(str).tolist()
    returns = yearly_df["Return %"].tolist()
    colors = ["#10b981" if r >= 0 else "#ef4444" for r in returns]

    bars = ax.bar(years, returns, color=colors, width=0.6, edgecolor="#374151", linewidth=0.8)

    ax.axhline(0, color="#4b5563", linestyle="-", linewidth=1.0)
    ax.set_title("QuantLab Annual Return Histogram (%)", fontsize=15, fontweight="bold", pad=15)
    ax.set_xlabel("Year", fontsize=11, fontweight="bold")
    ax.set_ylabel("Annual Return %", fontsize=11, fontweight="bold")

    # Add labels on bars
    for bar, ret in zip(bars, returns):
        height = bar.get_height()
        va = "bottom" if height >= 0 else "top"
        offset = 0.5 if height >= 0 else -1.2
        ax.annotate(
            f"{ret:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, offset),
            textcoords="offset points",
            ha="center",
            va=va,
            fontsize=9,
            fontweight="bold",
            color="#065f46" if ret >= 0 else "#991b1b"
        )

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(out_dir / "yearly_return_histogram.png", dpi=300)
    plt.close(fig)
