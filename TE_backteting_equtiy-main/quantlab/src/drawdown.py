"""Drawdown Engine and Curve Generation for QuantLab Backtesting Engine."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from .config import BacktestConfig


@dataclass
class DrawdownMetrics:
    """Detailed drawdown statistics."""
    max_drawdown: float
    max_drawdown_pct: float
    peak_date: str
    trough_date: str
    recovery_date: str
    max_drawdown_duration: int  # in days


def build_daily_equity_curve(
    trades: pd.DataFrame,
    config: BacktestConfig
) -> pd.DataFrame:
    """
    Build continuous daily equity series from earliest entry to latest exit date.
    Trades realize P&L on exit_date.
    """
    start_date = trades["entry_date"].min()
    end_date = trades["exit_date"].max()

    date_range = pd.date_range(start=start_date, end=end_date, freq="D")
    daily_df = pd.DataFrame({"date": date_range})

    # Group trade P&L by exit_date
    daily_pnl = trades.groupby("exit_date")["pnl"].sum().reset_index()
    daily_pnl = daily_pnl.rename(columns={"exit_date": "date", "pnl": "daily_pnl"})

    daily_df = daily_df.merge(daily_pnl, on="date", how="left")
    daily_df["daily_pnl"] = daily_df["daily_pnl"].fillna(0.0)

    daily_df["cumulative_pnl"] = daily_df["daily_pnl"].cumsum()
    daily_df["equity"] = config.initial_capital + daily_df["cumulative_pnl"]
    daily_df["cumulative_return_pct"] = (daily_df["cumulative_pnl"] / config.initial_capital) * 100.0

    # Drawdown series
    daily_df["running_peak"] = daily_df["equity"].cummax()
    daily_df["drawdown"] = daily_df["equity"] - daily_df["running_peak"]
    daily_df["drawdown_pct"] = (daily_df["drawdown"] / daily_df["running_peak"]) * 100.0

    return daily_df


def calculate_drawdown_metrics(daily_df: pd.DataFrame) -> DrawdownMetrics:
    """
    Computes maximum drawdown, peak date, trough date, recovery date, and duration.
    """
    if daily_df.empty:
        return DrawdownMetrics(0.0, 0.0, "N/A", "N/A", "N/A", 0)

    # Max drawdown trough is minimum value of drawdown
    trough_idx = daily_df["drawdown"].idxmin()
    max_dd = float(daily_df.loc[trough_idx, "drawdown"])  # negative or 0
    max_dd_pct = float(daily_df.loc[trough_idx, "drawdown_pct"])  # negative or 0
    trough_date = daily_df.loc[trough_idx, "date"]

    # Peak before trough: the running peak at trough_date
    peak_equity = daily_df.loc[trough_idx, "running_peak"]
    # Find the latest date before or on trough_date where equity equaled peak_equity
    prior_peak_candidates = daily_df.loc[:trough_idx]
    peak_idx = prior_peak_candidates[prior_peak_candidates["equity"] >= peak_equity - 1e-6].index[-1]
    peak_date = daily_df.loc[peak_idx, "date"]

    # Recovery date: first date after trough where equity reaches or exceeds peak_equity
    post_trough = daily_df.loc[trough_idx:]
    recovery_candidates = post_trough[post_trough["equity"] >= peak_equity]

    if not recovery_candidates.empty:
        recovery_date_val = recovery_candidates.iloc[0]["date"]
        recovery_str = recovery_date_val.strftime("%Y-%m-%d")
        duration_days = (recovery_date_val - peak_date).days
    else:
        recovery_str = "Not Recovered"
        duration_days = (daily_df["date"].iloc[-1] - peak_date).days

    return DrawdownMetrics(
        max_drawdown=max_dd,
        max_drawdown_pct=max_dd_pct,
        peak_date=peak_date.strftime("%Y-%m-%d"),
        trough_date=trough_date.strftime("%Y-%m-%d"),
        recovery_date=recovery_str,
        max_drawdown_duration=int(duration_days)
    )


def export_drawdown_data(daily_df: pd.DataFrame, output_dir: str) -> None:
    """Export equity_curve.csv and drawdown.csv."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # equity_curve.csv: Date, Equity, Cumulative P&L, Cumulative Return %
    eq_csv = daily_df[["date", "equity", "cumulative_pnl", "cumulative_return_pct"]].copy()
    eq_csv["date"] = eq_csv["date"].dt.strftime("%Y-%m-%d")
    eq_csv.rename(columns={
        "date": "Date",
        "equity": "Equity",
        "cumulative_pnl": "Cumulative P&L",
        "cumulative_return_pct": "Cumulative Return %"
    }).to_csv(out_dir / "equity_curve.csv", index=False)

    # drawdown.csv: Date, Equity, Running Peak, Drawdown, Drawdown %
    dd_csv = daily_df[["date", "equity", "running_peak", "drawdown", "drawdown_pct"]].copy()
    dd_csv["date"] = dd_csv["date"].dt.strftime("%Y-%m-%d")
    dd_csv.rename(columns={
        "date": "Date",
        "equity": "Equity",
        "running_peak": "Running Peak",
        "drawdown": "Drawdown",
        "drawdown_pct": "Drawdown %"
    }).to_csv(out_dir / "drawdown.csv", index=False)


def plot_equity_and_drawdown(
    daily_df: pd.DataFrame,
    metrics: DrawdownMetrics,
    output_dir: str
) -> None:
    """Generate high quality equity_curve.png and drawdown_curve.png."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dates = np.array(daily_df["date"].dt.to_pydatetime())

    # 1. Equity Curve Plot
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)

    ax.plot(dates, daily_df["equity"] / 1e7, color="#10b981", linewidth=2.0, label="Portfolio Equity (₹ Cr)")
    ax.fill_between(dates, daily_df["equity"] / 1e7, daily_df["equity"].iloc[0] / 1e7, color="#10b981", alpha=0.15)
    ax.axhline(daily_df["equity"].iloc[0] / 1e7, color="#6b7280", linestyle="--", linewidth=1.2, label="Initial Capital (₹1 Cr)")

    final_eq_cr = daily_df["equity"].iloc[-1] / 1e7
    ax.annotate(
        f"Final: ₹{final_eq_cr:.2f} Cr",
        xy=(dates[-1], final_eq_cr),
        xytext=(-80, 15),
        textcoords="offset points",
        arrowprops=dict(arrowstyle="->", color="#10b981"),
        fontweight="bold",
        color="#047857"
    )

    ax.set_title("QuantLab Portfolio Equity Curve", fontsize=15, fontweight="bold", pad=15)
    ax.set_xlabel("Date", fontsize=11, fontweight="bold")
    ax.set_ylabel("Equity (₹ Crores)", fontsize=11, fontweight="bold")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    plt.xticks(rotation=25)
    ax.legend(loc="upper left", frameon=True)
    plt.tight_layout()
    plt.savefig(out_dir / "equity_curve.png", dpi=300)
    plt.close(fig)

    # 2. Drawdown Curve Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), dpi=300, sharex=True)

    # Drawdown in Currency
    ax1.plot(dates, daily_df["drawdown"], color="#ef4444", linewidth=1.5, label="Drawdown (₹)")
    ax1.fill_between(dates, daily_df["drawdown"], 0, color="#ef4444", alpha=0.25)
    ax1.set_title("QuantLab Drawdown (Currency)", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Drawdown (₹)", fontsize=10, fontweight="bold")
    ax1.axhline(0, color="#9ca3af", linestyle="-", linewidth=0.8)
    ax1.legend(loc="lower left")

    # Drawdown %
    ax2.plot(dates, daily_df["drawdown_pct"], color="#dc2626", linewidth=1.5, label="Drawdown %")
    ax2.fill_between(dates, daily_df["drawdown_pct"], 0, color="#dc2626", alpha=0.3)
    ax2.set_title("QuantLab Drawdown (%)", fontsize=13, fontweight="bold")
    ax2.set_ylabel("Drawdown %", fontsize=10, fontweight="bold")
    ax2.set_xlabel("Date", fontsize=11, fontweight="bold")
    ax2.axhline(0, color="#9ca3af", linestyle="-", linewidth=0.8)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax2.xaxis.set_major_locator(mdates.YearLocator())
    ax2.legend(loc="lower left")

    plt.xticks(rotation=25)
    plt.tight_layout()
    plt.savefig(out_dir / "drawdown_curve.png", dpi=300)
    plt.close(fig)
