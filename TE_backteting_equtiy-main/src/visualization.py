"""
visualization.py

Publication-quality financial chart generation engine for backtesting analysis.
Renders:
1. Signals Chart (Stock Price with Technical Indicators and BUY/SELL triggers)
2. Equity Curve (Portfolio Growth vs Initial Capital)
3. Underwater Drawdown Chart
4. Rolling Sharpe Ratio (Risk-adjusted return dynamics over rolling 60-day window)
5. Monthly Returns (Bar chart of monthly portfolio performance)
6. Trade Return Distribution (Histogram of completed trade returns)
7. Portfolio Allocation (Stacked area of Cash vs Equity asset allocation)
"""

import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Ensure local imports work
sys.path.append(os.path.abspath(os.path.dirname(__file__)))


def setup_plot_style() -> None:
    """Configure modern, clean plot aesthetics for matplotlib."""
    plt.rcParams["figure.figsize"] = (12, 6)
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.3
    plt.rcParams["grid.linestyle"] = "--"
    plt.rcParams["axes.facecolor"] = "#F8F9FA"
    plt.rcParams["savefig.facecolor"] = "#FFFFFF"
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.labelsize"] = 11
    plt.rcParams["axes.titlesize"] = 13
    plt.rcParams["xtick.labelsize"] = 9
    plt.rcParams["ytick.labelsize"] = 9
    plt.rcParams["legend.fontsize"] = 10
    plt.rcParams["axes.edgecolor"] = "#DCDCDC"


def plot_signals(df: pd.DataFrame, ticker: str, output_path: str) -> None:
    """Generate price chart with indicators and BUY/SELL signals."""
    setup_plot_style()
    fig, ax = plt.subplots()

    price_col = "Adj Close" if "Adj Close" in df.columns else "Close"
    ax.plot(df.index, df[price_col], label=f"{ticker} Price", color="#2C3E50", linewidth=1.5)

    # Plot available indicators
    for col in df.columns:
        if col.startswith("SMA") or col.startswith("EMA"):
            color = "#3498DB" if "20" in col or "12" in col else "#E67E22"
            style = "--" if "20" in col or "12" in col else "-"
            ax.plot(df.index, df[col], label=col, color=color, linewidth=1.2, linestyle=style)

    # BUY markers
    buy_dates = df[df["Signal"] == 1].index
    buy_prices = df.loc[buy_dates, price_col]
    ax.scatter(buy_dates, buy_prices, label="BUY Signal", color="#2ECC71", marker="^", s=100, zorder=5)

    # SELL markers
    sell_dates = df[df["Signal"] == -1].index
    sell_prices = df.loc[sell_dates, price_col]
    ax.scatter(sell_dates, sell_prices, label="SELL Signal", color="#E74C3C", marker="v", s=100, zorder=5)

    ax.set_title(f"{ticker} - Technical Strategy Signals", fontweight="semibold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (INR)")
    ax.legend(loc="upper left", frameon=True, facecolor="white")
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_equity_curve(portfolio_df: pd.DataFrame, ticker: str, output_path: str) -> None:
    """Generate portfolio value equity curve."""
    setup_plot_style()
    fig, ax = plt.subplots()

    dates = portfolio_df.index
    portfolio_values = portfolio_df["Portfolio Value"]

    ax.plot(dates, portfolio_values, label="Strategy Equity Curve", color="#27AE60", linewidth=2.0)
    ax.fill_between(dates, portfolio_values, portfolio_values.iloc[0], color="#2ECC71", alpha=0.1)
    ax.axhline(portfolio_values.iloc[0], color="#7F8C8D", linestyle=":", label="Initial Capital", linewidth=1.0)

    ax.set_title(f"{ticker} Backtest - Portfolio Equity Curve", fontweight="semibold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value (INR)")
    ax.legend(loc="upper left", frameon=True, facecolor="white")
    ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_drawdown(portfolio_df: pd.DataFrame, output_path: str) -> None:
    """Generate underwater drawdown chart."""
    setup_plot_style()
    fig, ax = plt.subplots()

    portfolio_values = portfolio_df["Portfolio Value"]
    running_max = portfolio_values.cummax()
    drawdowns = (portfolio_values - running_max) / running_max

    ax.plot(portfolio_df.index, drawdowns * 100, color="#E74C3C", linewidth=1.0)
    ax.fill_between(portfolio_df.index, drawdowns * 100, 0, color="#E74C3C", alpha=0.3, label="Drawdown (%)")

    ax.set_title("Strategy Historical Drawdown (Underwater)", fontweight="semibold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown (%)")
    ax.legend(loc="lower left", frameon=True, facecolor="white")
    ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: f"{x:.1f}%"))

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_rolling_sharpe(portfolio_df: pd.DataFrame, output_path: str, window: int = 60) -> None:
    """Generate rolling annualized Sharpe ratio chart over specified window."""
    setup_plot_style()
    fig, ax = plt.subplots()

    daily_returns = portfolio_df["Portfolio Return"]
    rolling_mean = daily_returns.rolling(window=window).mean()
    rolling_std = daily_returns.rolling(window=window).std()
    rolling_sharpe = (rolling_mean / rolling_std.replace(0, np.nan)) * np.sqrt(252)

    ax.plot(portfolio_df.index, rolling_sharpe, color="#8E44AD", linewidth=1.5, label=f"{window}-Day Rolling Sharpe")
    ax.axhline(0, color="gray", linestyle="--", linewidth=1.0)
    ax.axhline(1.0, color="#2ECC71", linestyle=":", label="Good Sharpe (>1.0)")

    ax.set_title(f"Rolling Sharpe Ratio ({window}-Day Window)", fontweight="semibold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sharpe Ratio")
    ax.legend(loc="upper left", frameon=True, facecolor="white")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_monthly_returns(portfolio_df: pd.DataFrame, output_path: str) -> None:
    """Generate monthly portfolio returns bar chart."""
    setup_plot_style()
    fig, ax = plt.subplots()

    # Resample portfolio value by end of month
    monthly_val = portfolio_df["Portfolio Value"].resample("ME").last()
    monthly_ret = monthly_val.pct_change().dropna() * 100

    colors = ["#2ECC71" if r >= 0 else "#E74C3C" for r in monthly_ret]
    monthly_labels = [d.strftime("%b %Y") for d in monthly_ret.index]

    ax.bar(monthly_labels, monthly_ret, color=colors, alpha=0.85, width=0.6)
    ax.axhline(0, color="black", linewidth=0.8)

    ax.set_title("Monthly Return Breakdown (%)", fontweight="semibold")
    ax.set_xlabel("Month")
    ax.set_ylabel("Return (%)")
    plt.xticks(rotation=45, ha="right", fontsize=8)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_trade_distribution(trade_log_df: pd.DataFrame, output_path: str) -> None:
    """Generate histogram distribution of closed trade returns."""
    setup_plot_style()
    fig, ax = plt.subplots()

    trade_returns = []
    if not trade_log_df.empty:
        type_col = "BUY/SELL" if "BUY/SELL" in trade_log_df.columns else "Type"
        buys = trade_log_df[trade_log_df[type_col] == "BUY"]
        sells = trade_log_df[trade_log_df[type_col] == "SELL"]
        n_trades = min(len(buys), len(sells))
        
        for i in range(n_trades):
            buy_val = buys.iloc[i]["Shares"] * buys.iloc[i]["Execution Price"] + buys.iloc[i].get("Commission", 0.0)
            sell_val = sells.iloc[i]["Shares"] * sells.iloc[i]["Execution Price"] - sells.iloc[i].get("Commission", 0.0)
            if buy_val > 0:
                ret = ((sell_val - buy_val) / buy_val) * 100.0
                trade_returns.append(ret)

    if trade_returns:
        n_bins = max(5, min(20, len(trade_returns)))
        n, bins, patches = ax.hist(trade_returns, bins=n_bins, color="#3498DB", edgecolor="black", alpha=0.7)
        for b, patch in zip(bins, patches):
            if b < 0:
                patch.set_facecolor("#E74C3C")
            else:
                patch.set_facecolor("#2ECC71")

        mean_ret = np.mean(trade_returns)
        ax.axvline(mean_ret, color="#2C3E50", linestyle="--", linewidth=1.5, label=f"Mean Return ({mean_ret:.2f}%)")
    else:
        ax.text(0.5, 0.5, "No completed trades to display", ha="center", va="center", transform=ax.transAxes)

    ax.set_title("Trade Return Distribution (%)", fontweight="semibold")
    ax.set_xlabel("Trade Return (%)")
    ax.set_ylabel("Frequency")
    ax.legend(loc="upper right", frameon=True, facecolor="white")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_portfolio_allocation(portfolio_df: pd.DataFrame, output_path: str) -> None:
    """Generate stacked area chart of Cash vs Equity Allocation."""
    setup_plot_style()
    fig, ax = plt.subplots()

    dates = portfolio_df.index
    cash = portfolio_df["Cash"]
    equity = portfolio_df["Portfolio Value"] - cash

    ax.stackplot(dates, cash, equity, labels=["Cash", "Equity Position"], colors=["#BDC3C7", "#2980B9"], alpha=0.85)

    ax.set_title("Portfolio Asset Allocation (Cash vs Equity)", fontweight="semibold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Value (INR)")
    ax.legend(loc="upper left", frameon=True, facecolor="white")
    ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


if __name__ == "__main__":
    print("Testing visualization.py charts for Phase 4...")
    try:
        from strategy import generate_signals
        from backtester import run_backtest
        
        sample_path = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "RELIANCE.csv")
        if os.path.exists(sample_path):
            df_sample = pd.read_csv(sample_path, index_col="Date")
            df_sample.index = pd.to_datetime(df_sample.index)
            
            df_sig = generate_signals(df_sample, strategy_name="SMA")
            port_df, trade_df = run_backtest(df_sig)
            
            out_dir = "results_test"
            plot_signals(df_sig, "RELIANCE", f"{out_dir}/signals.png")
            plot_equity_curve(port_df, "RELIANCE", f"{out_dir}/equity.png")
            plot_drawdown(port_df, f"{out_dir}/drawdown.png")
            plot_rolling_sharpe(port_df, f"{out_dir}/rolling_sharpe.png")
            plot_monthly_returns(port_df, f"{out_dir}/monthly_returns.png")
            plot_trade_distribution(trade_df, f"{out_dir}/trade_dist.png")
            plot_portfolio_allocation(port_df, f"{out_dir}/allocation.png")
            print("Visualization test successful! Charts written to results_test/")
    except Exception as e:
        print(f"Self-test error: {e}")
