"""
benchmark.py

Benchmark comparison engine for Indian equity strategies.
Primary benchmark: Nifty 50 (NIFTY50).

Features:
1. Loads benchmark historical data.
2. Aligns benchmark dates with strategy backtest dates without look-ahead bias.
3. Computes benchmark performance (CAGR, Volatility, Sharpe Ratio, Max Drawdown).
4. Compares strategy performance vs benchmark (Excess Return, Beta, Correlation, Tracking Error, Information Ratio).
"""

import os
import sys
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_benchmark_data(benchmark_symbol: str = "NIFTY50") -> pd.DataFrame:
    """
    Load historical benchmark OHLCV data.

    Parameters
    ----------
    benchmark_symbol : str
        Benchmark identifier (default: "NIFTY50").

    Returns
    -------
    pd.DataFrame
        Cleaned benchmark dataframe indexed by Date.
    """
    processed_path = os.path.join(BASE_DIR, "data", "processed", f"{benchmark_symbol}.csv")
    raw_path = os.path.join(BASE_DIR, "data", "raw", f"{benchmark_symbol}.csv")

    path = processed_path if os.path.exists(processed_path) else raw_path
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Benchmark dataset '{benchmark_symbol}' not found at {processed_path} or {raw_path}. "
            f"Please ensure Nifty 50 data is downloaded into data/processed/"
        )

    df = pd.read_csv(path, index_col="Date")
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    price_col = "Adj Close" if "Adj Close" in df.columns else "Close"
    if price_col not in df.columns:
        raise ValueError(f"Benchmark dataset missing price column '{price_col}'")

    if "Returns" not in df.columns:
        df["Returns"] = df[price_col].pct_change().fillna(0.0)

    return df


def calculate_benchmark_metrics(benchmark_df: pd.DataFrame, initial_capital: float = 100000.0) -> Dict[str, float]:
    """
    Calculate quantitative performance metrics for the benchmark.
    """
    price_col = "Adj Close" if "Adj Close" in benchmark_df.columns else "Close"
    prices = benchmark_df[price_col]

    start_val = prices.iloc[0]
    end_val = prices.iloc[-1]
    total_return = (end_val - start_val) / start_val

    start_date = benchmark_df.index.min()
    end_date = benchmark_df.index.max()
    days = (end_date - start_date).days
    years = days / 365.25 if days > 0 else 0.0

    cagr = ((end_val / start_val) ** (1.0 / years) - 1.0) if (years > 0 and start_val > 0) else 0.0

    returns = benchmark_df["Returns"] if "Returns" in benchmark_df.columns else prices.pct_change().fillna(0.0)
    daily_vol = returns.std()
    ann_vol = daily_vol * np.sqrt(252) if not pd.isna(daily_vol) and daily_vol > 0 else 0.0

    mean_ret = returns.mean()
    sharpe = (mean_ret / daily_vol) * np.sqrt(252) if daily_vol > 0 else 0.0

    cummax = prices.cummax()
    drawdowns = (prices - cummax) / cummax
    max_dd = drawdowns.min() if not pd.isna(drawdowns.min()) else 0.0

    return {
        "Total Return": float(total_return),
        "CAGR": float(cagr),
        "Annual Volatility": float(ann_vol),
        "Sharpe Ratio": float(sharpe),
        "Maximum Drawdown": float(max_dd)
    }


def compare_strategy_vs_benchmark(
    portfolio_df: pd.DataFrame,
    benchmark_df: Optional[pd.DataFrame] = None,
    benchmark_symbol: str = "NIFTY50",
    initial_capital: float = 100000.0
) -> Dict[str, Any]:
    """
    Compare strategy performance against benchmark and compute relative risk metrics.

    Returns
    -------
    Dict[str, Any]
        Structured comparison parameters.
    """
    if benchmark_df is None:
        try:
            benchmark_df = load_benchmark_data(benchmark_symbol)
        except Exception as e:
            return {
                "status": "ERROR",
                "warning": f"Benchmark data for {benchmark_symbol} unavailable: {e}"
            }

    # Align dates
    combined = pd.DataFrame(index=portfolio_df.index)
    combined["Strategy_Return"] = portfolio_df["Portfolio Return"] if "Portfolio Return" in portfolio_df.columns else portfolio_df["Portfolio Value"].pct_change().fillna(0.0)

    price_col = "Adj Close" if "Adj Close" in benchmark_df.columns else "Close"
    bench_aligned = benchmark_df.reindex(portfolio_df.index).ffill().bfill()
    combined["Benchmark_Return"] = bench_aligned[price_col].pct_change().fillna(0.0)

    # Strategy Metrics
    strat_final = portfolio_df["Portfolio Value"].iloc[-1]
    strat_total_return = (strat_final - initial_capital) / initial_capital
    days = (portfolio_df.index.max() - portfolio_df.index.min()).days
    years = days / 365.25 if days > 0 else 0.0

    strat_cagr = ((strat_final / initial_capital) ** (1.0 / years) - 1.0) if years > 0 else 0.0
    strat_vol = combined["Strategy_Return"].std() * np.sqrt(252)
    strat_sharpe = (combined["Strategy_Return"].mean() / combined["Strategy_Return"].std()) * np.sqrt(252) if combined["Strategy_Return"].std() > 0 else 0.0

    running_max_strat = portfolio_df["Portfolio Value"].cummax()
    strat_mdd = ((portfolio_df["Portfolio Value"] - running_max_strat) / running_max_strat).min()

    # Benchmark Metrics
    bench_start_price = bench_aligned[price_col].iloc[0]
    bench_end_price = bench_aligned[price_col].iloc[-1]
    bench_cagr = ((bench_end_price / bench_start_price) ** (1.0 / years) - 1.0) if (years > 0 and bench_start_price > 0) else 0.0
    bench_vol = combined["Benchmark_Return"].std() * np.sqrt(252)
    bench_sharpe = (combined["Benchmark_Return"].mean() / combined["Benchmark_Return"].std()) * np.sqrt(252) if combined["Benchmark_Return"].std() > 0 else 0.0

    cummax_bench = bench_aligned[price_col].cummax()
    bench_mdd = ((bench_aligned[price_col] - cummax_bench) / cummax_bench).min()

    # Relative Metrics
    excess_cagr = strat_cagr - bench_cagr
    cov = combined["Strategy_Return"].cov(combined["Benchmark_Return"])
    bench_var = combined["Benchmark_Return"].var()
    beta = (cov / bench_var) if bench_var > 0 else 1.0

    correlation = combined["Strategy_Return"].corr(combined["Benchmark_Return"])
    if pd.isna(correlation):
        correlation = 0.0

    diff_returns = combined["Strategy_Return"] - combined["Benchmark_Return"]
    tracking_error = diff_returns.std() * np.sqrt(252)
    information_ratio = (excess_cagr / tracking_error) if (tracking_error > 0) else 0.0

    outperformance_verdict = "Outperformed Benchmark (Positive Alpha)" if excess_cagr > 0 else "Underperformed Benchmark (Negative Alpha)"

    return {
        "status": "SUCCESS",
        "benchmark_symbol": benchmark_symbol,
        "strategy_cagr": float(strat_cagr),
        "benchmark_cagr": float(bench_cagr),
        "excess_return": float(excess_cagr),
        "outperformance_verdict": outperformance_verdict,
        "strategy_sharpe": float(strat_sharpe),
        "benchmark_sharpe": float(bench_sharpe),
        "strategy_volatility": float(strat_vol),
        "benchmark_volatility": float(bench_vol),
        "strategy_max_drawdown": float(strat_mdd),
        "benchmark_max_drawdown": float(bench_mdd),
        "beta": float(beta),
        "correlation": float(correlation),
        "tracking_error": float(tracking_error),
        "information_ratio": float(information_ratio)
    }


if __name__ == "__main__":
    print("Testing benchmark.py module...")
    bench_df = load_benchmark_data("NIFTY50")
    print(f"Loaded benchmark dataset: {len(bench_df)} rows")
    b_metrics = calculate_benchmark_metrics(bench_df)
    print("Benchmark Metrics:", b_metrics)
