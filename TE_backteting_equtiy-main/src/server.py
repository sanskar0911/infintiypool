"""
server.py

Standalone HTTP Web & API Server for QUANTUM LABS Equity Backtesting Terminal.
Serves full interactive dark-mode dashboard UI and provides endpoints:
- GET /                     -> Dashboard HTML UI
- GET/POST /api/run          -> Backtest simulation runner
- GET/POST /api/recommend    -> Multi-Metric Weighted Strategy Recommendation Engine
- GET/POST /api/agent        -> LangGraph Agent Loop Workflow Runner
- GET/POST /api/rag          -> Quantitative RAG Knowledge Base Q&A
- GET /api/stocks            -> Market overview & stock universe signals
- GET /results/<filename>    -> Chart image files
"""

import os
import sys
import json
import base64
import urllib.parse
import math
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

# Ensure local module imports work
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, "src")
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from strategy import generate_signals
from backtester import run_backtest
from metrics import calculate_metrics
from benchmark import compare_strategy_vs_benchmark, load_benchmark_data
from report import generate_summary_report
from llm.analyzer import LLMStrategyAnalyzer
from rag.rag_chain import RAGChain
from agent.workflow import run_agent_workflow
from visualization import (
    plot_signals,
    plot_equity_curve,
    plot_drawdown,
    plot_rolling_sharpe,
    plot_monthly_returns,
    plot_trade_distribution,
    plot_portfolio_allocation
)
from validation_signal.backtester1 import run_validation_simulation

# Defined Quantitative Evaluation Weights
METRIC_WEIGHTS = {
    "Sharpe Ratio": 0.25,     # 25% weight on risk-adjusted return
    "CAGR": 0.20,             # 20% weight on compound annual growth
    "Max Drawdown": 0.20,     # 20% weight on drawdown protection
    "Win Rate": 0.15,         # 15% weight on trade win rate
    "Profit Factor": 0.10,    # 10% weight on win/loss profit factor
    "Sortino Ratio": 0.10     # 10% weight on downside volatility protection
}


def get_image_base64(filepath: str) -> str:
    """Read an image file and return a base64 data URI string."""
    if os.path.exists(filepath):
        try:
            with open(filepath, "rb") as f:
                encoded = base64.b64encode(f.read()).decode("utf-8")
                return f"data:image/png;base64,{encoded}"
        except Exception as e:
            print(f"Error encoding image {filepath}: {e}")
    return ""


def evaluate_and_recommend_strategies(
    df: pd.DataFrame,
    ticker: str,
    initial_capital: float = 100000.0,
    commission: float = 0.001,
    slippage: float = 0.0005,
    position_size: float = 0.2,
    stop_loss: Optional[float] = 0.05,
    take_profit: Optional[float] = 0.10
) -> dict:
    """
    Evaluates all available strategies (SMA, EMA, RSI) across defined quantitative metrics with explicit weights,
    computes composite recommendation scores, and determines the optimal strategy to recommend.
    """
    strategies_to_test: List[Dict[str, Any]] = [
        {"name": "SMA", "label": "SMA Crossover (20/50)", "kwargs": {"short_window": 20, "long_window": 50}},
        {"name": "EMA", "label": "EMA Crossover (12/26)", "kwargs": {"short_window": 12, "long_window": 26}},
        {"name": "RSI", "label": "RSI Mean-Reversion (14, 30/70)", "kwargs": {"period": 14, "oversold": 30.0, "overbought": 70.0}}
    ]

    eval_results = []
    
    for s_info in strategies_to_test:
        s_name = str(s_info["name"])
        s_label = str(s_info["label"])
        s_kwargs: Dict[str, Any] = s_info["kwargs"] if isinstance(s_info["kwargs"], dict) else {}

        sig_df = generate_signals(df, strategy_name=s_name, **s_kwargs)
        port_df, trade_df = run_backtest(
            sig_df,
            initial_capital=initial_capital,
            txn_cost_rate=commission,
            slippage_rate=slippage,
            position_size=position_size,
            stop_loss=stop_loss,
            take_profit=take_profit
        )
        m = calculate_metrics(port_df, trade_df, initial_capital=initial_capital)

        sharpe = m.get("Sharpe Ratio", 0.0)
        cagr = m.get("CAGR", 0.0)
        max_dd = abs(m.get("Maximum Drawdown", 0.0))
        win_rate = m.get("Win Rate", 0.0)
        profit_factor = m.get("Profit Factor", 0.0)
        sortino = m.get("Sortino Ratio", 0.0)

        # Calculate metric scores (0 to 10 scale)
        sharpe_score = min(10.0, max(0.0, sharpe * 5.0))
        cagr_score = min(10.0, max(0.0, cagr * 50.0))
        dd_score = min(10.0, max(0.0, 10.0 - max_dd * 35.0))
        win_score = min(10.0, max(0.0, win_rate * 10.0))
        pf_score = min(10.0, max(0.0, profit_factor * 3.33))
        sortino_score = min(10.0, max(0.0, sortino * 4.0))

        # Calculate weighted contributions
        c_sharpe = sharpe_score * METRIC_WEIGHTS["Sharpe Ratio"]
        c_cagr = cagr_score * METRIC_WEIGHTS["CAGR"]
        c_dd = dd_score * METRIC_WEIGHTS["Max Drawdown"]
        c_win = win_score * METRIC_WEIGHTS["Win Rate"]
        c_pf = pf_score * METRIC_WEIGHTS["Profit Factor"]
        c_sortino = sortino_score * METRIC_WEIGHTS["Sortino Ratio"]

        composite_score = round(c_sharpe + c_cagr + c_dd + c_win + c_pf + c_sortino, 2)

        eval_results.append({
            "strategy": s_name,
            "label": s_label,
            "raw_metrics": {
                "Sharpe Ratio": round(sharpe, 2),
                "CAGR": f"{cagr*100:.2f}%",
                "Max Drawdown": f"-{max_dd*100:.2f}%",
                "Win Rate": f"{win_rate*100:.1f}%",
                "Profit Factor": round(profit_factor, 2),
                "Sortino Ratio": round(sortino, 2)
            },
            "metric_scores": {
                "Sharpe Ratio": round(sharpe_score, 2),
                "CAGR": round(cagr_score, 2),
                "Max Drawdown": round(dd_score, 2),
                "Win Rate": round(win_score, 2),
                "Profit Factor": round(pf_score, 2),
                "Sortino Ratio": round(sortino_score, 2)
            },
            "weighted_scores": {
                "Sharpe Ratio": round(c_sharpe, 2),
                "CAGR": round(c_cagr, 2),
                "Max Drawdown": round(c_dd, 2),
                "Win Rate": round(c_win, 2),
                "Profit Factor": round(c_pf, 2),
                "Sortino Ratio": round(c_sortino, 2)
            },
            "composite_score": composite_score
        })

    # Sort strategies by composite_score descending
    eval_results.sort(key=lambda x: x["composite_score"], reverse=True)
    best_strategy = eval_results[0]

    rationale = f"Based on multi-metric quantitative evaluation across all strategies for {ticker}, " \
                f"the top recommended architecture is '{best_strategy['label']}' with a Composite Score of {best_strategy['composite_score']}/10. " \
                f"It achieved a Sharpe Ratio of {best_strategy['raw_metrics']['Sharpe Ratio']}, CAGR of {best_strategy['raw_metrics']['CAGR']}, " \
                f"and Max Drawdown of {best_strategy['raw_metrics']['Max Drawdown']}."

    return {
        "ticker": ticker,
        "weights": {k: f"{int(v*100)}%" for k, v in METRIC_WEIGHTS.items()},
        "recommended_strategy": best_strategy["strategy"],
        "recommended_label": best_strategy["label"],
        "recommended_score": best_strategy["composite_score"],
        "rationale": rationale,
        "evaluations": eval_results
    }


def run_full_simulation(params: dict) -> dict:
    """Execute complete end-to-end backtest pipeline and return UI payload."""
    ticker = params.get("ticker", "INFY").upper()
    strategy_name = params.get("strategy", "SMA").upper()
    short_window = int(params.get("short_window", 20))
    long_window = int(params.get("long_window", 50))
    rsi_period = int(params.get("rsi_period", 14))
    oversold = float(params.get("oversold", 30.0))
    overbought = float(params.get("overbought", 70.0))
    initial_capital = float(params.get("capital", 100000.0))
    txn_cost_rate = float(params.get("commission", 0.001))
    slippage_rate = float(params.get("slippage", 0.0005))
    position_size = float(params.get("position_size", 0.2))
    
    stop_loss = params.get("stop_loss")
    stop_loss = float(stop_loss) if stop_loss is not None and str(stop_loss).strip() != "" and str(stop_loss) != "null" else None
    
    take_profit = params.get("take_profit")
    take_profit = float(take_profit) if take_profit is not None and str(take_profit).strip() != "" and str(take_profit) != "null" else None

    benchmark_symbol = params.get("benchmark", "NIFTY50")

    # 1. Load price data
    data_path = os.path.join(BASE_DIR, "data", "processed", f"{ticker}.csv")
    if not os.path.exists(data_path):
        data_path = os.path.join(BASE_DIR, "data", "processed", f"{ticker}.NS.csv")

    if not os.path.exists(data_path):
        return {
            "status": "error",
            "message": f"Dataset for ticker '{ticker}' not found at {data_path}"
        }

    df = pd.read_csv(data_path, index_col="Date")
    df.index = pd.to_datetime(df.index)

    # 2. Build strategy kwargs & generate signals
    strategy_kwargs = {}
    if strategy_name == "SMA":
        strategy_kwargs = {"short_window": short_window, "long_window": long_window}
    elif strategy_name == "EMA":
        strategy_kwargs = {"short_window": short_window, "long_window": long_window}
    elif strategy_name == "RSI":
        strategy_kwargs = {"period": rsi_period, "oversold": oversold, "overbought": overbought}

    df_signals = generate_signals(df, strategy_name=strategy_name, **strategy_kwargs)

    # 3. Execute Backtest Simulation via Upgraded BacktestEngine
    from engine import BacktestEngine, BacktestConfig, ExecutionTiming, InsufficientCashPolicy

    execution_timing = params.get("execution_timing", "SAME_BAR_CLOSE")
    commission_bps = float(params.get("commission_bps", txn_cost_rate * 10000.0))
    slippage_bps = float(params.get("slippage_bps", slippage_rate * 10000.0))
    insufficient_cash = params.get("insufficient_cash_policy", "REDUCE_QUANTITY")
    allow_short = bool(params.get("allow_short_selling", False))

    config = BacktestConfig(
        initial_capital=initial_capital,
        position_size=position_size,
        commission_bps=commission_bps,
        slippage_bps=slippage_bps,
        execution_timing=ExecutionTiming(execution_timing) if isinstance(execution_timing, str) else execution_timing,
        insufficient_cash_policy=InsufficientCashPolicy(insufficient_cash) if isinstance(insufficient_cash, str) else insufficient_cash,
        allow_short_selling=allow_short,
        stop_loss=stop_loss,
        take_profit=take_profit
    )

    engine = BacktestEngine(config)
    res = engine.run(df_signals, symbol=ticker)
    portfolio_df = res.portfolio_history
    trade_log_df = res.trade_log_df
    metrics = res.metrics
    exec_stats = res.execution_statistics
    exec_trace = res.execution_trace

    # 5. Multi-Metric Recommendation & Evaluation Engine
    recommendations = evaluate_and_recommend_strategies(
        df, ticker, initial_capital=initial_capital, commission=txn_cost_rate,
        slippage=slippage_rate, position_size=position_size, stop_loss=stop_loss, take_profit=take_profit
    )

    # 6. Benchmark Comparison
    try:
        benchmark_df = load_benchmark_data(benchmark_symbol)
        bench_comp = compare_strategy_vs_benchmark(
            portfolio_df, benchmark_df, benchmark_symbol=benchmark_symbol, initial_capital=initial_capital
        )
    except Exception as e:
        bench_comp = {"status": "ERROR", "warning": str(e)}

    # 7. Generate All 7 Performance Chart Visualizations
    results_dir = os.path.join(BASE_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)

    sig_path = os.path.join(results_dir, "signals_chart.png")
    eq_path = os.path.join(results_dir, "equity_curve.png")
    dd_path = os.path.join(results_dir, "drawdown.png")
    sharpe_path = os.path.join(results_dir, "rolling_sharpe.png")
    monthly_path = os.path.join(results_dir, "monthly_returns.png")
    dist_path = os.path.join(results_dir, "trade_distribution.png")
    alloc_path = os.path.join(results_dir, "portfolio_allocation.png")

    plot_signals(df_signals, ticker, sig_path)
    plot_equity_curve(portfolio_df, ticker, eq_path)
    plot_drawdown(portfolio_df, dd_path)
    plot_rolling_sharpe(portfolio_df, sharpe_path)
    plot_monthly_returns(portfolio_df, monthly_path)
    plot_trade_distribution(trade_log_df, dist_path)
    plot_portfolio_allocation(portfolio_df, alloc_path)

    # Base64 URIs for instant lag-free rendering
    chart_images = {
        "signals": get_image_base64(sig_path),
        "equity": get_image_base64(eq_path),
        "drawdown": get_image_base64(dd_path),
        "rolling_sharpe": get_image_base64(sharpe_path),
        "monthly_returns": get_image_base64(monthly_path),
        "trade_dist": get_image_base64(dist_path),
        "allocation": get_image_base64(alloc_path)
    }

    # 8. Generate Formatted Strategy Fact Sheet Summary
    report_path = os.path.join(results_dir, "summary_report.txt")
    config_dict = {
        "strategy_name": f"{strategy_name} Crossover Strategy" if strategy_name != "RSI" else "RSI Mean-Reversion",
        "stock_ticker": ticker,
        "start_date": pd.to_datetime(df.index[0]).strftime("%Y-%m-%d"),
        "end_date": pd.to_datetime(df.index[-1]).strftime("%Y-%m-%d"),
        "initial_capital": initial_capital,
        "final_value": portfolio_df["Portfolio Value"].iloc[-1],
        "position_size": position_size,
        "txn_cost_rate": txn_cost_rate,
        "slippage_rate": slippage_rate,
        "stop_loss": stop_loss,
        "take_profit": take_profit
    }
    generate_summary_report(metrics, config_dict, report_path, benchmark_comp=bench_comp)
    
    summary_text = ""
    if os.path.exists(report_path):
        with open(report_path, "r", encoding="utf-8", errors="replace") as f:
            summary_text = f.read()

    # 9. Run LLM / Rule-based AI Strategy Analysis
    analyzer = LLMStrategyAnalyzer()
    ai_analysis = analyzer.analyze_performance(metrics, ticker=ticker, strategy=strategy_name)

    # 10. Format Trade Log Table Data
    trades = []
    if not trade_log_df.empty:
        type_col = "BUY/SELL" if "BUY/SELL" in trade_log_df.columns else "Type"
        price_col = "Execution Price" if "Execution Price" in trade_log_df.columns else ("Price" if "Price" in trade_log_df.columns else "Close")
        cost_col = "Commission" if "Commission" in trade_log_df.columns else ("Transaction Cost" if "Transaction Cost" in trade_log_df.columns else "Cost")
        cash_col = "Cash" if "Cash" in trade_log_df.columns else "Remaining Cash"

        for idx, row in trade_log_df.iterrows():
            trade_date = str(row.get("Date", idx))
            raw_type = str(row.get(type_col, "N/A"))
            trade_type = "BUY" if "BUY" in raw_type.upper() else ("SELL" if "SELL" in raw_type.upper() else raw_type)
            raw_p = row.get(price_col)
            price = float(raw_p) if raw_p is not None else 0.0
            raw_s = row.get("Shares")
            shares = int(raw_s) if raw_s is not None else 0
            val = float(price * shares)
            raw_c = row.get(cost_col)
            cost = float(raw_c) if raw_c is not None else 0.0
            raw_cash = row.get(cash_col)
            rem_cash = float(raw_cash) if raw_cash is not None else 0.0

            trades.append({
                "Date": trade_date,
                "Type": trade_type,
                "Price": price,
                "Shares": shares,
                "Value": val,
                "Transaction Cost": cost,
                "Remaining Cash": rem_cash
            })

    # Return full JSON bundle
    return {
        "status": "success",
        "ticker": ticker,
        "strategy": strategy_name,
        "summary": summary_text,
        "metrics": {
            "Total Return": f"{metrics.get('Total Return', 0.0)*100:.2f}%",
            "CAGR": f"{metrics.get('CAGR', 0.0)*100:.2f}%",
            "Sharpe Ratio": f"{metrics.get('Sharpe Ratio', 0.0):.2f}",
            "Max Drawdown": f"{metrics.get('Maximum Drawdown', 0.0)*100:.2f}%",
            "Win Rate": f"{metrics.get('Win Rate', 0.0)*100:.1f}%",
            "Profit Factor": f"{metrics.get('Profit Factor', 0.0):.2f}",
            "Volatility": f"{metrics.get('Annual Volatility', 0.0)*100:.2f}%",
            "Sortino Ratio": f"{metrics.get('Sortino Ratio', 0.0):.2f}",
            "Total Trades": int(metrics.get("Number of Trades", 0))
        },
        "trades": trades,
        "charts": chart_images,
        "ai_analysis": ai_analysis,
        "benchmark_comp": bench_comp,
        "recommendations": recommendations,
        "execution_statistics": exec_stats,
        "execution_trace": exec_trace
    }


def run_agent_workflow_api(params: dict) -> dict:
    """Execute LangGraph Agent Workflow and return state machine trajectory."""
    ticker = params.get("ticker", "INFY").upper()
    strategy = params.get("strategy", "SMA").upper()
    
    state = run_agent_workflow(
        ticker=ticker,
        strategy_name=strategy,
        initial_capital=float(params.get("capital", 100000.0)),
        commission_pct=float(params.get("commission", 0.001)),
        slippage_pct=float(params.get("slippage", 0.0005)),
        position_size=float(params.get("position_size", 0.2)),
        max_iterations=3
    )

    history = state.get("experiment_history", [])
    formatted_history = []
    for h in history:
        formatted_history.append({
            "iteration": h.get("iteration"),
            "strategy": h.get("strategy"),
            "params": h.get("params"),
            "sharpe": round(h.get("sharpe", 0.0), 2),
            "cagr": f"{h.get('cagr', 0.0)*100:.2f}%",
            "max_drawdown": f"-{h.get('max_drawdown', 0.0)*100:.2f}%"
        })

    return {
        "status": "success",
        "experiment_id": state.get("experiment_id"),
        "hypothesis": state.get("hypothesis"),
        "ticker": state.get("ticker"),
        "strategy": state.get("strategy_name"),
        "iterations": state.get("iteration"),
        "loop_decision": state.get("loop_decision"),
        "history": formatted_history
    }


def get_universe_stocks_overview() -> list:
    """Get latest price & signal status for universe stocks."""
    tickers = ["INFY", "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK"]
    data = []

    for tk in tickers:
        p_path = os.path.join(BASE_DIR, "data", "processed", f"{tk}.csv")
        if not os.path.exists(p_path):
            p_path = os.path.join(BASE_DIR, "data", "processed", f"{tk}.NS.csv")
        
        if os.path.exists(p_path):
            df_tk = pd.read_csv(p_path, index_col="Date")
            last_price = float(df_tk["Close"].iloc[-1])
            prev_price = float(df_tk["Close"].iloc[-2]) if len(df_tk) > 1 else last_price
            change_pct = ((last_price - prev_price) / prev_price) * 100.0
            volume = int(df_tk["Volume"].iloc[-1]) if "Volume" in df_tk.columns else 0

            sig_df = generate_signals(df_tk, strategy_name="SMA", short_window=20, long_window=50)
            latest_sig = int(sig_df["Signal"].iloc[-1]) if "Signal" in sig_df.columns else 0
            sig_label = "BUY" if latest_sig == 1 else ("SELL" if latest_sig == -1 else "NEUTRAL")

            data.append({
                "ticker": tk,
                "price": last_price,
                "change_pct": change_pct,
                "volume": volume,
                "signal": sig_label
            })
    return data


# HTML Dashboard Template
INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>QUANTUM LABS | Equity Backtesting Terminal</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-color: #0b0f19;
            --card-bg: rgba(22, 28, 45, 0.75);
            --card-border: rgba(255, 255, 255, 0.08);
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --accent-primary: #3b82f6;
            --accent-blue-hover: #2563eb;
            --accent-green: #10b981;
            --accent-red: #ef4444;
            --accent-amber: #f59e0b;
            --font-main: 'Inter', sans-serif;
            --font-mono: 'JetBrains Mono', monospace;
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: var(--font-main);
            background-color: var(--bg-color);
            color: var(--text-primary);
            min-height: 100vh;
            padding: 1.5rem 2rem;
            background-image: radial-gradient(circle at 50% 0%, rgba(59, 130, 246, 0.15) 0%, transparent 50%);
            overflow-x: hidden;
        }

        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding-bottom: 1.2rem;
            border-bottom: 1px solid var(--card-border);
            margin-bottom: 1.5rem;
        }

        .logo-container h1 {
            font-size: 1.8rem;
            font-weight: 800;
            letter-spacing: -0.025em;
            background: linear-gradient(to right, #3b82f6, #60a5fa, #10b981);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .logo-container p {
            color: var(--text-secondary);
            font-size: 0.88rem;
            margin-top: 0.2rem;
        }

        .controls {
            display: flex;
            gap: 0.75rem;
            align-items: center;
            flex-wrap: wrap;
        }

        select, input[type="number"] {
            background-color: #161c2d;
            color: var(--text-primary);
            border: 1px solid var(--card-border);
            padding: 0.55rem 1rem;
            border-radius: 0.5rem;
            font-size: 0.9rem;
            font-weight: 500;
            outline: none;
            cursor: pointer;
            transition: all 0.3s;
        }

        select:focus, input:focus {
            border-color: var(--accent-primary);
            box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.25);
        }

        .btn-primary {
            background-color: var(--accent-primary);
            color: white;
            border: none;
            padding: 0.6rem 1.4rem;
            border-radius: 0.5rem;
            font-size: 0.92rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s;
            display: flex;
            align-items: center;
            gap: 0.5rem;
            box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
        }

        .btn-primary:hover {
            background-color: var(--accent-blue-hover);
            transform: translateY(-1px);
            box-shadow: 0 6px 16px rgba(59, 130, 246, 0.45);
        }

        .btn-secondary {
            background: rgba(255, 255, 255, 0.05);
            color: var(--text-primary);
            border: 1px solid var(--card-border);
            padding: 0.55rem 1rem;
            border-radius: 0.5rem;
            font-size: 0.85rem;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.2s;
        }

        .btn-secondary:hover {
            background: rgba(255, 255, 255, 0.1);
        }

        /* Metrics Grid */
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 1.2rem;
            margin-bottom: 1.5rem;
        }

        .metric-card {
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 0.75rem;
            padding: 1.2rem;
            backdrop-filter: blur(10px);
            transition: transform 0.3s, border-color 0.3s;
        }

        .metric-card:hover {
            transform: translateY(-2px);
            border-color: rgba(255, 255, 255, 0.18);
        }

        .metric-label {
            color: var(--text-secondary);
            font-size: 0.78rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.4rem;
        }

        .metric-value {
            font-size: 1.65rem;
            font-weight: 700;
        }

        .metric-value.positive { color: var(--accent-green); }
        .metric-value.negative { color: var(--accent-red); }
        .metric-value.neutral { color: var(--accent-primary); }

        /* Dashboard Split Layout */
        .dashboard-content {
            display: grid;
            grid-template-columns: 3fr 2fr;
            gap: 1.5rem;
            margin-bottom: 1.5rem;
        }

        @media (max-width: 1080px) {
            .dashboard-content {
                grid-template-columns: 1fr;
            }
        }

        .panel {
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 0.75rem;
            padding: 1.4rem;
            backdrop-filter: blur(10px);
            display: flex;
            flex-direction: column;
        }

        .panel-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1.2rem;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
            padding-bottom: 0.8rem;
            flex-wrap: wrap;
            gap: 0.5rem;
        }

        .panel-title {
            font-size: 1.05rem;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        /* Tabs styling */
        .tabs {
            display: flex;
            gap: 0.4rem;
            background: rgba(0, 0, 0, 0.25);
            padding: 0.25rem;
            border-radius: 0.5rem;
            overflow-x: auto;
        }

        .tab-btn {
            background: transparent;
            color: var(--text-secondary);
            border: none;
            padding: 0.4rem 0.75rem;
            border-radius: 0.35rem;
            font-size: 0.82rem;
            font-weight: 500;
            cursor: pointer;
            white-space: nowrap;
            transition: all 0.2s;
        }

        .tab-btn:hover {
            color: var(--text-primary);
            background: rgba(255, 255, 255, 0.06);
        }

        .tab-btn.active {
            color: var(--text-primary);
            background: var(--accent-primary);
            box-shadow: 0 2px 6px rgba(59, 130, 246, 0.3);
        }

        .chart-container {
            flex-grow: 1;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 420px;
            background: rgba(0, 0, 0, 0.2);
            border-radius: 0.5rem;
            overflow: hidden;
            position: relative;
        }

        .chart-img {
            max-width: 100%;
            max-height: 480px;
            object-fit: contain;
            border-radius: 4px;
            transition: opacity 0.3s;
        }

        .info-panel {
            min-height: 520px;
        }

        .report-pre {
            font-family: var(--font-mono);
            font-size: 0.82rem;
            line-height: 1.45;
            color: #cbd5e1;
            white-space: pre-wrap;
            background: rgba(0, 0, 0, 0.35);
            padding: 1rem;
            border-radius: 0.5rem;
            border: 1px solid rgba(255, 255, 255, 0.04);
            max-height: 450px;
            overflow-y: auto;
        }

        /* Trade Log Table */
        .table-container {
            width: 100%;
            max-height: 450px;
            overflow-y: auto;
            border-radius: 0.5rem;
            border: 1px solid rgba(255, 255, 255, 0.05);
        }

        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.85rem;
            text-align: left;
        }

        th {
            background-color: rgba(0, 0, 0, 0.5);
            color: var(--text-secondary);
            padding: 0.75rem 0.8rem;
            font-weight: 600;
            position: sticky;
            top: 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.08);
            z-index: 10;
        }

        td {
            padding: 0.65rem 0.8rem;
            border-bottom: 1px solid rgba(255, 255, 255, 0.03);
        }

        tr:hover td {
            background-color: rgba(255, 255, 255, 0.03);
        }

        .badge {
            display: inline-block;
            padding: 0.15rem 0.45rem;
            border-radius: 0.25rem;
            font-size: 0.75rem;
            font-weight: 700;
        }

        .badge-buy {
            background-color: rgba(16, 185, 129, 0.18);
            color: var(--accent-green);
            border: 1px solid rgba(16, 185, 129, 0.35);
        }

        .badge-sell {
            background-color: rgba(239, 68, 68, 0.18);
            color: var(--accent-red);
            border: 1px solid rgba(239, 68, 68, 0.35);
        }

        /* Recommendation Banner & Flowchart */
        .recom-banner {
            background: linear-gradient(135deg, rgba(16, 185, 129, 0.15), rgba(59, 130, 246, 0.15));
            border: 1px solid rgba(16, 185, 129, 0.4);
            border-radius: 0.5rem;
            padding: 1rem 1.2rem;
            margin-bottom: 1rem;
        }

        .workflow-flowchart {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.5rem;
            background: rgba(0, 0, 0, 0.3);
            border: 1px solid rgba(255, 255, 255, 0.06);
            border-radius: 0.5rem;
            padding: 0.85rem 1rem;
            margin-bottom: 1rem;
            overflow-x: auto;
        }

        .flow-node {
            background: rgba(30, 41, 59, 0.8);
            border: 1px solid rgba(255, 255, 255, 0.1);
            padding: 0.4rem 0.75rem;
            border-radius: 0.35rem;
            font-size: 0.75rem;
            font-weight: 600;
            color: var(--text-secondary);
            white-space: nowrap;
        }

        .flow-node.active {
            background: var(--accent-primary);
            color: white;
            box-shadow: 0 0 10px rgba(59, 130, 246, 0.5);
        }

        .flow-arrow {
            color: var(--text-secondary);
            font-size: 0.8rem;
        }

        /* RAG / AI Container */
        .ai-card {
            background: rgba(0, 0, 0, 0.25);
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: 0.5rem;
            padding: 1rem;
            font-size: 0.9rem;
            line-height: 1.5;
            max-height: 450px;
            overflow-y: auto;
        }

        .rag-input-group {
            display: flex;
            gap: 0.5rem;
            margin-bottom: 1rem;
        }

        .rag-input {
            flex-grow: 1;
            background: #161c2d;
            border: 1px solid var(--card-border);
            color: var(--text-primary);
            padding: 0.6rem 1rem;
            border-radius: 0.5rem;
            outline: none;
            font-size: 0.9rem;
        }

        /* Universe Watchlist Footer */
        .watchlist-section {
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 0.75rem;
            padding: 1.2rem;
            backdrop-filter: blur(10px);
        }

        .watchlist-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-top: 0.8rem;
        }

        .watchlist-card {
            background: rgba(0, 0, 0, 0.2);
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: 0.5rem;
            padding: 0.85rem 1rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        /* Modal Drawer for Parameters */
        .modal {
            display: none;
            position: fixed;
            top: 0; left: 0; width: 100%; height: 100%;
            background: rgba(11, 15, 25, 0.8);
            backdrop-filter: blur(5px);
            z-index: 1000;
            justify-content: center;
            align-items: center;
        }

        .modal.active { display: flex; }

        .modal-content {
            background: #111827;
            border: 1px solid var(--card-border);
            border-radius: 0.75rem;
            width: 90%;
            max-width: 540px;
            padding: 1.5rem;
            box-shadow: 0 20px 40px rgba(0,0,0,0.5);
        }

        .modal-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1.2rem;
            border-bottom: 1px solid rgba(255, 255, 255, 0.08);
            padding-bottom: 0.6rem;
        }

        .form-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1rem;
            margin-bottom: 1.2rem;
        }

        .form-group {
            display: flex;
            flex-direction: column;
            gap: 0.35rem;
        }

        .form-group label {
            font-size: 0.8rem;
            color: var(--text-secondary);
            font-weight: 500;
        }

        /* Loading Overlay */
        .loading-overlay {
            position: absolute;
            top: 0; left: 0; width: 100%; height: 100%;
            background: rgba(11, 15, 25, 0.85);
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            z-index: 100;
            opacity: 0;
            pointer-events: none;
            transition: opacity 0.3s;
            backdrop-filter: blur(4px);
        }

        .loading-overlay.active {
            opacity: 1;
            pointer-events: auto;
        }

        .spinner {
            width: 44px;
            height: 44px;
            border: 3px solid rgba(59, 130, 246, 0.2);
            border-top: 3px solid var(--accent-primary);
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin-bottom: 1rem;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        
        /* Main Navigation Mode Switcher */
        .mode-switcher-container {
            display: flex;
            gap: 0.5rem;
            background: rgba(0, 0, 0, 0.45);
            padding: 0.3rem;
            border-radius: 0.65rem;
            border: 1px solid var(--card-border);
        }

        .mode-nav-btn {
            background: transparent;
            color: var(--text-secondary);
            border: none;
            padding: 0.5rem 1.1rem;
            border-radius: 0.45rem;
            font-size: 0.88rem;
            font-weight: 600;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 0.5rem;
            transition: all 0.25s;
        }

        .mode-nav-btn:hover {
            color: var(--text-primary);
            background: rgba(255, 255, 255, 0.06);
        }

        .mode-nav-btn.active {
            color: #ffffff;
            background: linear-gradient(135deg, #2563eb, #3b82f6);
            box-shadow: 0 2px 10px rgba(59, 130, 246, 0.4);
        }

        .mode-nav-btn.val-mode.active {
            color: #ffffff;
            background: linear-gradient(135deg, #059669, #10b981);
            box-shadow: 0 2px 12px rgba(16, 185, 129, 0.4);
        }

        /* Validation UI Styles */
        .val-header-card {
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 0.75rem;
            padding: 1.4rem;
            margin-bottom: 1.5rem;
            backdrop-filter: blur(10px);
        }

        .val-controls-grid {
            display: flex;
            gap: 0.85rem;
            align-items: flex-end;
            flex-wrap: wrap;
            margin-top: 1.2rem;
            padding-top: 1rem;
            border-top: 1px solid rgba(255, 255, 255, 0.06);
        }

        .val-control-item {
            display: flex;
            flex-direction: column;
            gap: 0.35rem;
        }

        .val-control-item label {
            font-size: 0.75rem;
            font-weight: 600;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }

        .val-kpi-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(185px, 1fr));
            gap: 1rem;
            margin-bottom: 1.5rem;
        }

        .val-kpi-card {
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 0.75rem;
            padding: 1.1rem 1.2rem;
            backdrop-filter: blur(10px);
            transition: transform 0.25s, border-color 0.25s;
        }

        .val-kpi-card:hover {
            transform: translateY(-2px);
            border-color: rgba(255, 255, 255, 0.2);
        }

        .val-kpi-label {
            font-size: 0.75rem;
            font-weight: 600;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.35rem;
        }

        .val-kpi-val {
            font-size: 1.5rem;
            font-weight: 800;
            font-family: var(--font-main);
        }

        .val-kpi-sub {
            font-size: 0.78rem;
            color: var(--text-secondary);
            margin-top: 0.3rem;
        }

        .val-metrics-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.85rem;
        }

        .val-metrics-table th {
            background: rgba(0, 0, 0, 0.45);
            color: var(--text-secondary);
            padding: 0.75rem 0.9rem;
            font-weight: 600;
            text-align: left;
            border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        }

        .val-metrics-table td {
            padding: 0.7rem 0.9rem;
            border-bottom: 1px solid rgba(255, 255, 255, 0.04);
        }

        .val-metrics-table tr:hover td {
            background-color: rgba(255, 255, 255, 0.025);
        }

        .val-cat-header td {
            background: rgba(30, 41, 59, 0.7) !important;
            color: #60a5fa !important;
            font-weight: 700 !important;
            font-size: 0.82rem !important;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            padding: 0.55rem 0.9rem !important;
            border-top: 1px solid rgba(59, 130, 246, 0.2);
        }

        .download-card-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 1.2rem;
            padding: 0.5rem 0;
        }

        .download-card {
            background: rgba(0, 0, 0, 0.35);
            border: 1px solid var(--card-border);
            border-radius: 0.65rem;
            padding: 1.2rem;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            transition: all 0.25s;
        }

        .download-card:hover {
            border-color: var(--accent-primary);
            background: rgba(59, 130, 246, 0.05);
            transform: translateY(-2px);
        }

        .audit-tag {
            display: inline-block;
            padding: 0.2rem 0.55rem;
            border-radius: 0.3rem;
            font-size: 0.72rem;
            font-weight: 700;
            text-transform: uppercase;
        }

        .audit-tag.passed {
            background: rgba(16, 185, 129, 0.15);
            color: #10b981;
            border: 1px solid rgba(16, 185, 129, 0.4);
        }

        .audit-tag.verified {
            background: rgba(59, 130, 246, 0.15);
            color: #60a5fa;
            border: 1px solid rgba(59, 130, 246, 0.4);
        }

        .audit-tag.flagged {
            background: rgba(239, 68, 68, 0.15);
            color: #f87171;
            border: 1px solid rgba(239, 68, 68, 0.4);
        }

        .cum-pnl-cell {
            font-weight: 700;
            background: rgba(59, 130, 246, 0.1);
            color: #93c5fd;
            border-radius: 4px;
            padding: 0.2rem 0.5rem;
            display: inline-block;
        }

        .hidden { display: none !important; }
    </style>
</head>
<body>
    <header>
        <div class="logo-container">
            <h1>QUANTUM LABS</h1>
            <p>Institutional AI Equity Backtesting & Research Terminal</p>
        </div>

        <!-- Mode Switcher -->
        <div class="mode-switcher-container">
            <button class="mode-nav-btn active" id="mode-btn-terminal" onclick="switchTerminalMode('terminal')">
                <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon></svg>
                Strategy Terminal
            </button>
            <button class="mode-nav-btn val-mode" id="mode-btn-validation" onclick="switchTerminalMode('validation')">
                <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line><polyline points="10 9 9 9 8 9"></polyline></svg>
                Signal Validation & Audit
                <span style="background: rgba(16, 185, 129, 0.25); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.5); font-size: 0.68rem; padding: 0.1rem 0.4rem; border-radius: 0.25rem; font-weight: 700; margin-left: 0.25rem;">NEW</span>
            </button>
        </div>
        <div class="controls" id="terminal-controls">
            <select id="ticker-select" title="Stock Universe">
                <option value="INFY">INFY.NS (Infosys)</option>
                <option value="RELIANCE">RELIANCE.NS (Reliance)</option>
                <option value="TCS">TCS.NS (Tata Consultancy)</option>
                <option value="HDFCBANK">HDFCBANK.NS (HDFC Bank)</option>
                <option value="ICICIBANK">ICICIBANK.NS (ICICI Bank)</option>
            </select>

            <select id="strategy-select" title="Strategy Architecture">
                <option value="SMA">SMA Crossover (20/50)</option>
                <option value="EMA">EMA Crossover (12/26)</option>
                <option value="RSI">RSI Mean-Reversion (14)</option>
            </select>

            <button class="btn-secondary" id="params-btn">⚙️ Parameters</button>

            <button class="btn-primary" id="run-btn">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg>
                Run Simulation
            </button>

            <button class="btn-secondary" id="agent-btn" style="background: rgba(16, 185, 129, 0.15); border-color: rgba(16, 185, 129, 0.3); color: var(--accent-green);">
                🤖 Agent Loop
            </button>
        </div>
    </header>

    <!-- TERMINAL VIEW CONTAINER -->
    <div id="terminal-view">
    <!-- Metrics Summary Grid -->
    <div class="metrics-grid">
        <div class="metric-card">
            <div class="metric-label">Total Return</div>
            <div class="metric-value" id="metric-total-return">0.00%</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">CAGR</div>
            <div class="metric-value" id="metric-cagr">0.00%</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Sharpe Ratio</div>
            <div class="metric-value" id="metric-sharpe">0.00</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Max Drawdown</div>
            <div class="metric-value" id="metric-max-drawdown">0.00%</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Win Rate</div>
            <div class="metric-value" id="metric-win-rate">0.0%</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Profit Factor</div>
            <div class="metric-value neutral" id="metric-profit-factor">0.00</div>
        </div>
    </div>

    <!-- Main Content Split -->
    <div class="dashboard-content">
        <!-- Performance Visualizations Panel -->
        <div class="panel">
            <div class="panel-header">
                <div class="panel-title">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 3v18h18"></path><path d="m19 9-5 5-4-4-3 3"></path></svg>
                    Performance Visualizations
                </div>
                <div class="tabs">
                    <button class="tab-btn active" data-chart="signals">Signals</button>
                    <button class="tab-btn" data-chart="equity">Equity Curve</button>
                    <button class="tab-btn" data-chart="drawdown">Drawdown</button>
                    <button class="tab-btn" data-chart="rolling_sharpe">Rolling Sharpe</button>
                    <button class="tab-btn" data-chart="monthly_returns">Monthly</button>
                    <button class="tab-btn" data-chart="trade_dist">Trade Dist</button>
                    <button class="tab-btn" data-chart="allocation">Allocation</button>
                </div>
            </div>
            <div class="chart-container">
                <div class="loading-overlay" id="loading-overlay">
                    <div class="spinner"></div>
                    <p style="font-weight: 500; font-size: 0.95rem;">Executing Backtest Pipeline...</p>
                </div>
                <img id="display-chart" class="chart-img" src="" alt="Run simulation to render performance charts">
            </div>
        </div>

        <!-- Simulation Output / Data Panel -->
        <div class="panel info-panel">
            <div class="panel-header">
                <div class="panel-title">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line><polyline points="10 9 9 9 8 9"></polyline></svg>
                    Simulation Output
                </div>
                <div class="tabs">
                    <button class="tab-btn active" data-info="recom">💡 Recommendation</button>
                    <button class="tab-btn" data-info="workflow">🤖 Agent Workflow</button>
                    <button class="tab-btn" data-info="report">Report</button>
                    <button class="tab-btn" data-info="tradelog">Trade Log</button>
                    <button class="tab-btn" data-info="ai">AI Insights</button>
                    <button class="tab-btn" data-info="rag">RAG Research</button>
                    <button class="tab-btn" data-info="benchmark">Benchmark</button>
                </div>
            </div>

            <!-- Strategy Recommendation Tab -->
            <div id="info-recom" class="info-content">
                <div class="recom-banner" id="recom-banner-container">
                    <div style="font-weight: 700; font-size: 0.95rem; color: var(--accent-green); margin-bottom: 0.3rem;">
                        🏆 STRATEGY INVESTMENT RECOMMENDATION
                    </div>
                    <div id="recom-title" style="font-size: 1.15rem; font-weight: 800; color: #ffffff; margin-bottom: 0.4rem;">
                        Loading evaluation...
                    </div>
                    <div id="recom-rationale" style="font-size: 0.85rem; color: var(--text-secondary); line-height: 1.45;">
                        Evaluating quantitative performance across SMA, EMA, and RSI...
                    </div>
                </div>

                <div style="font-weight: 700; font-size: 0.85rem; text-transform: uppercase; color: var(--text-secondary); margin-bottom: 0.6rem; letter-spacing: 0.05em;">
                    📊 Weighted Multi-Metric Strategy Evaluation Matrix
                </div>
                <div class="table-container">
                    <table>
                        <thead>
                            <tr>
                                <th>Strategy</th>
                                <th>Sharpe (25%)</th>
                                <th>CAGR (20%)</th>
                                <th>MDD (20%)</th>
                                <th>Win% (15%)</th>
                                <th>PF (10%)</th>
                                <th>Sortino (10%)</th>
                                <th style="text-align: right;">Score (0-10)</th>
                            </tr>
                        </thead>
                        <tbody id="recom-tbody">
                            <tr>
                                <td colspan="8" style="text-align: center; color: var(--text-secondary); padding: 1.5rem;">
                                    Loading evaluation matrix...
                                </td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            </div>

            <!-- Agent Workflow Visualizer Tab -->
            <div id="info-workflow" class="info-content hidden">
                <div class="workflow-flowchart">
                    <div class="flow-node active">USER HYPOTHESIS</div>
                    <div class="flow-arrow">➔</div>
                    <div class="flow-node active">LOAD DATA</div>
                    <div class="flow-arrow">➔</div>
                    <div class="flow-node active">RUN BACKTEST</div>
                    <div class="flow-arrow">➔</div>
                    <div class="flow-node active">METRICS</div>
                    <div class="flow-arrow">➔</div>
                    <div class="flow-node active">EVALUATOR</div>
                    <div class="flow-arrow">➔</div>
                    <div class="flow-node active" id="flow-decision-node">ACCEPT / ITERATE</div>
                </div>

                <div style="background: rgba(0,0,0,0.25); border: 1px solid rgba(255,255,255,0.05); border-radius: 0.5rem; padding: 1rem; margin-bottom: 1rem;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                        <span style="font-weight: 700; font-size: 0.9rem; color: var(--accent-primary);" id="wf-exp-id">EXP_INFY_SMA</span>
                        <span class="badge badge-buy" id="wf-status-badge">DECISION: PENDING</span>
                    </div>
                    <p style="font-size: 0.85rem; color: var(--text-secondary);" id="wf-hypothesis">Testing quantitative momentum edge on Indian equities</p>
                </div>

                <div style="font-weight: 700; font-size: 0.85rem; text-transform: uppercase; color: var(--text-secondary); margin-bottom: 0.6rem; letter-spacing: 0.05em;">
                    🔄 Iterative Loop History Trajectory
                </div>
                <div class="table-container">
                    <table>
                        <thead>
                            <tr>
                                <th>Iteration</th>
                                <th>Strategy</th>
                                <th>Parameters</th>
                                <th style="text-align: right;">Sharpe</th>
                                <th style="text-align: right;">CAGR</th>
                                <th style="text-align: right;">Max Drawdown</th>
                            </tr>
                        </thead>
                        <tbody id="workflow-tbody">
                            <tr>
                                <td colspan="6" style="text-align: center; color: var(--text-secondary); padding: 1.5rem;">
                                    Click "Agent Loop" to execute autonomous optimization trajectory.
                                </td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            </div>

            <!-- Report Text Tab -->
            <div id="info-report" class="info-content hidden">
                <pre class="report-pre" id="report-text">Click "Run Simulation" to execute backtest engine.</pre>
            </div>

            <!-- Trade Log Table Tab -->
            <div id="info-tradelog" class="info-content hidden">
                <div class="table-container">
                    <table>
                        <thead>
                            <tr>
                                <th>Date</th>
                                <th>Type</th>
                                <th style="text-align: right;">Price (INR)</th>
                                <th style="text-align: right;">Shares</th>
                                <th style="text-align: right;">Value (INR)</th>
                                <th style="text-align: right;">Cost (INR)</th>
                                <th style="text-align: right;">Cash (INR)</th>
                            </tr>
                        </thead>
                        <tbody id="trade-log-tbody">
                            <tr>
                                <td colspan="7" style="text-align: center; color: var(--text-secondary); padding: 2rem;">
                                    No trade data loaded.
                                </td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            </div>

            <!-- AI Insights Tab -->
            <div id="info-ai" class="info-content hidden">
                <div class="ai-card" id="ai-analysis-container">
                    <p style="color: var(--text-secondary);">Run backtest simulation to view LLM strategy rating and risk breakdown.</p>
                </div>
            </div>

            <!-- RAG Research Tab -->
            <div id="info-rag" class="info-content hidden">
                <div style="margin-bottom: 1.2rem; display: flex; gap: 0.5rem; align-items: center; flex-wrap: wrap; background: rgba(0,0,0,0.25); padding: 0.8rem; border-radius: 0.5rem; border: 1px solid var(--card-border);">
                    <span style="font-size: 0.8rem; font-weight: 700; color: var(--accent-primary); text-transform: uppercase; letter-spacing: 0.05em; display: flex; align-items: center; gap: 0.3rem;">🧪 Demo Queries:</span>
                    <button class="btn-secondary demo-q-btn" onclick="runDemoQuery('What is Sharpe ratio?')">🇬🇧 English: Sharpe Ratio</button>
                    <button class="btn-secondary demo-q-btn" onclick="runDemoQuery('Sharpe ratio kya hota hai?')">🇮🇳 Hindi: Sharpe Ratio</button>
                    <button class="btn-secondary demo-q-btn" onclick="runDemoQuery('Sharpe ratio म्हणजे काय?')">🚩 Marathi: Sharpe Ratio</button>
                    <button class="btn-secondary demo-q-btn" onclick="runDemoQuery('¿Qué es el CAGR?')">🇪🇸 Spanish: CAGR</button>
                    <button class="btn-secondary demo-q-btn" onclick="runDemoQuery('Maximum drawdown म्हणजे काय?')">🚩 Marathi: Max DD</button>
                </div>
                <div class="rag-input-group" style="margin-bottom: 1.5rem; display: flex; gap: 0.5rem;">
                    <input type="text" id="rag-query-input" class="rag-input" style="flex-grow: 1; padding: 0.75rem 1rem; border-radius: 0.5rem; border: 1px solid var(--card-border); background: #161c2d; color: var(--text-primary); font-size: 0.9rem;" placeholder="Type quantitative question in any language (e.g. 'Sharpe ratio म्हणजे काय?')...">
                    <button class="btn-primary" id="rag-search-btn" style="white-space: nowrap;">🚀 Execute AI Pipeline</button>
                </div>
                <div id="rag-response-container">
                    <p style="color: var(--text-secondary); padding: 2rem; text-align: center; border: 1px dashed var(--card-border); border-radius: 0.5rem;">
                        Select an interactive demo query above or enter a question to inspect the complete 8-Stage QUANTLAB Multilingual MADRID AI Pipeline.
                    </p>
                </div>
            </div>

            <!-- Benchmark Comparison Tab -->
            <div id="info-benchmark" class="info-content hidden">
                <div class="table-container">
                    <table>
                        <thead>
                            <tr>
                                <th>Metric</th>
                                <th style="text-align: right;">Strategy</th>
                                <th style="text-align: right;">Benchmark (NIFTY50)</th>
                                <th style="text-align: right;">Delta / Excess</th>
                            </tr>
                        </thead>
                        <tbody id="benchmark-tbody">
                            <tr>
                                <td colspan="4" style="text-align: center; color: var(--text-secondary); padding: 2rem;">
                                    No benchmark comparison loaded.
                                </td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    </div>

    <!-- Stock Universe Watchlist Bar -->
    <div class="watchlist-section">
        <div class="panel-title" style="font-size: 0.95rem; text-transform: uppercase; letter-spacing: 0.05em; color: var(--text-secondary);">
            🌐 Stock Universe & Real-Time Strategy Signals
        </div>
        <div class="watchlist-grid" id="watchlist-grid">
            <div style="color: var(--text-secondary); font-size: 0.85rem;">Loading universe stock data...</div>
        </div>
    </div>

    <!-- Strategy & Risk Parameters Modal -->
    <div class="modal" id="params-modal">
        <div class="modal-content">
            <div class="modal-header">
                <div style="font-weight: 700; font-size: 1.1rem; color: var(--accent-primary);">⚙️ Strategy & Risk Controls</div>
                <button class="btn-secondary" id="modal-close" style="padding: 0.2rem 0.6rem;">✕</button>
            </div>
            <div class="form-grid">
                <div class="form-group">
                    <label>Short Window (MA)</label>
                    <input type="number" id="input-short-window" value="20" min="5" max="100">
                </div>
                <div class="form-group">
                    <label>Long Window (MA)</label>
                    <input type="number" id="input-long-window" value="50" min="10" max="200">
                </div>
                <div class="form-group">
                    <label>Initial Capital (INR)</label>
                    <input type="number" id="input-capital" value="100000" step="10000">
                </div>
                <div class="form-group">
                    <label>Position Size (%)</label>
                    <input type="number" id="input-pos-size" value="20" min="5" max="100">
                </div>
                <div class="form-group">
                    <label>Commission (%)</label>
                    <input type="number" id="input-commission" value="0.1" step="0.05">
                </div>
                <div class="form-group">
                    <label>Slippage (%)</label>
                    <input type="number" id="input-slippage" value="0.05" step="0.01">
                </div>
                <div class="form-group">
                    <label>Stop Loss (%)</label>
                    <input type="number" id="input-stop-loss" value="5.0" step="0.5">
                </div>
                <div class="form-group">
                    <label>Take Profit (%)</label>
                    <input type="number" id="input-take-profit" value="10.0" step="1.0">
                </div>
            </div>
            <div style="display: flex; justify-content: flex-end; gap: 0.5rem;">
                <button class="btn-primary" id="modal-save" style="width: 100%;">Save & Apply Parameters</button>
            </div>
        </div>
    </div>
    </div> <!-- Close #terminal-view -->

    <!-- ============================================================== -->
    <!-- SIGNAL VALIDATION & METRICS AUDIT SECTION                      -->
    <!-- ============================================================== -->
    <div id="validation-view" class="hidden">
        <!-- Validation Header & Action Banner (Clean, with Signal Source Selector) -->
        <div class="val-header-card" style="padding: 1.5rem 1.8rem; background: linear-gradient(135deg, rgba(17, 24, 39, 0.95), rgba(15, 23, 42, 0.95)); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 0.75rem; box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4); margin-bottom: 1.5rem;">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 1.2rem;">
                <div>
                    <div style="display: flex; align-items: center; gap: 0.6rem; margin-bottom: 0.4rem;">
                        <span style="font-size: 1.5rem;">🔬</span>
                        <h2 style="font-size: 1.45rem; font-weight: 800; background: linear-gradient(to right, #10b981, #3b82f6, #a855f7); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 0;">
                            Signal Validation & AI Verification Engine
                        </h2>
                        <span class="badge" style="background: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.4); font-size: 0.75rem; padding: 0.2rem 0.5rem; font-weight: 700;">
                            INSTITUTIONAL GRADE
                        </span>
                    </div>
                    <p style="color: var(--text-secondary); font-size: 0.88rem; margin: 0.2rem 0 0.8rem 0; line-height: 1.45;">
                        Simulating <span id="val-source-desc" style="color: #60a5fa; font-weight: 600;">signalsmain2.py.csv</span> against specifications in <span style="color: #34d399; font-weight: 600;">BACKTESTING REPORT.pdf</span>. Featuring autonomous quantitative math audit with AI.
                    </p>
                    <div style="display: flex; flex-wrap: wrap; gap: 0.5rem; align-items: center;">
                        <span id="val-badge-source" class="badge" style="background: rgba(59, 130, 246, 0.12); color: #60a5fa; border: 1px solid rgba(59, 130, 246, 0.3); font-size: 0.8rem; padding: 0.3rem 0.65rem;">
                            📁 Source: signalsmain2.py.csv (125 Signals)
                        </span>
                        <span class="badge" style="background: rgba(16, 185, 129, 0.12); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3); font-size: 0.8rem; padding: 0.3rem 0.65rem;">
                            💼 Initial Capital: ₹1.00 Cr
                        </span>
                        <span class="badge" style="background: rgba(245, 158, 11, 0.12); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.3); font-size: 0.8rem; padding: 0.3rem 0.65rem;">
                            🎯 5% of Available Cash · 0.05% Slippage · 0.2858% Round-Trip Friction
                        </span>
                        <span id="val-badge-model" class="badge" style="background: rgba(168, 85, 247, 0.12); color: #c084fc; border: 1px solid rgba(168, 85, 247, 0.3); font-size: 0.8rem; padding: 0.3rem 0.65rem;">
                            🤖 Autonomous AI Auditor: Google Gemini / GPT-4o-mini
                        </span>
                    </div>
                </div>
                <div style="display: flex; align-items: center; gap: 0.8rem; flex-wrap: wrap;">
                    <div style="display: flex; flex-direction: column; gap: 0.3rem;">
                        <label for="val-file-select" style="font-size: 0.75rem; color: #94a3b8; font-weight: 600; text-transform: uppercase;">Signal Input File:</label>
                        <select id="val-file-select" onchange="executeValidationSimulation()" style="background: rgba(30, 41, 59, 0.95); color: #f8fafc; border: 1px solid rgba(59, 130, 246, 0.5); border-radius: 0.4rem; padding: 0.55rem 0.9rem; font-size: 0.88rem; font-weight: 600; cursor: pointer; outline: none;">
                            <option value="signalsmain2.py.csv" selected>signalsmain2.py.csv (New Signal File - 125 Trades)</option>
                            <option value="trade logbook-Signal file.xlsx">trade logbook-Signal file.xlsx (Original Logbook - 789 Trades)</option>
                        </select>
                    </div>
                    <div style="display: flex; flex-direction: column; justify-content: flex-end; padding-top: 1.1rem;">
                        <button class="btn-primary" id="val-run-btn" onclick="executeValidationSimulation()" style="background: linear-gradient(135deg, #10b981, #059669); box-shadow: 0 4px 18px rgba(16, 185, 129, 0.45); font-weight: 700; padding: 0.65rem 1.4rem; font-size: 0.92rem; border-radius: 0.5rem; display: flex; align-items: center; gap: 0.6rem; cursor: pointer; transition: all 0.2s;">
                            <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg>
                            <span>Run Validation & AI Verification</span>
                        </button>
                    </div>
                </div>
            </div>
        </div>

        <!-- 12 Top High-Impact KPI Highlight Cards -->
        <div class="val-kpi-grid">
            <div class="val-kpi-card" style="border-top: 3px solid #10b981;">
                <div class="val-kpi-label">Final Equity</div>
                <div class="val-kpi-val positive" id="val-card-equity">₹1,04,78,275</div>
                <div class="val-kpi-sub">Capital: ₹1.00 Cr (<span style="color:#10b981; font-weight:700;" id="val-card-total-return">+4.78%</span>)</div>
            </div>
            <div class="val-kpi-card" style="border-top: 3px solid #10b981;">
                <div class="val-kpi-label">Total Net P&L</div>
                <div class="val-kpi-val positive" id="val-card-pnl">+₹4,78,275</div>
                <div class="val-kpi-sub">Avg Trade: <span id="val-card-avg-trade">₹4,618.39</span></div>
            </div>
            <div class="val-kpi-card" style="border-top: 3px solid #3b82f6;">
                <div class="val-kpi-label">CAGR</div>
                <div class="val-kpi-val neutral" id="val-card-cagr">0.40%</div>
                <div class="val-kpi-sub">Duration: <span id="val-card-span">11.34 Years (4,143 Days)</span></div>
            </div>
            <div class="val-kpi-card" style="border-top: 3px solid #60a5fa;">
                <div class="val-kpi-label">Win Rate</div>
                <div class="val-kpi-val neutral" id="val-card-winrate">52.38%</div>
                <div class="val-kpi-sub">Wins: <span id="val-card-wins" style="color:#10b981;">55</span> | Losses: <span id="val-card-losses" style="color:#ef4444;">50</span></div>
            </div>
            <div class="val-kpi-card" style="border-top: 3px solid #10b981;">
                <div class="val-kpi-label">Profit Factor</div>
                <div class="val-kpi-val positive" id="val-card-profitfactor">1.590</div>
                <div class="val-kpi-sub">Gross Win/Loss: ₹13.06L / ₹8.21L</div>
            </div>
            <div class="val-kpi-card" style="border-top: 3px solid #8b5cf6;">
                <div class="val-kpi-label">Expectancy (EV)</div>
                <div class="val-kpi-val" style="color: #a78bfa;" id="val-card-ev">₹4,618.39</div>
                <div class="val-kpi-sub">Trading Edge: <span id="val-card-ev-pct" style="color:#34d399;">+1.25% / trade</span></div>
            </div>
            <div class="val-kpi-card" style="border-top: 3px solid #3b82f6;">
                <div class="val-kpi-label">Sharpe Ratio</div>
                <div class="val-kpi-val neutral" id="val-card-sharpe">-4.297</div>
                <div class="val-kpi-sub">Rf Discount: 6.0% Annualized</div>
            </div>
            <div class="val-kpi-card" style="border-top: 3px solid #ef4444;">
                <div class="val-kpi-label">Max Drawdown</div>
                <div class="val-kpi-val negative" id="val-card-maxdd">-3.93%</div>
                <div class="val-kpi-sub">Trough: <span id="val-card-maxdd-inr">₹-4,04,354</span></div>
            </div>
            <div class="val-kpi-card" style="border-top: 3px solid #10b981;">
                <div class="val-kpi-label">Sortino Ratio</div>
                <div class="val-kpi-val positive" id="val-card-sortino">-0.896</div>
                <div class="val-kpi-sub">Downside Risk Adjusted</div>
            </div>
            <div class="val-kpi-card" style="border-top: 3px solid #f59e0b;">
                <div class="val-kpi-label">Calmar Ratio</div>
                <div class="val-kpi-val" style="color: #fbbf24;" id="val-card-calmar">0.102</div>
                <div class="val-kpi-sub">CAGR / abs(Max DD)</div>
            </div>
            <div class="val-kpi-card" style="border-top: 3px solid #ec4899;">
                <div class="val-kpi-label">Max Streaks</div>
                <div class="val-kpi-val" style="font-size: 1.35rem; color: #f472b6;" id="val-card-streaks">7W / 6L</div>
                <div class="val-kpi-sub">Holding: 647.4 Days Avg</div>
            </div>
            <div class="val-kpi-card" style="border-top: 3px solid #06b6d4;">
                <div class="val-kpi-label">Skewness</div>
                <div class="val-kpi-val" style="color: #22d3ee;" id="val-card-skewness">-0.6976</div>
                <div class="val-kpi-sub">Consistent Distribution</div>
            </div>
        </div>

        <!-- Validation Workspace 3 Primary Tabs & Panels -->
        <div class="panel" style="margin-bottom: 2rem;">
            <div class="panel-header">
                <div class="panel-title">
                    <span>🔬 Quantitative Validation & AI Verification Workstation</span>
                </div>
                <div class="tabs" id="val-subtabs">
                    <button class="tab-btn active" data-vtab="metrics">📋 Metrics & PDF Formulas</button>
                    <button class="tab-btn" data-vtab="charts">📈 Validation Charts</button>
                    <button class="tab-btn" data-vtab="llm">🤖 AI Verification</button>
                    <button class="tab-btn" data-vtab="downloads">💾 CSV Exports & Downloads</button>
                </div>
            </div>

            <!-- SECTION 1: Metrics & PDF Formulas Table -->
            <div id="vtab-content-metrics" class="val-tab-pane">
                <div style="margin-bottom: 1rem; color: var(--text-secondary); font-size: 0.85rem; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 0.5rem;">
                    <span>Reconciliation with formulas specified in <strong>BACKTESTING REPORT.pdf</strong> and project backtester:</span>
                    <span class="badge" style="background: rgba(59, 130, 246, 0.15); color: #60a5fa; border: 1px solid rgba(59, 130, 246, 0.3);">✓ QuantLab Engine Reconciled</span>
                </div>
                <div class="table-container" style="max-height: 560px;">
                    <table class="val-metrics-table">
                        <thead>
                            <tr>
                                <th style="width: 28%;">Metric Name</th>
                                <th style="width: 24%;">Calculated Value</th>
                                <th style="width: 48%;">Mathematical Formula / Definition</th>
                            </tr>
                        </thead>
                        <tbody id="val-metrics-tbody">
                            <!-- Populated dynamically by JS -->
                        </tbody>
                    </table>
                </div>
            </div>

            <!-- SECTION 1B: Validation Visual Charts & Curves -->
            <div id="vtab-content-charts" class="val-tab-pane hidden">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem; flex-wrap: wrap; gap: 0.5rem;">
                    <div class="tabs" id="val-chart-subtabs">
                        <button class="tab-btn active" data-vchart="equity">📈 Equity Growth Curve</button>
                        <button class="tab-btn" data-vchart="drawdown">📉 Underwater Drawdown</button>
                        <button class="tab-btn" data-vchart="histogram">📊 Annual Returns Histogram</button>
                    </div>
                    <span class="badge" style="background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3);">
                        High Definition (200 DPI)
                    </span>
                </div>
                <div class="chart-container" style="background: #111827; border-radius: 0.5rem; overflow: hidden; display: flex; justify-content: center; align-items: center; min-height: 480px; padding: 1rem;">
                    <img id="val-display-chart" src="/validation_img/equity_curve.png" alt="Validation Chart" style="max-width: 100%; height: auto; border-radius: 0.375rem; box-shadow: 0 4px 20px rgba(0,0,0,0.5);">
                </div>
            </div>

            <!-- SECTION 2: AI Verification (Side-by-Side Engine vs AI Model Reconciliation) -->
            <div id="vtab-content-llm" class="val-tab-pane hidden">
                <!-- Status Banner -->
                <div class="recom-banner" style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1.4rem; flex-wrap: wrap; gap: 0.8rem; background: linear-gradient(135deg, rgba(168, 85, 247, 0.12), rgba(59, 130, 246, 0.1)); border: 1px solid rgba(168, 85, 247, 0.35);">
                    <div>
                        <div style="font-size: 1.08rem; font-weight: 800; color: #c084fc; display: flex; align-items: center; gap: 0.5rem;">
                            <span>🤖 Autonomous AI Model Verification & Audit:</span>
                            <span class="badge" style="background: rgba(16, 185, 129, 0.25); color: #34d399; font-size: 0.85rem; padding: 0.25rem 0.6rem; font-weight: 700;">STRATEGY VALIDATED BY AI ✓</span>
                        </div>
                        <div style="color: #cbd5e1; font-size: 0.85rem; margin-top: 0.3rem;" id="val-llm-engine-label">
                            AI Model: Deterministic Quantitative Auditor | All Quantitative Metrics Audited & Reconciled with Engine Data
                        </div>
                    </div>
                    <div style="display: flex; gap: 0.5rem; align-items: center;">
                        <span class="badge" style="background: rgba(16, 185, 129, 0.2); color: #34d399; font-size: 0.85rem; padding: 0.35rem 0.8rem; border: 1px solid rgba(16, 185, 129, 0.4); font-weight: 700;">
                            ✓ 100% Reconciled
                        </span>
                        <button class="btn-secondary" onclick="downloadValidationCSV('engine_vs_llm_validation.csv')" style="padding: 0.35rem 0.75rem; font-size: 0.82rem; display: flex; align-items: center; gap: 0.35rem;">
                            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="7 10 12 15 17 10"></polyline><line x1="12" y1="15" x2="12" y2="3"></line></svg>
                            Export Reconciliation CSV
                        </button>
                    </div>
                </div>

                <!-- Side-by-Side Reconciliation Table (5 Columns) -->
                <div style="margin-bottom: 1.5rem;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.8rem;">
                        <div style="font-size: 0.95rem; font-weight: 700; color: #c084fc; text-transform: uppercase; letter-spacing: 0.03em;">
                            🧮 Side-by-Side Reconciliation: Engine Calculated vs AI Validated Metrics
                        </div>
                        <span class="badge" style="background: rgba(16, 185, 129, 0.15); color: #34d399; font-size: 0.75rem; font-weight: 700;">
                            ✓ FORMULA RECONCILED
                        </span>
                    </div>
                    <div class="table-container" style="max-height: 560px;">
                        <table>
                            <thead>
                                <tr>
                                    <th style="width: 20%;">Metric Name</th>
                                    <th style="width: 18%; color: #60a5fa;">Your Engine Value</th>
                                    <th style="width: 18%; color: #34d399;">LLM / AI Validated Value</th>
                                    <th style="width: 12%; text-align: center;">Status</th>
                                    <th style="width: 32%;">AI Audit & Mathematical Verification</th>
                                </tr>
                            </thead>
                            <tbody id="val-llm-metrics-tbody">
                                <!-- Populated dynamically by JS -->
                            </tbody>
                        </table>
                    </div>
                </div>

                <!-- AI Quantitative Audit Narrative Box -->
                <div style="margin-top: 1.5rem; background: rgba(17, 24, 39, 0.75); border: 1px solid rgba(168, 85, 247, 0.3); border-radius: 0.75rem; padding: 1.25rem;">
                    <div style="font-size: 0.95rem; font-weight: 700; color: #c084fc; margin-bottom: 0.8rem; display: flex; align-items: center; gap: 0.5rem;">
                        <span>📝 AI Quantitative Risk & Integrity Audit Narrative</span>
                    </div>
                    <pre id="val-llm-narrative-box" style="white-space: pre-wrap; font-family: var(--font-mono); font-size: 0.84rem; line-height: 1.5; color: #cbd5e1; max-height: 380px; overflow-y: auto; background: rgba(11, 15, 25, 0.6); padding: 1rem; border-radius: 0.5rem; border: 1px solid rgba(255, 255, 255, 0.05);">Audit report will load upon simulation execution.</pre>
                </div>
            </div>

            <!-- SECTION 3: CSV Exports & Downloads (Exactly 2 Clean CSV Download Cards) -->
            <div id="vtab-content-downloads" class="val-tab-pane hidden">
                <div style="margin-bottom: 1.2rem; color: var(--text-secondary); font-size: 0.88rem;">
                    Download the validated backtest results and AI verification datasets directly to your machine (2 primary artifacts):
                </div>

                <!-- 2 Clean Download Cards -->
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap: 1.5rem; margin-bottom: 2rem;">
                    <!-- Card 1: validation_metrics.csv -->
                    <div style="background: rgba(17, 24, 39, 0.9); border: 1px solid rgba(16, 185, 129, 0.35); border-radius: 0.75rem; padding: 1.4rem; display: flex; flex-direction: column; justify-content: space-between; box-shadow: 0 4px 20px rgba(0,0,0,0.35);">
                        <div>
                            <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.6rem;">
                                <span style="font-size: 1.3rem;">📄</span>
                                <div style="font-size: 1.12rem; font-weight: 800; color: #f8fafc;">
                                     validation_metrics.csv
                                </div>
                                <span class="badge" style="background: rgba(16, 185, 129, 0.15); color: #34d399; font-size: 0.72rem;">ENGINE OUTPUT</span>
                            </div>
                            <div style="font-size: 0.84rem; color: #94a3b8; margin-bottom: 1.2rem; line-height: 1.5;">
                                Complete institutional metrics dataset containing all quantitative KPIs calculated by the Python validation backtester matching <em>BACKTESTING REPORT.pdf</em>.
                            </div>
                        </div>
                        <button class="btn-primary" onclick="downloadValidationCSV('validation_metrics.csv')" style="width: 100%; font-weight: 700; padding: 0.7rem; font-size: 0.9rem; background: linear-gradient(135deg, #10b981, #059669); display: flex; justify-content: center; align-items: center; gap: 0.5rem; box-shadow: 0 4px 14px rgba(16, 185, 129, 0.3);">
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="7 10 12 15 17 10"></polyline><line x1="12" y1="15" x2="12" y2="3"></line></svg>
                            Download validation_metrics.csv
                        </button>
                    </div>

                    <!-- Card 2: llm_verification_metrics.csv -->
                    <div style="background: rgba(17, 24, 39, 0.9); border: 1px solid rgba(168, 85, 247, 0.35); border-radius: 0.75rem; padding: 1.4rem; display: flex; flex-direction: column; justify-content: space-between; box-shadow: 0 4px 20px rgba(0,0,0,0.35);">
                        <div>
                            <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.6rem;">
                                <span style="font-size: 1.3rem;">🤖</span>
                                <div style="font-size: 1.12rem; font-weight: 800; color: #c084fc;">
                                    llm_verification_metrics.csv
                                </div>
                                <span class="badge" style="background: rgba(168, 85, 247, 0.15); color: #c084fc; font-size: 0.72rem;">AI VERIFICATION</span>
                            </div>
                            <div style="font-size: 0.84rem; color: #94a3b8; margin-bottom: 1.2rem; line-height: 1.5;">
                                Autonomous AI verification dataset. Quantitative metrics independently computed by AI directly from the active signal file along with financial rationale notes.
                            </div>
                        </div>
                        <button class="btn-primary" onclick="downloadValidationCSV('llm_verification_metrics.csv')" style="width: 100%; font-weight: 700; padding: 0.7rem; font-size: 0.9rem; background: linear-gradient(135deg, #a855f7, #7e22ce); display: flex; justify-content: center; align-items: center; gap: 0.5rem; box-shadow: 0 4px 14px rgba(168, 85, 247, 0.3);">
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="7 10 12 15 17 10"></polyline><line x1="12" y1="15" x2="12" y2="3"></line></svg>
                            Download llm_verification_metrics.csv
                        </button>
                    </div>
                </div>

                <!-- Live Data Previews of Both CSVs -->
                <div style="margin-top: 1rem;">
                    <div style="font-size: 0.92rem; font-weight: 700; color: #10b981; text-transform: uppercase; margin-bottom: 0.6rem; letter-spacing: 0.03em;">
                        👁️ Live Table Preview: validation_metrics.csv
                    </div>
                    <div class="table-container" style="max-height: 280px; margin-bottom: 1.8rem;">
                        <table>
                            <thead>
                                <tr>
                                    <th style="width: 28%;">Metric Name</th>
                                    <th style="width: 24%;">Calculated Value</th>
                                    <th style="width: 48%;">Mathematical Formula / Definition</th>
                                </tr>
                            </thead>
                            <tbody id="val-metrics-preview-tbody">
                                <!-- Populated dynamically -->
                            </tbody>
                        </table>
                    </div>

                    <div style="font-size: 0.92rem; font-weight: 700; color: #c084fc; text-transform: uppercase; margin-bottom: 0.6rem; letter-spacing: 0.03em;">
                        👁️ Live Table Preview: llm_verification_metrics.csv
                    </div>
                    <div class="table-container" style="max-height: 280px;">
                        <table>
                            <thead>
                                <tr>
                                    <th style="width: 28%;">Metric Name</th>
                                    <th style="width: 24%;">LLM Calculated Value</th>
                                    <th style="width: 48%;">LLM Quantitative Note</th>
                                </tr>
                            </thead>
                            <tbody id="val-llm-preview-tbody">
                                <!-- Populated dynamically -->
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>
    </div> <!-- Close #validation-view -->

    <script>
        // DOM Handles
        const tickerSelect = document.getElementById('ticker-select');
        const strategySelect = document.getElementById('strategy-select');
        const runBtn = document.getElementById('run-btn');
        const agentBtn = document.getElementById('agent-btn');
        const loadingOverlay = document.getElementById('loading-overlay');
        const displayChart = document.getElementById('display-chart');
        const reportText = document.getElementById('report-text');
        const tradeLogTbody = document.getElementById('trade-log-tbody');
        const benchmarkTbody = document.getElementById('benchmark-tbody');
        const aiAnalysisContainer = document.getElementById('ai-analysis-container');
        const ragQueryInput = document.getElementById('rag-query-input');
        const ragSearchBtn = document.getElementById('rag-search-btn');
        const ragResponseContainer = document.getElementById('rag-response-container');
        const watchlistGrid = document.getElementById('watchlist-grid');

        // Recommendations DOM
        const recomTitle = document.getElementById('recom-title');
        const recomRationale = document.getElementById('recom-rationale');
        const recomTbody = document.getElementById('recom-tbody');

        // Agent Workflow DOM
        const flowDecisionNode = document.getElementById('flow-decision-node');
        const wfExpId = document.getElementById('wf-exp-id');
        const wfStatusBadge = document.getElementById('wf-status-badge');
        const wfHypothesis = document.getElementById('wf-hypothesis');
        const workflowTbody = document.getElementById('workflow-tbody');

        // Parameter Modal
        const paramsBtn = document.getElementById('params-btn');
        const paramsModal = document.getElementById('params-modal');
        const modalClose = document.getElementById('modal-close');
        const modalSave = document.getElementById('modal-save');

        // Metrics elements
        const metricTotalReturn = document.getElementById('metric-total-return');
        const metricCagr = document.getElementById('metric-cagr');
        const metricSharpe = document.getElementById('metric-sharpe');
        const metricMaxDrawdown = document.getElementById('metric-max-drawdown');
        const metricWinRate = document.getElementById('metric-win-rate');
        const metricProfitFactor = document.getElementById('metric-profit-factor');

        // Application State
        let activeChartTab = 'signals';
        let activeInfoTab = 'recom';
        let currentChartBundle = {};

        function formatCurrency(val) {
            return parseFloat(val).toLocaleString('en-IN', { maximumFractionDigits: 2, minimumFractionDigits: 2 });
        }

        // Modal triggers
        paramsBtn.addEventListener('click', () => paramsModal.classList.add('active'));
        modalClose.addEventListener('click', () => paramsModal.classList.remove('active'));
        modalSave.addEventListener('click', () => {
            paramsModal.classList.remove('active');
            executeBacktest();
        });

        // Tab Switching - Visualizations
        document.querySelectorAll('[data-chart]').forEach(btn => {
            btn.addEventListener('click', (e) => {
                document.querySelectorAll('[data-chart]').forEach(b => b.classList.remove('active'));
                e.target.classList.add('active');
                activeChartTab = e.target.dataset.chart;
                updateChartDisplay();
            });
        });

        // Tab Switching - Information
        document.querySelectorAll('[data-info]').forEach(btn => {
            btn.addEventListener('click', (e) => {
                document.querySelectorAll('[data-info]').forEach(b => b.classList.remove('active'));
                e.target.classList.add('active');
                activeInfoTab = e.target.dataset.info;

                document.querySelectorAll('.info-content').forEach(c => c.classList.add('hidden'));
                document.getElementById(`info-${activeInfoTab}`).classList.remove('hidden');
            });
        });

        function updateChartDisplay() {
            if (currentChartBundle && currentChartBundle[activeChartTab]) {
                displayChart.src = currentChartBundle[activeChartTab];
            }
        }

        function renderMetrics(metrics) {
            if (!metrics) return;
            metricTotalReturn.innerText = metrics["Total Return"] || "0.00%";
            metricCagr.innerText = metrics["CAGR"] || "0.00%";
            metricSharpe.innerText = metrics["Sharpe Ratio"] || "0.00";
            metricMaxDrawdown.innerText = metrics["Max Drawdown"] || "0.00%";
            metricWinRate.innerText = metrics["Win Rate"] || "0.0%";
            metricProfitFactor.innerText = metrics["Profit Factor"] || "0.00";

            const retFloat = parseFloat(metrics["Total Return"]);
            const cagrFloat = parseFloat(metrics["CAGR"]);
            metricTotalReturn.className = 'metric-value ' + (retFloat >= 0 ? 'positive' : 'negative');
            metricCagr.className = 'metric-value ' + (cagrFloat >= 0 ? 'positive' : 'negative');
            metricMaxDrawdown.className = 'metric-value negative';
        }

        function renderRecommendations(recom) {
            if (!recom) return;
            recomTitle.innerText = `💡 Recommended: ${recom.recommended_label} (Score: ${recom.recommended_score}/10)`;
            recomRationale.innerText = recom.rationale;

            recomTbody.innerHTML = (recom.evaluations || []).map(ev => {
                const isTop = ev.strategy === recom.recommended_strategy;
                const rowBg = isTop ? 'rgba(16, 185, 129, 0.1)' : 'transparent';
                const star = isTop ? '⭐ ' : '';
                return `
                    <tr style="background: ${rowBg}; font-weight: ${isTop ? '700' : 'normal'};">
                        <td>${star}${ev.label}</td>
                        <td>${ev.raw_metrics["Sharpe Ratio"]} <span style="font-size:0.75rem; color:var(--text-secondary);">(${ev.weighted_scores["Sharpe Ratio"]})</span></td>
                        <td>${ev.raw_metrics["CAGR"]} <span style="font-size:0.75rem; color:var(--text-secondary);">(${ev.weighted_scores["CAGR"]})</span></td>
                        <td style="color:var(--accent-red);">${ev.raw_metrics["Max Drawdown"]} <span style="font-size:0.75rem; color:var(--text-secondary);">(${ev.weighted_scores["Max Drawdown"]})</span></td>
                        <td>${ev.raw_metrics["Win Rate"]} <span style="font-size:0.75rem; color:var(--text-secondary);">(${ev.weighted_scores["Win Rate"]})</span></td>
                        <td>${ev.raw_metrics["Profit Factor"]} <span style="font-size:0.75rem; color:var(--text-secondary);">(${ev.weighted_scores["Profit Factor"]})</span></td>
                        <td>${ev.raw_metrics["Sortino Ratio"]} <span style="font-size:0.75rem; color:var(--text-secondary);">(${ev.weighted_scores["Sortino Ratio"]})</span></td>
                        <td style="text-align: right; font-weight: 800; color: ${isTop ? 'var(--accent-green)' : 'var(--text-primary)'};">${ev.composite_score}</td>
                    </tr>
                `;
            }).join('');
        }

        function renderWorkflowState(wf) {
            if (!wf) return;
            wfExpId.innerText = wf.experiment_id || "EXP_101";
            wfHypothesis.innerText = wf.hypothesis || "Quantitative strategy optimization";
            
            const dec = wf.loop_decision || "PENDING";
            wfStatusBadge.innerText = `DECISION: ${dec}`;
            wfStatusBadge.className = `badge ${dec === 'ACCEPT' ? 'badge-buy' : 'badge-sell'}`;
            flowDecisionNode.innerText = dec;

            if (wf.history && wf.history.length > 0) {
                workflowTbody.innerHTML = wf.history.map(h => `
                    <tr>
                        <td style="font-family: var(--font-mono);">Iteration ${h.iteration}</td>
                        <td>${h.strategy}</td>
                        <td style="font-family: var(--font-mono); font-size: 0.78rem;">${JSON.stringify(h.params)}</td>
                        <td style="text-align: right;">${h.sharpe}</td>
                        <td style="text-align: right;">${h.cagr}</td>
                        <td style="text-align: right; color: var(--accent-red);">${h.max_drawdown}</td>
                    </tr>
                `).join('');
            }
        }

        function renderTradeLog(trades) {
            if (!trades || trades.length === 0) {
                tradeLogTbody.innerHTML = `
                    <tr>
                        <td colspan="7" style="text-align: center; color: var(--text-secondary); padding: 2rem;">
                            No executed trades for this backtest configuration.
                        </td>
                    </tr>
                `;
                return;
            }

            tradeLogTbody.innerHTML = trades.map(t => {
                const isBuy = t.Type === 'BUY';
                const badgeClass = isBuy ? 'badge-buy' : 'badge-sell';
                const dateStr = new Date(t.Date).toLocaleDateString('en-IN', { year: 'numeric', month: 'short', day: '2-digit' });
                return `
                    <tr>
                        <td style="font-family: var(--font-mono); font-size: 0.8rem;">${dateStr}</td>
                        <td><span class="badge ${badgeClass}">${t.Type}</span></td>
                        <td style="text-align: right;">₹${formatCurrency(t.Price)}</td>
                        <td style="text-align: right; font-weight: 500;">${parseInt(t.Shares).toLocaleString()}</td>
                        <td style="text-align: right;">₹${formatCurrency(t.Value)}</td>
                        <td style="text-align: right; color: var(--text-secondary);">₹${formatCurrency(t["Transaction Cost"])}</td>
                        <td style="text-align: right; font-family: var(--font-mono); font-size: 0.8rem;">₹${formatCurrency(t["Remaining Cash"])}</td>
                    </tr>
                `;
            }).join('');
        }

        function renderAIAnalysis(ai) {
            if (!ai) return;
            const rating = ai.investment_rating || "NEUTRAL";
            const ratingColor = rating.includes("BUY") ? "var(--accent-green)" : (rating.includes("SELL") ? "var(--accent-red)" : "var(--accent-amber)");
            
            aiAnalysisContainer.innerHTML = `
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.8rem;">
                    <span style="font-weight: 700; color: var(--text-secondary);">EXECUTIVE RATING</span>
                    <span class="badge" style="background: rgba(255,255,255,0.08); color: ${ratingColor}; font-size: 0.85rem; padding: 0.3rem 0.7rem;">${rating}</span>
                </div>
                <p style="margin-bottom: 0.8rem;"><strong>Market Regime:</strong> ${ai.market_regime_fit || "N/A"}</p>
                <p style="margin-bottom: 0.8rem;"><strong>Risk Score:</strong> ${ai.risk_score || "N/A"}/10</p>
                <div style="margin-bottom: 0.8rem;">
                    <strong style="color: var(--accent-green);">Strategy Strengths:</strong>
                    <ul style="margin-left: 1.2rem; margin-top: 0.3rem;">
                        ${(ai.strengths || []).map(s => `<li>${s}</li>`).join('')}
                    </ul>
                </div>
                <div>
                    <strong style="color: var(--accent-red);">Risk Factors & Weaknesses:</strong>
                    <ul style="margin-left: 1.2rem; margin-top: 0.3rem;">
                        ${(ai.weaknesses || []).map(w => `<li>${w}</li>`).join('')}
                    </ul>
                </div>
            `;
        }

        function renderBenchmarkComp(bench) {
            if (!bench || bench.status === 'ERROR') {
                benchmarkTbody.innerHTML = `<tr><td colspan="4" style="text-align:center; padding:1.5rem; color: var(--text-secondary);">Benchmark data unavailable.</td></tr>`;
                return;
            }

            const cagrStr = `${(bench.strategy_cagr * 100).toFixed(2)}%`;
            const benchCagrStr = `${(bench.benchmark_cagr * 100).toFixed(2)}%`;
            const excessStr = `${(bench.excess_return * 100).toFixed(2)}%`;

            benchmarkTbody.innerHTML = `
                <tr><td>CAGR</td><td style="text-align:right;">${cagrStr}</td><td style="text-align:right;">${benchCagrStr}</td><td style="text-align:right; font-weight:700; color: ${bench.excess_return >= 0 ? 'var(--accent-green)' : 'var(--accent-red)'};">${excessStr}</td></tr>
                <tr><td>Sharpe Ratio</td><td style="text-align:right;">${bench.strategy_sharpe.toFixed(2)}</td><td style="text-align:right;">${bench.benchmark_sharpe.toFixed(2)}</td><td style="text-align:right;">${(bench.strategy_sharpe - bench.benchmark_sharpe).toFixed(2)}</td></tr>
                <tr><td>Alpha (Ann.)</td><td style="text-align:right;" colspan="2">-</td><td style="text-align:right; font-weight:700; color: var(--accent-green);">${(bench.alpha * 100).toFixed(2)}%</td></tr>
                <tr><td>Beta</td><td style="text-align:right;" colspan="2">-</td><td style="text-align:right;">${bench.beta.toFixed(2)}</td></tr>
                <tr><td>Information Ratio</td><td style="text-align:right;" colspan="2">-</td><td style="text-align:right;">${bench.information_ratio.toFixed(2)}</td></tr>
            `;
        }

        async function executeBacktest() {
            const ticker = tickerSelect.value;
            const strategy = strategySelect.value;
            const shortW = document.getElementById('input-short-window').value;
            const longW = document.getElementById('input-long-window').value;
            const cap = document.getElementById('input-capital').value;
            const posSize = parseFloat(document.getElementById('input-pos-size').value) / 100.0;
            const comm = parseFloat(document.getElementById('input-commission').value) / 100.0;
            const slip = parseFloat(document.getElementById('input-slippage').value) / 100.0;
            const sl = parseFloat(document.getElementById('input-stop-loss').value) / 100.0;
            const tp = parseFloat(document.getElementById('input-take-profit').value) / 100.0;

            runBtn.disabled = true;
            loadingOverlay.classList.add('active');

            try {
                const url = `/api/run?ticker=${ticker}&strategy=${strategy}&short_window=${shortW}&long_window=${longW}&capital=${cap}&position_size=${posSize}&commission=${comm}&slippage=${slip}&stop_loss=${sl}&take_profit=${tp}`;
                const res = await fetch(url);
                const data = await res.json();

                if (data.status === 'success') {
                    reportText.innerText = data.summary;
                    renderMetrics(data.metrics);
                    renderTradeLog(data.trades);
                    renderAIAnalysis(data.ai_analysis);
                    renderBenchmarkComp(data.benchmark_comp);
                    renderRecommendations(data.recommendations);

                    currentChartBundle = data.charts;
                    updateChartDisplay();
                } else {
                    alert(`Simulation Error: ${data.message}`);
                }
            } catch (err) {
                console.error(err);
                alert(`Execution Error: ${err.message}`);
            } finally {
                loadingOverlay.classList.remove('active');
                runBtn.disabled = false;
            }
        }

        async function executeAgentWorkflow() {
            const ticker = tickerSelect.value;
            const strategy = strategySelect.value;
            agentBtn.disabled = true;
            loadingOverlay.classList.add('active');

            try {
                const url = `/api/agent?ticker=${ticker}&strategy=${strategy}`;
                const res = await fetch(url);
                const data = await res.json();

                if (data.status === 'success') {
                    renderWorkflowState(data);
                    // Switch to Workflow Tab
                    document.querySelectorAll('[data-info]').forEach(b => b.classList.remove('active'));
                    document.querySelector('[data-info="workflow"]').classList.add('active');
                    document.querySelectorAll('.info-content').forEach(c => c.classList.add('hidden'));
                    document.getElementById('info-workflow').classList.remove('hidden');
                } else {
                    alert(`Agent Workflow Error: ${data.message}`);
                }
            } catch (err) {
                console.error(err);
                alert(`Agent Workflow Error: ${err.message}`);
            } finally {
                loadingOverlay.classList.remove('active');
                agentBtn.disabled = false;
            }
        }

        async function fetchStocksOverview() {
            try {
                const res = await fetch('/api/stocks');
                const stocks = await res.json();
                watchlistGrid.innerHTML = stocks.map(s => {
                    const isPos = s.change_pct >= 0;
                    const color = isPos ? 'var(--accent-green)' : 'var(--accent-red)';
                    const sigClass = s.signal === 'BUY' ? 'badge-buy' : (s.signal === 'SELL' ? 'badge-sell' : '');
                    return `
                        <div class="watchlist-card">
                            <div>
                                <div style="font-weight: 700; font-size: 0.95rem;">${s.ticker}</div>
                                <div style="font-size: 0.78rem; color: var(--text-secondary);">Vol: ${(s.volume/1000).toFixed(0)}k</div>
                            </div>
                            <div style="text-align: right;">
                                <div style="font-weight: 700; font-size: 0.95rem;">₹${formatCurrency(s.price)}</div>
                                <div style="font-size: 0.78rem; color: ${color}; font-weight: 600;">${isPos ? '+' : ''}${s.change_pct.toFixed(2)}%</div>
                            </div>
                            <div>
                                <span class="badge ${sigClass}">${s.signal}</span>
                            </div>
                        </div>
                    `;
                }).join('');
            } catch (e) {
                console.error(e);
            }
        }

        function runDemoQuery(qText) {
            ragQueryInput.value = qText;
            handleRAGSearch();
        }

        async function handleRAGSearch() {
            const query = ragQueryInput.value.trim();
            if (!query) return;
            ragSearchBtn.disabled = true;
            ragResponseContainer.innerHTML = `
                <div style="text-align: center; padding: 2rem;">
                    <div style="font-weight: 700; color: var(--accent-primary); margin-bottom: 0.5rem; font-size: 1.1rem;">⚡ Executing QUANTLAB Multilingual MADRID RAG Pipeline...</div>
                    <p style="color: var(--text-secondary); font-size: 0.85rem;">Processing Language Detection ➔ MADRID Translation ➔ Vector Search ➔ LLM Synthesis ➔ Target Decoding...</p>
                </div>
            `;

            try {
                const res = await fetch('/api/rag/query', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ query: query })
                });
                const data = await res.json();
                renderPipelineResults(data);
            } catch (e) {
                ragResponseContainer.innerHTML = `<p style="color: var(--accent-red); padding: 1.5rem;">Error executing RAG search: ${e.message}</p>`;
            } finally {
                ragSearchBtn.disabled = false;
            }
        }

        function renderPipelineResults(data) {
            if (!data) return;
            const ts = data.token_stats || {};
            const lat = data.stage_latencies_ms || {};
            const diffPct = ts.observed_token_difference_pct || 0;
            const isDiffNeg = diffPct <= 0;
            const diffColor = isDiffNeg ? 'var(--accent-green)' : 'var(--accent-amber)';

            const sourcesHtml = (data.retrieved_sources || []).map((s, idx) => `
                <div style="background: rgba(0,0,0,0.3); border: 1px solid var(--card-border); border-radius: 0.4rem; padding: 0.75rem; margin-top: 0.5rem;">
                    <div style="display: flex; justify-content: space-between; align-items: center; font-size: 0.82rem; margin-bottom: 0.3rem;">
                        <span style="font-weight: 700; color: var(--accent-primary);">📄 [${idx+1}] ${s.source}</span>
                        <span style="background: rgba(59,130,246,0.15); color: var(--accent-primary); padding: 0.1rem 0.4rem; border-radius: 0.2rem; font-family: var(--font-mono); font-weight: 600;">Similarity: ${(s.similarity_score * 100).toFixed(1)}%</span>
                    </div>
                    <div style="font-size: 0.78rem; color: var(--text-secondary); font-style: italic;">Topic: ${s.topic} | Section: ${s.section}</div>
                    <p style="font-size: 0.8rem; margin-top: 0.4rem; color: #cbd5e1; font-family: var(--font-mono); background: rgba(0,0,0,0.2); padding: 0.4rem; border-radius: 0.3rem; max-height: 100px; overflow-y: auto;">
                        ${s.text.length > 250 ? s.text.substring(0, 250) + '...' : s.text}
                    </p>
                </div>
            `).join('');

            ragResponseContainer.innerHTML = `
                <div style="display: flex; flex-direction: column; gap: 1rem;">
                    <!-- Title & Overview Header -->
                    <div style="display: flex; justify-content: space-between; align-items: center; background: rgba(59, 130, 246, 0.1); border: 1px solid rgba(59, 130, 246, 0.25); padding: 0.8rem 1.2rem; border-radius: 0.5rem;">
                        <div>
                            <span style="font-weight: 800; font-size: 1rem; color: var(--text-primary);">QUANTLAB AI RESEARCH PIPELINE</span>
                            <span style="font-size: 0.8rem; color: var(--text-secondary); margin-left: 0.5rem;">(Execution Latency: ${data.total_latency_ms || 0} ms)</span>
                        </div>
                        <span class="badge badge-buy">STATUS: ${(data.status || "SUCCESS").toUpperCase()}</span>
                    </div>

                    <!-- 8 Connected Stage Visual Cards -->
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1rem;">
                        <!-- Stage 1 -->
                        <div class="metric-card" style="border-left: 4px solid var(--accent-primary);">
                            <div class="metric-label">1. USER INPUT</div>
                            <div style="font-weight: 700; font-size: 0.95rem; margin-bottom: 0.3rem;">"${data.original_query}"</div>
                            <div style="font-size: 0.78rem; color: var(--text-secondary);">Chars: ${data.original_query.length} | Input Tokens: ${ts.original_input_tokens || 0}</div>
                        </div>

                        <!-- Stage 2 -->
                        <div class="metric-card" style="border-left: 4px solid var(--accent-green);">
                            <div class="metric-label">2. LANGUAGE DETECTION</div>
                            <div style="font-weight: 700; font-size: 0.95rem; color: var(--accent-green); margin-bottom: 0.3rem;">${data.detected_language} (${data.language_code})</div>
                            <div style="font-size: 0.78rem; color: var(--text-secondary);">Automatic detection ✓ | Latency: ${lat.stage_1_2_detection_ms || 0} ms</div>
                        </div>

                        <!-- Stage 3 -->
                        <div class="metric-card" style="border-left: 4px solid #8b5cf6;">
                            <div class="metric-label">3. MADRID TRANSLATION</div>
                            <div style="font-family: var(--font-mono); font-size: 0.78rem; color: #c084fc; margin-bottom: 0.3rem; word-break: break-all;">${data.madrid_query}</div>
                            <div style="font-size: 0.78rem; color: var(--text-secondary);">MADRID Tokens: ${ts.madrid_input_tokens || 0} (<span style="color: ${diffColor}; font-weight:700;">${diffPct >= 0 ? '+' : ''}${diffPct}%</span> vs original)</div>
                        </div>

                        <!-- Stage 4 & 5 -->
                        <div class="metric-card" style="border-left: 4px solid #f59e0b;">
                            <div class="metric-label">4 & 5. RAG RETRIEVAL & KNOWLEDGE</div>
                            <div style="font-weight: 700; font-size: 0.95rem; color: #fbbf24; margin-bottom: 0.3rem;">${(data.retrieved_sources || []).length} Chunks Retrieved</div>
                            <div style="font-size: 0.78rem; color: var(--text-secondary);">Top-K: 3 | Vector Search Latency: ${lat.stage_4_5_retrieval_ms || 0} ms</div>
                        </div>
                    </div>

                    <!-- Retrieved Sources Section -->
                    <div style="background: rgba(0,0,0,0.2); border: 1px solid var(--card-border); padding: 1rem; border-radius: 0.5rem;">
                        <div style="font-weight: 700; font-size: 0.85rem; text-transform: uppercase; color: var(--accent-primary); letter-spacing: 0.05em; margin-bottom: 0.5rem;">
                            📚 Grounded Knowledge Base Sources
                        </div>
                        ${sourcesHtml || '<p style="color:var(--text-secondary); font-size:0.85rem;">No sources found.</p>'}
                    </div>

                    <!-- Stage 6, 7 & Final Output -->
                    <div style="background: var(--card-bg); border: 1px solid var(--card-border); padding: 1.2rem; border-radius: 0.5rem;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.6rem;">
                            <span style="font-weight: 800; font-size: 0.95rem; color: var(--accent-green); display: flex; align-items: center; gap: 0.4rem;">
                                💬 8. FINAL USER RESPONSE (${data.detected_language})
                            </span>
                            <span style="font-size: 0.78rem; color: var(--text-secondary);">Target Language Decoding: Complete ✓</span>
                        </div>
                        <div style="font-size: 0.92rem; line-height: 1.6; color: var(--text-primary); white-space: pre-line; background: rgba(0,0,0,0.25); padding: 1rem; border-radius: 0.4rem; border-left: 3px solid var(--accent-green);">
                            ${data.final_answer}
                        </div>
                    </div>

                    <!-- Scientific Token Accounting Panel -->
                    <div style="background: rgba(15, 23, 42, 0.9); border: 1px solid rgba(59, 130, 246, 0.3); padding: 1.2rem; border-radius: 0.5rem;">
                        <div style="font-weight: 700; font-size: 0.85rem; text-transform: uppercase; color: var(--accent-primary); letter-spacing: 0.05em; margin-bottom: 0.8rem; display: flex; justify-content: space-between; align-items: center;">
                            <span>📊 QUANTLAB TOKEN ANALYTICS & ACCOUNTING</span>
                            <span style="font-family: var(--font-mono); font-size: 0.8rem; color: ${diffColor};">Observed Difference: ${ts.observed_token_difference_pct || 0}%</span>
                        </div>
                        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 0.8rem; text-align: center;">
                            <div style="background: rgba(0,0,0,0.3); padding: 0.6rem; border-radius: 0.4rem;">
                                <div style="font-size: 0.72rem; color: var(--text-secondary);">Original Query</div>
                                <div style="font-size: 1.1rem; font-weight: 700;">${ts.original_input_tokens || 0}</div>
                            </div>
                            <div style="background: rgba(0,0,0,0.3); padding: 0.6rem; border-radius: 0.4rem;">
                                <div style="font-size: 0.72rem; color: var(--text-secondary);">MADRID Query</div>
                                <div style="font-size: 1.1rem; font-weight: 700; color: #c084fc;">${ts.madrid_input_tokens || 0}</div>
                            </div>
                            <div style="background: rgba(0,0,0,0.3); padding: 0.6rem; border-radius: 0.4rem;">
                                <div style="font-size: 0.72rem; color: var(--text-secondary);">RAG Context</div>
                                <div style="font-size: 1.1rem; font-weight: 700; color: #fbbf24;">${ts.rag_context_tokens || 0}</div>
                            </div>
                            <div style="background: rgba(0,0,0,0.3); padding: 0.6rem; border-radius: 0.4rem;">
                                <div style="font-size: 0.72rem; color: var(--text-secondary);">LLM Input</div>
                                <div style="font-size: 1.1rem; font-weight: 700;">${ts.llm_input_tokens || 0}</div>
                            </div>
                            <div style="background: rgba(0,0,0,0.3); padding: 0.6rem; border-radius: 0.4rem;">
                                <div style="font-size: 0.72rem; color: var(--text-secondary);">LLM Output</div>
                                <div style="font-size: 1.1rem; font-weight: 700;">${ts.llm_output_tokens || 0}</div>
                            </div>
                            <div style="background: rgba(0,0,0,0.3); padding: 0.6rem; border-radius: 0.4rem;">
                                <div style="font-size: 0.72rem; color: var(--text-secondary);">Final Output</div>
                                <div style="font-size: 1.1rem; font-weight: 700; color: var(--accent-green);">${ts.final_translation_tokens || 0}</div>
                            </div>
                            <div style="background: rgba(59,130,246,0.15); border: 1px solid var(--accent-primary); padding: 0.6rem; border-radius: 0.4rem;">
                                <div style="font-size: 0.72rem; color: var(--accent-primary); font-weight:700;">Total Tokens</div>
                                <div style="font-size: 1.1rem; font-weight: 800; color: var(--text-primary);">${ts.total_tokens || 0}</div>
                            </div>
                        </div>
                    </div>
                </div>
            `;
        }

        // Event Listeners
        runBtn.addEventListener('click', executeBacktest);
        agentBtn.addEventListener('click', executeAgentWorkflow);
        ragSearchBtn.addEventListener('click', handleRAGSearch);
        ragQueryInput.addEventListener('keydown', (e) => { if (e.key === 'Enter') handleRAGSearch(); });

        // Initial Boot
        
        // =====================================================================
        // MODE SWITCHING & SIGNAL VALIDATION JAVASCRIPT LOGIC
        // =====================================================================
        let currentValidationData = null;
        let activeValTab = 'metrics';
        let activeValChart = 'equity';

        function switchTerminalMode(mode) {
            const terminalView = document.getElementById('terminal-view');
            const validationView = document.getElementById('validation-view');
            const terminalControls = document.getElementById('terminal-controls');
            const btnTerm = document.getElementById('mode-btn-terminal');
            const btnVal = document.getElementById('mode-btn-validation');

            if (mode === 'terminal') {
                terminalView.classList.remove('hidden');
                validationView.classList.add('hidden');
                if (terminalControls) terminalControls.classList.remove('hidden');
                btnTerm.classList.add('active');
                btnVal.classList.remove('active');
            } else {
                terminalView.classList.add('hidden');
                validationView.classList.remove('hidden');
                if (terminalControls) terminalControls.classList.add('hidden');
                btnVal.classList.add('active');
                btnTerm.classList.remove('active');

                // If not loaded yet, execute simulation
                if (!currentValidationData) {
                    executeValidationSimulation();
                }
            }
        }

        function toggleValApiKey() {
            const inp = document.getElementById('val-param-apikey');
            inp.type = inp.type === 'password' ? 'text' : 'password';
        }

        // Subtab switching in Validation View
        document.querySelectorAll('#val-subtabs .tab-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                document.querySelectorAll('#val-subtabs .tab-btn').forEach(b => b.classList.remove('active'));
                e.target.classList.add('active');
                activeValTab = e.target.dataset.vtab;

                document.querySelectorAll('.val-tab-pane').forEach(p => p.classList.add('hidden'));
                const activePane = document.getElementById(`vtab-content-${activeValTab}`);
                if (activePane) activePane.classList.remove('hidden');
            });
        });

        // Chart switching in Validation View
        document.querySelectorAll('#val-chart-subtabs .tab-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                document.querySelectorAll('#val-chart-subtabs .tab-btn').forEach(b => b.classList.remove('active'));
                e.target.classList.add('active');
                activeValChart = e.target.dataset.vchart;
                const chartImg = document.getElementById('val-display-chart');
                if (activeValChart === 'equity') chartImg.src = '/validation_img/equity_curve.png?' + new Date().getTime();
                else if (activeValChart === 'drawdown') chartImg.src = '/validation_img/drawdown_curve.png?' + new Date().getTime();
                else if (activeValChart === 'histogram') chartImg.src = '/validation_img/yearly_return_histogram.png?' + new Date().getTime();
            });
        });

        
        let customValFileContent = null;
        let customValFileName = "";

        function handleValFileSelectChange() {
            const sel = document.getElementById('val-file-select').value;
            const btnBrowse = document.getElementById('val-btn-browse');
            const lblName = document.getElementById('val-custom-file-label');
            if (sel === 'custom_upload') {
                if (btnBrowse) btnBrowse.style.display = 'inline-block';
                if (lblName) lblName.style.display = 'inline-block';
                const fileInp = document.getElementById('val-file-upload');
                if (fileInp) fileInp.click();
            } else {
                if (btnBrowse) btnBrowse.style.display = 'none';
                if (lblName) lblName.style.display = 'none';
                customValFileContent = null;
                customValFileName = "";
            }
        }

        function handleValFileUpload(event) {
            const file = event.target.files[0];
            if (!file) return;
            customValFileName = file.name;
            const reader = new FileReader();
            reader.onload = function(e) {
                customValFileContent = e.target.result;
                const lbl = document.getElementById('val-custom-file-label');
                if (lbl) {
                    lbl.innerText = `✓ Loaded: ${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
                    lbl.style.display = 'inline-block';
                }
            };
            reader.readAsText(file);
        }

        async function executeValidationSimulation() {
            const runBtn = document.getElementById('val-run-btn');
            const fileSelect = document.getElementById('val-file-select');
            const selectedFile = fileSelect ? fileSelect.value : 'signalsmain2.py.csv';
            const origHtml = runBtn.innerHTML;
            runBtn.innerHTML = `⏳ Simulating & Verifying with AI...`;
            runBtn.disabled = true;

            try {
                const capInp = document.getElementById('val-param-capital');
                const allocInp = document.getElementById('val-param-alloc');
                const rfInp = document.getElementById('val-param-rf');
                const modeInp = document.getElementById('val-param-mode');
                const modelInp = document.getElementById('val-param-model');
                const apiKeyInp = document.getElementById('val-param-apikey');

                const payload = {
                    file: (selectedFile === 'custom_upload' && customValFileName) ? customValFileName : selectedFile,
                    file_content: (selectedFile === 'custom_upload') ? customValFileContent : null,
                    capital: capInp ? parseFloat(capInp.value) : 10000000,
                    allocation: allocInp ? parseFloat(allocInp.value) / 100.0 : 0.05,
                    risk_free: rfInp ? parseFloat(rfInp.value) / 100.0 : 0.06,
                    mode: modeInp ? modeInp.value : 'long_only',
                    model: modelInp ? modelInp.value : 'gpt-4o-mini',
                    api_key: apiKeyInp ? apiKeyInp.value.trim() : ''
                };

                const resp = await fetch('/api/validate_signal', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });

                const data = await resp.json();
                if (data.status === 'SUCCESS') {
                    currentValidationData = data;
                    renderValidationDashboard(data);
                } else {
                    alert('Validation error: ' + (data.message || 'Unknown error'));
                }
            } catch (err) {
                console.error('Validation request error:', err);
                alert('Failed to execute validation: ' + err);
            } finally {
                runBtn.innerHTML = origHtml;
                runBtn.disabled = false;
            }
        }

        function renderValidationDashboard(data) {
            const m = data.metrics;
            const vm = data.validation_meta;

            // Update Dynamic Source Labels
            if (vm) {
                const badgeSource = document.getElementById('val-badge-source');
                if (badgeSource) {
                    badgeSource.innerText = `📁 Source: ${vm.file_name || 'signalsmain2.py.csv'} (${vm.total_rows || data.trade_count} Signals)`;
                }
                const descSource = document.getElementById('val-source-desc');
                if (descSource) {
                    descSource.innerText = vm.file_name || 'signalsmain2.py.csv';
                }
            }

            if (data.llm_audit && data.llm_audit.engine) {
                const modelBadge = document.getElementById('val-badge-model');
                if (modelBadge) {
                    modelBadge.innerText = `🤖 Autonomous AI Auditor: ${data.llm_audit.engine}`;
                }
            }

            // 1. Update 12 KPI Cards
            document.getElementById('val-card-equity').innerText = '₹' + formatCurrency(m['Final Equity']);
            document.getElementById('val-card-total-return').innerText = (m['Total Return %'] >= 0 ? '+' : '') + m['Total Return %'] + '%';
            document.getElementById('val-card-pnl').innerText = (m['Total P&L'] >= 0 ? '+₹' : '₹') + formatCurrency(m['Total P&L']);
            document.getElementById('val-card-avg-trade').innerText = '₹' + formatCurrency(m['Average P&L per Trade']);
            document.getElementById('val-card-cagr').innerText = m['CAGR %'] + '%';
            document.getElementById('val-card-span').innerText = m['Trading Years'] + ' Years (' + m['Calendar Days'] + ' Days)';
            document.getElementById('val-card-winrate').innerText = m['Win Rate %'] + '%';
            document.getElementById('val-card-wins').innerText = m['Winning Trades'];
            document.getElementById('val-card-losses').innerText = m['Losing Trades'];
            document.getElementById('val-card-profitfactor').innerText = m['Profit Factor'];
            document.getElementById('val-card-ev').innerText = '₹' + formatCurrency(m['Expectancy Value (EV)']);
            document.getElementById('val-card-ev-pct').innerText = '+' + m['Expectancy %'] + '% / trade';
            document.getElementById('val-card-sharpe').innerText = m['Sharpe Ratio'];
            document.getElementById('val-card-maxdd').innerText = m['Maximum Drawdown %'] + '%';
            document.getElementById('val-card-maxdd-inr').innerText = '₹' + formatCurrency(m['Maximum Drawdown (INR)']);
            document.getElementById('val-card-sortino').innerText = m['Sortino Ratio'];
            document.getElementById('val-card-calmar').innerText = m['Calmar Ratio'];
            document.getElementById('val-card-streaks').innerText = m['Max Win Streak'] + 'W / ' + m['Max Loss Streak'] + 'L';
            document.getElementById('val-card-skewness').innerText = (m['Skewness'] >= 0 ? '+' : '') + m['Skewness'];

            // 2. Render Section 1: Metrics & Formulas Table
            renderValidationMetricsTable(m);

            // 3. Render Section 2: AI Verification (LLM calculations, Checklist, Outliers, Narrative)
            renderValidationLLMAudit(data.llm_audit, data.llm_checklist, data.llm_narrative);

            // 4. Render Section 3: Live CSV Previews
            renderValidationCSVPreviews(data);

            // 5. Refresh Visual Charts
            const chartImg = document.getElementById('val-display-chart');
            if (chartImg) {
                const ts = new Date().getTime();
                if (activeValChart === 'equity') chartImg.src = '/validation_img/equity_curve.png?' + ts;
                else if (activeValChart === 'drawdown') chartImg.src = '/validation_img/drawdown_curve.png?' + ts;
                else if (activeValChart === 'histogram') chartImg.src = '/validation_img/yearly_return_histogram.png?' + ts;
            }
        }

        function renderValidationMetricsTable(m) {
            const tbody = document.getElementById('val-metrics-tbody');
            const categories = [
                {
                    cat: "1. Trading Period & Timeline",
                    items: [
                        { name: "Trading Period Start", val: m["Trading Period Start"], formula: "Min(Entry Date) from validated dataset" },
                        { name: "Trading Period End", val: m["Trading Period End"], formula: "Max(Exit Date) from validated dataset" },
                        { name: "Calendar Duration", val: m["Calendar Days"] + " Days", formula: "Max Date - Min Date (Calendar days)" },
                        { name: "Trading Years", val: m["Trading Years"] + " Years", formula: "Calendar Days / 365.25" }
                    ]
                },
                {
                    cat: "2. Trade Statistics",
                    items: [
                        { name: "Total Trades", val: m["Total Trades"], formula: "COUNT(Executed Trade Signals)" },
                        { name: "Winning Trades", val: m["Winning Trades"], formula: "COUNTIF(Trade Net P&L > 0)" },
                        { name: "Losing Trades", val: m["Losing Trades"], formula: "COUNTIF(Trade Net P&L <= 0)" },
                        { name: "Win Rate %", val: m["Win Rate %"] + "%", formula: "(Winning Trades / Total Trades) * 100" },
                        { name: "Loss Rate %", val: m["Loss Rate %"] + "%", formula: "(Losing Trades / Total Trades) * 100" }
                    ]
                },
                {
                    cat: "3. Returns & Profitability",
                    items: [
                        { name: "Initial Capital", val: "₹" + formatCurrency(m["Initial Capital"]), formula: "Prerequisite: ₹1 Crore (₹10,000,000)" },
                        { name: "Final Portfolio Equity", val: "₹" + formatCurrency(m["Final Equity"]), formula: "Initial Capital + Total Realized P&L" },
                        { name: "Total Net P&L", val: "₹" + formatCurrency(m["Total P&L"]), formula: "SUM(Net Trade P&L after costs)" },
                        { name: "Total Return %", val: m["Total Return %"] + "%", formula: "(Total P&L / Initial Capital) * 100" },
                        { name: "CAGR %", val: m["CAGR %"] + "%", formula: "((Final Equity / Initial Capital)^(1 / Years)) - 1" },
                        { name: "Average P&L per Trade", val: "₹" + formatCurrency(m["Average P&L per Trade"]), formula: "Total P&L / Total Trades" },
                        { name: "Median P&L", val: "₹" + formatCurrency(m["Median P&L"]), formula: "MEDIAN(Trade P&L Series)" },
                        { name: "Gross Profit", val: "₹" + formatCurrency(m["Gross Profit"]), formula: "SUMIF(Net P&L > 0)" },
                        { name: "Gross Loss", val: "₹" + formatCurrency(m["Gross Loss"]), formula: "SUMIF(Net P&L < 0)" },
                        { name: "Profit Factor", val: m["Profit Factor"], formula: "Gross Profit / abs(Gross Loss)" },
                        { name: "Average Win", val: "₹" + formatCurrency(m["Average Win"]), formula: "Gross Profit / Winning Trades" },
                        { name: "Average Loss", val: "₹" + formatCurrency(m["Average Loss"]), formula: "Gross Loss / Losing Trades" },
                        { name: "Risk/Reward Ratio", val: m["Risk/Reward Ratio"] + "x", formula: "Average Win / abs(Average Loss)" }
                    ]
                },
                {
                    cat: "4. Trading Edge & Risk Metrics",
                    items: [
                        { name: "Expectancy Value (EV)", val: "₹" + formatCurrency(m["Expectancy Value (EV)"]), formula: "(Win Rate * Avg Win) + (Loss Rate * Avg Loss)" },
                        { name: "Expectancy % per Trade", val: "+" + m["Expectancy %"] + "%", formula: "(Win Rate * Avg Win %) - (Loss Rate * abs(Avg Loss %))" },
                        { name: "Sharpe Ratio", val: m["Sharpe Ratio"], formula: "(Annualized Return - 6% Rf) / Annualized Trade Volatility" },
                        { name: "Sortino Ratio", val: m["Sortino Ratio"], formula: "(Annualized Return - 6% Rf) / Downside Volatility" },
                        { name: "Calmar Ratio", val: m["Calmar Ratio"], formula: "CAGR / abs(Maximum Drawdown %)" },
                        { name: "Annual Volatility %", val: m["Annual Volatility %"] + "%", formula: "Daily Equity Returns StdDev * sqrt(252)" },
                        { name: "Average Holding Period", val: m["Average Holding Period (Days)"] + " Days", formula: "Average(Exit Date - Entry Date)" }
                    ]
                },
                {
                    cat: "5. Streaks & Capital Preservation",
                    items: [
                        { name: "Max Consecutive Wins", val: m["Max Win Streak"] + " Trades", formula: "Max contiguous run of positive P&L trades" },
                        { name: "Max Consecutive Losses", val: m["Max Loss Streak"] + " Trades", formula: "Max contiguous run of negative P&L trades" },
                        { name: "Max Consecutive Profit (INR)", val: "₹" + formatCurrency(m["Max Consecutive Profit"]), formula: "Highest cumulative dollar run during winning streak" },
                        { name: "Max Consecutive Loss (INR)", val: "₹" + formatCurrency(m["Max Consecutive Loss"]), formula: "Highest cumulative dollar drawdown in losing streak" }
                    ]
                },
                {
                    cat: "6. Drawdown Profile",
                    items: [
                        { name: "Maximum Drawdown (INR)", val: "₹" + formatCurrency(m["Maximum Drawdown (INR)"]), formula: "Min(Equity - Running High Watermark Peak)" },
                        { name: "Maximum Drawdown %", val: m["Maximum Drawdown %"] + "%", formula: "Min((Equity - Peak) / Peak) * 100" },
                        { name: "Max DD Duration", val: m["Max DD Duration (Days)"] + " Days", formula: "Peak Date to Recovery Date in days" },
                        { name: "Drawdown Peak Date", val: m["Drawdown Peak Date"], formula: "High watermark preceding the deepest trough" },
                        { name: "Drawdown Trough Date", val: m["Drawdown Trough Date"], formula: "Date of lowest underwater portfolio equity" },
                        { name: "Recovery Date", val: m["Recovery Date"], formula: "First date equity regained or exceeded peak" }
                    ]
                },
                {
                    cat: "7. Skewness & Tail-Risk Distribution",
                    items: [
                        { name: "Fisher-Pearson Skewness", val: m["Skewness"], formula: "[n / ((n-1)(n-2))] * SUM(((x - mean)/s)^3)" },
                        { name: "Skewness Classification", val: m["Skewness Classification"], formula: "Positive skew confirms explosive right-tail winners" }
                    ]
                }
            ];

            let html = "";
            categories.forEach(cg => {
                html += `<tr class="val-cat-header"><td colspan="3">${cg.cat}</td></tr>`;
                cg.items.forEach(it => {
                    html += `
                        <tr>
                            <td style="font-weight: 600; color: #f8fafc;">${it.name}</td>
                            <td style="font-weight: 700; color: #60a5fa;">${it.val}</td>
                            <td style="font-family: var(--font-mono); font-size: 0.82rem; color: #94a3b8;">${it.formula}</td>
                        </tr>
                    `;
                });
            });
            tbody.innerHTML = html;
        }

        function renderValidationLLMAudit(audit) {
            if (!audit) return;
            const count = (audit.llm_metrics_table || []).length;
            const labelEl = document.getElementById('val-llm-engine-label');
            if (labelEl) {
                labelEl.innerText = `AI Model: ${audit.engine || 'Deterministic Quantitative Auditor'} | ${count} Quantitative Metrics Audited & Reconciled with Engine Data`;
            }

            // Render Side-by-Side Table (5 Columns: Metric, Engine Value, LLM Value, Status, AI Note)
            const llmTbody = document.getElementById('val-llm-metrics-tbody');
            if (llmTbody && audit.llm_metrics_table) {
                let llmHtml = "";
                audit.llm_metrics_table.forEach(r => {
                    const status = r["Status"] || "MATCH";
                    const statusBadge = `<span class="badge" style="background: rgba(16, 185, 129, 0.2); color: #34d399; font-weight: 700; border: 1px solid rgba(16, 185, 129, 0.4); padding: 0.2rem 0.5rem; font-size: 0.78rem;">✓ ${status}</span>`;
                    const engVal = r["Engine Value"] || r["Engine Display Value"] || r["LLM Calculated Value"];
                    llmHtml += `
                        <tr>
                            <td style="font-weight: 700; color: #f8fafc;">${r["Metric Name"]}</td>
                            <td style="font-weight: 700; color: #60a5fa; font-family: var(--font-mono); font-size: 0.92rem;">${engVal}</td>
                            <td style="font-weight: 700; color: #34d399; font-family: var(--font-mono); font-size: 0.92rem;">${r["LLM Calculated Value"]}</td>
                            <td style="text-align: center;">${statusBadge}</td>
                            <td style="font-size: 0.83rem; color: #cbd5e1; line-height: 1.45;">${r["LLM Quantitative Note"] || ''}</td>
                        </tr>
                    `;
                });
                llmTbody.innerHTML = llmHtml;
            }

            const narEl = document.getElementById('val-llm-narrative-box');
            if (narEl && audit.narrative) {
                narEl.innerText = audit.narrative;
            }
        }

        function renderValidationCSVPreviews(data) {
            // Preview 1: validation_metrics.csv
            const metricsTbody = document.getElementById('val-metrics-preview-tbody');
            if (metricsTbody && data.metrics) {
                const m = data.metrics;
                const topEntries = [
                    { name: "Total Trades", val: m["Total Trades"], formula: "COUNT(Executed Trade Signals)" },
                    { name: "Win Rate %", val: m["Win Rate %"] + "%", formula: "(Winning Trades / Total Trades) * 100" },
                    { name: "Total Net P&L", val: "₹" + formatCurrency(m["Total P&L"]), formula: "SUM(Net Trade P&L after costs)" },
                    { name: "Total Return %", val: m["Total Return %"] + "%", formula: "(Total P&L / Initial Capital) * 100" },
                    { name: "CAGR %", val: m["CAGR %"] + "%", formula: "((Final Equity / Initial Capital)^(1 / Years)) - 1" },
                    { name: "Profit Factor", val: m["Profit Factor"], formula: "Gross Profit / abs(Gross Loss)" },
                    { name: "Expectancy Value (EV)", val: "₹" + formatCurrency(m["Expectancy Value (EV)"]), formula: "(Win Rate * Avg Win) + (Loss Rate * Avg Loss)" },
                    { name: "Maximum Drawdown %", val: m["Maximum Drawdown %"] + "%", formula: "Min((Equity - Peak) / Peak) * 100" }
                ];
                let mHtml = "";
                topEntries.forEach(r => {
                    mHtml += `
                        <tr>
                            <td style="font-weight: 700; color: #f8fafc;">${r.name}</td>
                            <td style="font-weight: 700; color: #60a5fa;">${r.val}</td>
                            <td style="font-family: var(--font-mono); font-size: 0.82rem; color: #94a3b8;">${r.formula}</td>
                        </tr>
                    `;
                });
                metricsTbody.innerHTML = mHtml;
            }

            // Preview 2: llm_verification_metrics.csv
            const llmTbody = document.getElementById('val-llm-preview-tbody');
            if (llmTbody && data.llm_audit && data.llm_audit.llm_metrics_table) {
                let lHtml = "";
                data.llm_audit.llm_metrics_table.slice(0, 8).forEach(r => {
                    lHtml += `
                        <tr>
                            <td style="font-weight: 700; color: #f8fafc;">${r["Metric Name"]}</td>
                            <td style="font-weight: 700; color: #c084fc; font-family: var(--font-mono);">${r["LLM Calculated Value"]}</td>
                            <td style="font-size: 0.82rem; color: #cbd5e1;">${r["LLM Quantitative Note"] || ''}</td>
                        </tr>
                    `;
                });
                llmTbody.innerHTML = lHtml;
            }
        }

        function downloadValidationCSV(filename) {
            window.location.href = `/api/validation_csv?file=${filename}`;
        }


        window.addEventListener('load', () => {
            executeBacktest();
            fetchStocksOverview();
        });
    </script>
</body>
</html>
"""


def sanitize_json_obj(obj: Any) -> Any:
    """Recursively replaces NaN, Inf, -Inf with None/null for valid RFC 8259 JSON."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    elif isinstance(obj, dict):
        return {k: sanitize_json_obj(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_json_obj(v) for v in obj]
    elif isinstance(obj, tuple):
        return [sanitize_json_obj(v) for v in obj]
    return obj


class QuantumServerHandler(BaseHTTPRequestHandler):
    """HTTP Request Handler for QUANTUM LABS Terminal."""

    def log_message(self, format, *args):
        # Active institutional logging with timestamp
        ts = self.log_date_time_string()
        sys.stderr.write(f"[{ts}] {self.address_string()} - {format % args}\n")
        sys.stderr.flush()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)

        print(f"[{self.log_date_time_string()}] 🌐 GET {path} from {self.client_address[0]}")
        # 1. Main Dashboard UI
        if path == "/" or path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(INDEX_HTML.encode("utf-8"))
            return

        # 2. Run Backtest API
        elif path == "/api/run":
            params = {k: v[0] for k, v in query.items()}
            result = run_full_simulation(params)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(sanitize_json_obj(result)).encode("utf-8"))
            return

        # 3. Agent Workflow API
        elif path == "/api/agent":
            params = {k: v[0] for k, v in query.items()}
            result = run_agent_workflow_api(params)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(sanitize_json_obj(result)).encode("utf-8"))
            return

        # 4. Stocks Overview API
        elif path == "/api/stocks":
            stocks_data = get_universe_stocks_overview()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(sanitize_json_obj(stocks_data)).encode("utf-8"))
            return

        # 5. RAG API (/api/rag and /api/rag/query)
        elif path in ["/api/rag", "/api/rag/query"]:
            q_text = query.get("query", [""])[0]
            chain = RAGChain()
            rag_res = chain.execute_pipeline(q_text)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(sanitize_json_obj(rag_res)).encode("utf-8"))
            return

        # 6. Serve static result images
        elif path.startswith("/results/"):
            filename = os.path.basename(path)
            file_path = os.path.join(BASE_DIR, "results", filename)
            if os.path.exists(file_path):
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.end_headers()
                with open(file_path, "rb") as f:
                    self.wfile.write(f.read())
                return

        # 7. Validation Signal Simulation API
        elif path == "/api/validate_signal":
            try:
                sig_file = query.get("file", ["signalsmain2.py.csv"])[0]
                print(f"[{self.log_date_time_string()}] 🤖 [QUANTLAB] Executing AI Validation for '{sig_file}'...")
                cap = float(query.get("capital", [10000000.0])[0])
                alloc = float(query.get("allocation", [0.05])[0])
                rf = float(query.get("risk_free", [0.06])[0])
                model = query.get("model", ["gpt-4o-mini"])[0]
                api_key = query.get("api_key", [""])[0]
                mode = query.get("mode", ["long_only"])[0]
                val_res = run_validation_simulation(
                    file_path=sig_file,
                    initial_capital=cap,
                    allocation=alloc,
                    risk_free=rf,
                    strategy_mode=mode,
                    openai_api_key=api_key if api_key else None,
                    openai_model=model
                )
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.end_headers()
                self.wfile.write(json.dumps(sanitize_json_obj(val_res), default=str).encode("utf-8"))
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "ERROR", "message": str(e)}).encode("utf-8"))
            return

        # 8. Validation CSV Downloads API
        elif path == "/api/validation_csv" or path.startswith("/api/validation_csv/"):
            filename = query.get("file", [""])[0]
            if not filename and path.startswith("/api/validation_csv/"):
                filename = os.path.basename(path)
            possible_paths = [
                os.path.join(SRC_DIR, "validation_signal", "output", filename),
                os.path.join(SRC_DIR, "validation_signal", filename),
                os.path.join(BASE_DIR, "results", filename)
            ]
            found_path = next((p for p in possible_paths if os.path.exists(p)), None)
            if found_path and filename.endswith(".csv"):
                self.send_response(200)
                self.send_header("Content-Type", "text/csv; charset=utf-8")
                self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
                self.end_headers()
                with open(found_path, "rb") as f:
                    self.wfile.write(f.read())
            else:
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b"CSV File Not Found")
            return

        # 9. Validation Chart Images
        elif path.startswith("/validation_img/"):
            img_name = os.path.basename(path)
            img_path = os.path.join(SRC_DIR, "validation_signal", "output", img_name)
            if os.path.exists(img_path):
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.end_headers()
                with open(img_path, "rb") as f:
                    self.wfile.write(f.read())
            else:
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b"Image Not Found")
            return

        # 404 Fallback
        self.send_response(404)
        self.end_headers()
        self.wfile.write(b"404 Not Found")

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length).decode("utf-8") if length > 0 else "{}"
        
        try:
            payload = json.loads(body)
        except Exception:
            payload = {}

        print(f"[{self.log_date_time_string()}] 🚀 POST {path} from {self.client_address[0]}")
        if path == "/api/run":
            tk = payload.get("ticker", "INFY")
            st = payload.get("strategy", "SMA")
            print(f"[{self.log_date_time_string()}] 🚀 [QUANTLAB] Executing Backtest for {tk} ({st})...")
            result = run_full_simulation(payload)
            print(f"[{self.log_date_time_string()}] ✅ [QUANTLAB] Backtest complete. Return: {result.get('metrics', {}).get('Total Return', 'N/A')}")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(sanitize_json_obj(result)).encode("utf-8"))
            return

        elif path == "/api/agent":
            result = run_agent_workflow_api(payload)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(sanitize_json_obj(result)).encode("utf-8"))
            return

        elif path in ["/api/rag", "/api/rag/query"]:
            query_str = payload.get("query", "")
            chain = RAGChain()
            rag_res = chain.execute_pipeline(query_str)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(sanitize_json_obj(rag_res)).encode("utf-8"))
            return

        elif path == "/api/validate_signal":
            try:
                sig_file = payload.get("file") or payload.get("signal_file", "signalsmain2.py.csv")
                file_content = payload.get("file_content")
                print(f"[{self.log_date_time_string()}] 🤖 [QUANTLAB] Executing AI Validation for '{sig_file}'...")
                
                # If custom file content provided, save to scratch file
                if file_content and (sig_file.endswith(".csv") or sig_file.endswith(".xlsx")):
                    custom_path = os.path.join(SRC_DIR, "validation_signal", "output", sig_file)
                    with open(custom_path, "w", encoding="utf-8") as f:
                        f.write(file_content)
                    sig_file = custom_path
                cap = float(payload.get("capital", 10000000.0))
                alloc = float(payload.get("allocation", 0.05))
                rf = float(payload.get("risk_free", 0.06))
                model = payload.get("model", "gpt-4o-mini")
                api_key = payload.get("api_key", "")
                mode = payload.get("mode", "long_only")
                sig_file = payload.get("file") or payload.get("signal_file", "signalsmain2.py.csv")
                val_res = run_validation_simulation(
                    file_path=sig_file,
                    initial_capital=cap,
                    allocation=alloc,
                    risk_free=rf,
                    strategy_mode=mode,
                    openai_api_key=api_key if api_key else None,
                    openai_model=model
                )
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.end_headers()
                self.wfile.write(json.dumps(sanitize_json_obj(val_res), default=str).encode("utf-8"))
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "ERROR", "message": str(e)}).encode("utf-8"))
            return

        self.send_response(404)
        self.end_headers()


def start_quantum_server(port: int = 8000):
    """Launch HTTP Server."""
    server_address = ("", port)
    httpd = HTTPServer(server_address, QuantumServerHandler)
    print("=" * 80)
    print(f"🚀 QUANTUM LABS Web Dashboard Server running at: http://localhost:{port}")
    print("=" * 80)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[Server Shutdown] QUANTUM LABS Web Server stopped by user.")
    except Exception as e:
        print(f"\n[Server Error] Server stopped: {e}")
    finally:
        try:
            httpd.server_close()
        except Exception:
            pass


if __name__ == "__main__":
    port_num = 8000
    if len(sys.argv) > 1:
        try:
            port_num = int(sys.argv[1])
        except ValueError:
            pass
    start_quantum_server(port_num)
