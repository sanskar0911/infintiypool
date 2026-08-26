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
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Dict, Any, List
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
    stop_loss: float = 0.05,
    take_profit: float = 0.10
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

    # 3. Execute Backtest Simulation
    portfolio_df, trade_log_df = run_backtest(
        df_signals,
        initial_capital=initial_capital,
        txn_cost_rate=txn_cost_rate,
        slippage_rate=slippage_rate,
        position_size=position_size,
        stop_loss=stop_loss,
        take_profit=take_profit
    )

    # 4. Calculate Quantitative Metrics
    metrics = calculate_metrics(portfolio_df, trade_log_df, initial_capital=initial_capital)

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
        "start_date": df.index[0].strftime("%Y-%m-%d"),
        "end_date": df.index[-1].strftime("%Y-%m-%d"),
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
            trade_type = str(row.get(type_col, "N/A"))
            price = float(row.get(price_col, 0.0))
            shares = int(row.get("Shares", 0))
            val = float(price * shares)
            cost = float(row.get(cost_col, 0.0))
            rem_cash = float(row.get(cash_col, 0.0))

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
        "recommendations": recommendations
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

        .hidden { display: none !important; }
    </style>
</head>
<body>
    <header>
        <div class="logo-container">
            <h1>QUANTUM LABS</h1>
            <p>Institutional AI Equity Backtesting & Research Terminal</p>
        </div>
        <div class="controls">
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
                <div class="rag-input-group">
                    <input type="text" id="rag-query-input" class="rag-input" placeholder="Ask quantitative knowledge base (e.g. 'What is Sharpe Ratio formula?')">
                    <button class="btn-primary" id="rag-search-btn">Ask AI</button>
                </div>
                <div class="ai-card" id="rag-response-container">
                    <p style="color: var(--text-secondary);">Type a query above to search grounded knowledge base.</p>
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

        async function handleRAGSearch() {
            const query = ragQueryInput.value.trim();
            if (!query) return;
            ragSearchBtn.disabled = true;
            ragResponseContainer.innerHTML = `<p style="color: var(--text-secondary);">Searching Knowledge Base...</p>`;

            try {
                const res = await fetch('/api/rag', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ query: query })
                });
                const data = await res.json();
                ragResponseContainer.innerHTML = `
                    <div style="font-weight: 700; margin-bottom: 0.5rem; color: var(--accent-primary);">AI Response:</div>
                    <p style="margin-bottom: 0.8rem;">${data.answer}</p>
                    ${data.sources && data.sources.length > 0 ? `
                        <div style="font-size: 0.8rem; color: var(--text-secondary);">
                            <strong>Sources:</strong> ${data.sources.join(', ')}
                        </div>
                    ` : ''}
                `;
            } catch (e) {
                ragResponseContainer.innerHTML = `<p style="color: var(--accent-red);">Error executing RAG search: ${e.message}</p>`;
            } finally {
                ragSearchBtn.disabled = false;
            }
        }

        // Event Listeners
        runBtn.addEventListener('click', executeBacktest);
        agentBtn.addEventListener('click', executeAgentWorkflow);
        ragSearchBtn.addEventListener('click', handleRAGSearch);
        ragQueryInput.addEventListener('keydown', (e) => { if (e.key === 'Enter') handleRAGSearch(); });

        // Initial Boot
        window.addEventListener('load', () => {
            executeBacktest();
            fetchStocksOverview();
        });
    </script>
</body>
</html>
"""


class QuantumServerHandler(BaseHTTPRequestHandler):
    """HTTP Request Handler for QUANTUM LABS Terminal."""

    def log_message(self, format, *args):
        # Silent logger to prevent terminal clutter
        pass

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)

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
            self.wfile.write(json.dumps(result).encode("utf-8"))
            return

        # 3. Agent Workflow API
        elif path == "/api/agent":
            params = {k: v[0] for k, v in query.items()}
            result = run_agent_workflow_api(params)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(result).encode("utf-8"))
            return

        # 4. Stocks Overview API
        elif path == "/api/stocks":
            stocks_data = get_universe_stocks_overview()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(stocks_data).encode("utf-8"))
            return

        # 5. RAG API (GET query parameter fallback)
        elif path == "/api/rag":
            q_text = query.get("query", [""])[0]
            chain = RAGChain()
            rag_res = chain.answer_question(q_text)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(rag_res).encode("utf-8"))
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

        if path == "/api/run":
            result = run_full_simulation(payload)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(result).encode("utf-8"))
            return

        elif path == "/api/agent":
            result = run_agent_workflow_api(payload)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(result).encode("utf-8"))
            return

        elif path == "/api/rag":
            query_str = payload.get("query", "")
            chain = RAGChain()
            rag_res = chain.answer_question(query_str)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(rag_res).encode("utf-8"))
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
