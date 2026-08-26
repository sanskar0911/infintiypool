# Agentic Workflow Foundations (LangGraph & Tool Calling)

**Milestone Target:** 15-Aug-2026  
**Topic:** Agentic Workflow Foundations (LangChain, LangGraph, OpenAI Agents SDK, CrewAI)  
**Project:** Equity Strategy Backtesting Engine (Shankh / Decuple Internship 2026-27 Project TE1)  

---

## 1. Architecture Overview

The agentic workflow is built using **LangGraph** (`src/agent/`). It orchestrates a modular, state-driven research process where an autonomous agent invokes deterministic tools to ingest market data, execute backtests, compute performance metrics, and generate AI quantitative reports.

```text
               User Goal / Hypothesis
                         │
                         ▼
                    [Load Data] (DataLoader Tool)
                         │
                         ▼
                  [Run Backtest] (Backtest Tool)
                         │
                         ▼
               [Calculate Metrics] (Metrics Tool)
                         │
                         ▼
             [Research Evaluator Node]
                         │
              ┌──────────┴──────────┐
              │                     │
           Pass                 Iterate
              │                     │
              ▼                     ▼
     [LLM Analysis Tool]   [Controlled Iteration Node]
              │                     │
              ▼                     └───────┐
      [Final Output]                        │
                                            ▼
                                     (Loop back to Backtest)
```

---

## 2. Tool Definition & Registration

The agent interacts with the environment using three primary deterministic Python tools (`src/agent/tools.py`):

1. **`run_data_loader_tool(ticker)`**: Reads processed CSV market data from `data/processed/`.
2. **`run_backtest_tool(price_df, strategy_name, strategy_kwargs, ...)`**: Generates signals (SMA, EMA, RSI) and simulates portfolio execution, transaction costs, and slippage.
3. **`calculate_metrics_tool(portfolio_df, trade_log_df)`**: Computes Sharpe ratio, CAGR, Volatility, Max Drawdown, Win Rate, and Profit Factor.
4. **`analyze_performance_tool(metrics, ticker, strategy)`**: Passes calculated metrics to LLM Strategy Analyzer to generate structured JSON reports.

---

## 3. LangGraph StateGraph Construction

The core graph is constructed using `StateGraph` backed by a `TypedDict` state schema (`BacktestState`):

```python
builder = StateGraph(BacktestState)

# Node registration
builder.add_node("load_data", node_load_data)
builder.add_node("run_backtest", node_run_backtest)
builder.add_node("calculate_metrics", node_calculate_metrics)
builder.add_node("research_evaluator", node_evaluator)
builder.add_node("controlled_iteration", node_controlled_iteration)
builder.add_node("generate_artifacts", node_generate_artifacts)

# Edges
builder.set_entry_point("load_data")
builder.add_edge("load_data", "run_backtest")
builder.add_edge("run_backtest", "calculate_metrics")
builder.add_edge("calculate_metrics", "research_evaluator")

# Conditional decision edge from evaluator
builder.add_conditional_edges(
    "research_evaluator",
    route_loop_decision,
    {
        "iterate": "controlled_iteration",
        "finalize": "generate_artifacts"
    }
)
builder.add_edge("controlled_iteration", "run_backtest")
builder.add_edge("generate_artifacts", END)
```

---

## 4. Traceable Output & Observability

Every step executed by the LangGraph agent records its inputs, outputs, metric history, and state transitions into `state["experiment_history"]`.

### Execution Trace Example:
- **Step 1 [Load Data]**: Ingested 1,234 rows for `RELIANCE`.
- **Step 2 [Run Backtest]**: Executed SMA(20, 50) strategy. Trades executed: 18.
- **Step 3 [Calculate Metrics]**: Sharpe Ratio: `0.65`, CAGR: `8.4%`.
- **Step 4 [Research Evaluator]**: Sharpe `0.65` < target `0.80`. Decision: `ITERATE`.
- **Step 5 [Controlled Iteration]**: Adjusted SMA params to SMA(25, 60). Iteration count: 2.
- **Step 6 [Run Backtest - Iter 2]**: Trades executed: 14.
- **Step 7 [Calculate Metrics - Iter 2]**: Sharpe Ratio: `0.84`, CAGR: `12.1%`.
- **Step 8 [Research Evaluator - Iter 2]**: Sharpe `0.84` >= target `0.80`. Decision: `ACCEPT`.
- **Step 9 [Generate Artifacts]**: LLM analysis generated. Workflow completed.
