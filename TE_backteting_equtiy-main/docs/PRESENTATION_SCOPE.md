# Presentation Scope Declaration

**Project Title:** Backtesting Equity Strategies (Shankh / Decuple Internship 2026-27 Project TE1)  
**Target Milestone Date:** 10-Oct-2026  

---

## Declaration of Scope

> Advanced quantitative research features have been preserved in the backup/archive and are intentionally excluded from the current presentation scope.

The primary objective of this build is to provide a **clean, reliable, and demonstration-focused platform** that clearly reflects the learning milestones achieved throughout the internship program up to 10-Oct-2026.

---

## Summary of Included vs Preserved Modules

### 1. Included Presentation Modules
- **Python & Backtesting Engine**: Clean data ingestion, signal generation (SMA, EMA, RSI), equity backtest simulation, key financial metrics (Sharpe, CAGR, Drawdown, Win Rate, Profit Factor), and Plotly visualization.
- **Benchmark Comparison Engine**: Retained Nifty 50 benchmark analytics (`src/benchmark.py`) for relative strategy performance benchmarking.
- **AI Coding Demonstration**: Documented prompt engineering, AI code generation review, defect identification, and corrections (`docs/AI_CODING_DEMO.md`).
- **LLM Structured Output Abstraction**: Provider abstraction (OpenAI / Gemini / Rule Engine fallback) generating structured JSON investment ratings and risk assessments (`src/llm/`).
- **LangGraph Agent Workflow & Loop Engineering**: Multi-node state graph orchestrating backtests and metrics with explicit, controlled iteration loops and stopping criteria (`src/agent/`).
- **RAG Knowledge Assistant & Retrieval Evaluation**: FAISS vector store powered by markdown documentation corpus (`knowledge_base/`), semantic search, and empirical Top-1/Top-3 retrieval accuracy benchmarking (`src/rag/`).
- **Conversation State Memory**: Stateful multi-turn chat handling follow-up queries with reset capability.
- **Automation & Webhook Demonstration**: Lightweight webhook payload sender for post-backtest notifications (`automation/webhook_demo/`).
- **Security & Deployment Infrastructure**: `.env.example`, environment isolation, logging (`logs/app.log`), and unit testing (`tests/`).
- **Delivery Baseline Report**: Historical baseline metrics and accuracy gap analysis report (`docs/BASELINE_REPORT.md`).

---

### 2. Preserved Advanced Features (Located in `archive/`)
The following advanced research features have been safely isolated into the `archive/` directory to prevent unnecessary runtime complexity during presentation:
- Parameter Sweep & Grid Search Engine (`archive/experiments/parameter_sweep.py`)
- Walk-Forward Optimization Engine (`archive/experiments/walk_forward.py`)
- Out-of-Sample Split & Validation (`archive/experiments/out_of_sample.py`)
- Parameter Sensitivity & Robustness Engine (`archive/experiments/robustness.py`)
- Market Regime Classification & Analysis (`archive/experiments/regime_analysis.py`)
- Overfitting Warning Heuristics (`archive/experiments/research_warnings.py`)
- Portfolio Liquidity Filtering & Position Limits (`archive/portfolio.py`)
- Advanced UI Pages & Analytical Cards (`archive/pages_ui/`, `archive/components/`)

---

## Verification & Integrity Note
All preserved files remain intact in `archive/` and on the remote GitHub backup. The active application (`src/app.py`) operates solely on the active presentation pipeline to ensure zero broken imports and maximum runtime reliability during live demos.
