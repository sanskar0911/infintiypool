# Security, Logging & Deployment Guide

**Milestone Target:** 26-Sep-2026  
**Topic:** Deployment, Logging, Secrets, API Safety & Demo Discipline  
**Project:** Equity Strategy Backtesting Engine (Shankh / Decuple Internship 2026-27 Project TE1)  

---

## 1. Security & Secrets Management

- **Zero Hardcoded Secrets**: No real API keys are hardcoded in the codebase.
- **Environment Template**: `.env.example` provides the key configuration schema.
- **Git Exclusions**: `.gitignore` explicitly excludes `.env`, `*.key`, `logs/*.log`, and binary artifacts.

```bash
# Example environment initialization
cp .env.example .env
# Add real API key securely to local .env
```

---

## 2. Application Logging Architecture

The platform uses Python's standard `logging` library configured via `src/utils.py`.
Logs are output to both stdout and persistent log files under `logs/app.log`.

### Log Levels:
- `INFO`: Normal application execution, backtest completion, page navigation.
- `WARNING`: Benchmark fallback triggers, missing optional packages.
- `ERROR`: File missing, API call errors handled gracefully.

---

## 3. Local Execution Guide

```bash
# 1. Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch Streamlit Application
streamlit run src/app.py
```

---

## 4. Demo Checklist for Live Presentations

1. Verify Python virtual environment is active.
2. Confirm `.env` file is loaded locally.
3. Confirm data files exist in `data/processed/`.
4. Run `pytest tests/` to confirm all unit tests pass before launching.
5. Enable **Presentation Mode** in the Streamlit UI to walk through the 10 learning milestones sequentially.
