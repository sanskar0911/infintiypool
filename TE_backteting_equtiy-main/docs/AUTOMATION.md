# Webhook-Based Automation Workflow

**Milestone Target:** 12-Sep-2026  
**Topic:** n8n / Make / Zapier-Style Automation Workflows  
**Project:** Equity Strategy Backtesting Engine (Shankh / Decuple Internship 2026-27 Project TE1)  

---

## 1. Automation Flow Architecture

```text
Backtest Simulation Completed (app.py)
                 │
                 ▼
      Construct JSON Payload
(Ticker, Strategy, Sharpe, CAGR, AI Rating)
                 │
                 ▼
  Webhook Dispatcher (automation/webhook_demo/)
                 │
                 ▼
     n8n / Make Webhook Receiver
                 │
        ┌────────┴────────┐
        ▼                 ▼
  LLM Alert Step     Slack / Email API Call
(Formats summary)   (Dispatches notification)
```

---

## 2. Event Payload Structure

When a backtest completes, the application constructs a standardized JSON notification payload (`payload_example.json`):

```json
{
  "event": "BACKTEST_COMPLETED",
  "ticker": "INFY",
  "strategy": "SMA",
  "metrics": {
    "cagr": 0.148,
    "sharpe_ratio": 0.88,
    "max_drawdown": -0.125
  },
  "ai_rating": "Moderate Buy"
}
```

---

## 3. Integration Verification

The integration can be tested via the **Automation Demo UI page** or by running `python automation/webhook_demo/webhook_server.py`.
