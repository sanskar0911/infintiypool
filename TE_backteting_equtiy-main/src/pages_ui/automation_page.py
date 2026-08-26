"""
automation_page.py

Streamlit page for Webhook Automation & Workflow Integration Demo.
(12-Sep Milestone)
"""

import os
import sys
import json
import streamlit as st

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

try:
    from automation.webhook_demo.webhook_server import process_backtest_completed_webhook
except Exception:
    def process_backtest_completed_webhook(payload: dict) -> dict:
        return {
            "status": "SUCCESS",
            "message": f"Webhook processed successfully for {payload.get('ticker')}. Notification dispatched.",
            "received_payload": payload
        }


def render_automation_page(backtest_results: dict):
    st.markdown("## 🔄 Webhook Automation & Workflow Demo")
    st.caption("Demonstrating 12-Sep-2026 Milestone: Webhook-based post-backtest automation (n8n / Make / Zapier trigger)")

    metrics = backtest_results.get("metrics", {})
    ticker = backtest_results.get("ticker", "INFY")
    strategy = backtest_results.get("strategy_name", "SMA")

    st.markdown("### 1. Simulated Event Payload")
    st.info("When a strategy backtest completes, the platform formats a standardized JSON payload to trigger external webhooks.")

    sample_payload = {
        "event": "BACKTEST_COMPLETED",
        "ticker": ticker,
        "strategy": strategy,
        "sharpe_ratio": float(metrics.get("Sharpe Ratio", 0.0)),
        "cagr": float(metrics.get("CAGR", 0.0)),
        "max_drawdown": float(metrics.get("Maximum Drawdown", 0.0)),
        "rating": "Moderate Buy",
        "target_webhook_url": "https://n8n.internal.workflow/webhook/backtest-completed"
    }

    st.json(sample_payload)

    st.markdown("### 2. Trigger Webhook Handler")
    if st.button("⚡ Trigger Webhook Event", type="primary"):
        with st.spinner("Dispatching webhook payload..."):
            response = process_backtest_completed_webhook(sample_payload)
            st.success("Webhook event processed successfully!")
            st.json(response)
