"""
webhook_server.py

Lightweight automation webhook receiver simulating n8n / Make / Zapier integration.
Triggers when a strategy backtest completes, receiving JSON payload and emitting notification logs.
"""

import json
from typing import Dict, Any


def process_backtest_completed_webhook(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process incoming backtest completion webhook trigger.

    Parameters
    ----------
    payload : Dict[str, Any]
        {
          "event": "BACKTEST_COMPLETED",
          "ticker": str,
          "strategy": str,
          "sharpe_ratio": float,
          "cagr": float,
          "max_drawdown": float,
          "rating": str
        }

    Returns
    -------
    Dict[str, Any]
        Webhook response payload.
    """
    event = payload.get("event", "UNKNOWN")
    ticker = payload.get("ticker", "INFY")
    strategy = payload.get("strategy", "SMA")
    sharpe = payload.get("sharpe_ratio", 0.0)
    rating = payload.get("rating", "Neutral")

    print(f"[Webhook Event Received] Event: {event} | Ticker: {ticker} | Strategy: {strategy}")
    print(f"[Webhook Action] Triggering automated notification... Sharpe: {sharpe:.2f} | Rating: {rating}")

    return {
        "status": "SUCCESS",
        "message": f"Webhook processed successfully for {ticker} ({strategy}). Notification dispatched.",
        "received_payload": payload
    }


if __name__ == "__main__":
    sample_payload = {
        "event": "BACKTEST_COMPLETED",
        "ticker": "INFY",
        "strategy": "SMA",
        "sharpe_ratio": 0.88,
        "cagr": 0.148,
        "max_drawdown": -0.125,
        "rating": "Moderate Buy"
    }
    response = process_backtest_completed_webhook(sample_payload)
    print("Webhook Response:")
    print(json.dumps(response, indent=2))
