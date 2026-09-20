"""
utils_logger.py

Structured JSON logger for QUANTLAB RAG + MADRID requests (stripping credentials/keys).
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Any

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "rag_pipeline.log")

logger = logging.getLogger("QUANTLAB_RAG")
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def log_rag_request(payload: Dict[str, Any]):
    """
    Log safe JSON request telemetry excluding secrets.
    """
    safe_payload = {
        "timestamp": datetime.utcnow().isoformat(),
        "query": payload.get("original_query"),
        "language": payload.get("detected_language"),
        "language_code": payload.get("language_code"),
        "status": payload.get("status"),
        "retrieved_count": len(payload.get("retrieved_sources", [])),
        "total_tokens": payload.get("token_stats", {}).get("total_tokens", 0),
        "observed_token_diff_pct": payload.get("token_stats", {}).get("observed_token_difference_pct", 0.0),
        "total_latency_ms": payload.get("total_latency_ms", 0.0)
    }
    logger.info(json.dumps(safe_payload))
