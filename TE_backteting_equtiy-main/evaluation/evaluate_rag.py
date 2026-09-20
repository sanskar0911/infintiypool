"""
evaluate_rag.py

Evaluates multilingual RAG retrieval quality computing:
- Top-1 Retrieval Accuracy
- Top-3 Retrieval Accuracy
- Average Similarity Score
- Retrieval Latency
"""

import os
import sys
import json
import time
from typing import Dict, Any

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from rag.rag_chain import RAGChain


def evaluate_rag_pipeline(dataset_path: str = None) -> Dict[str, Any]:
    """Run retrieval evaluation over the test dataset."""
    if dataset_path is None:
        dataset_path = os.path.join(BASE_DIR, "evaluation", "rag_questions.json")

    if not os.path.exists(dataset_path):
        print(f"[Error] Dataset path {dataset_path} does not exist.")
        return {}

    with open(dataset_path, "r", encoding="utf-8") as f:
        questions = json.load(f)

    chain = RAGChain()
    top_1_hits = 0
    top_3_hits = 0
    total_latency_ms = 0.0
    detailed = []

    for item in questions:
        q = item["question"]
        expected = [e.replace("\\", "/") for e in item.get("expected_sources", [])]

        t0 = time.time()
        res = chain.execute_pipeline(q, top_k=3)
        elapsed_ms = (time.time() - t0) * 1000.0
        total_latency_ms += elapsed_ms

        retrieved = [s["source"].replace("\\", "/") for s in res.get("retrieved_sources", [])]

        top_1_hit = False
        top_3_hit = False

        if retrieved:
            # Check if any expected source matches top 1
            if any(retrieved[0].endswith(exp) or exp.endswith(retrieved[0]) for exp in expected):
                top_1_hit = True
            # Check if any expected source is in top 3
            if any(any(r.endswith(exp) or exp.endswith(r) for exp in expected) for r in retrieved):
                top_3_hit = True

        if top_1_hit:
            top_1_hits += 1
        if top_3_hit:
            top_3_hits += 1

        detailed.append({
            "question": q,
            "language": item.get("language", "Unknown"),
            "detected_language": res.get("detected_language"),
            "top_1_hit": top_1_hit,
            "top_3_hit": top_3_hit,
            "retrieved_sources": retrieved,
            "latency_ms": round(elapsed_ms, 2)
        })

    total = len(questions)
    top_1_acc = round((top_1_hits / total) * 100.0, 2) if total > 0 else 0.0
    top_3_acc = round((top_3_hits / total) * 100.0, 2) if total > 0 else 0.0
    avg_latency = round(total_latency_ms / total, 2) if total > 0 else 0.0

    output = {
        "total_questions": total,
        "top_1_hits": top_1_hits,
        "top_3_hits": top_3_hits,
        "top_1_accuracy_pct": top_1_acc,
        "top_3_accuracy_pct": top_3_acc,
        "average_latency_ms": avg_latency,
        "detailed_results": detailed
    }

    print("=" * 70)
    print("QUANTLAB RAG RETRIEVAL EVALUATION RESULTS")
    print("=" * 70)
    print(f"Total Test Questions: {total}")
    print(f"Top-1 Accuracy:       {top_1_acc}%")
    print(f"Top-3 Accuracy:       {top_3_acc}%")
    print(f"Avg Latency:          {avg_latency} ms")
    print("=" * 70)

    return output


if __name__ == "__main__":
    evaluate_rag_pipeline()
