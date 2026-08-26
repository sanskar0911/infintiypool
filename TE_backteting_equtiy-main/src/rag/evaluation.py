"""
evaluation.py

Empirical RAG Retrieval Evaluation module computing Top-1 and Top-3 accuracy.
"""

from typing import List, Dict, Any
from rag.retriever import RAGRetriever


# Ground-truth evaluation test dataset
EVALUATION_DATASET: List[Dict[str, Any]] = [
    {
        "question": "What is look-ahead bias in backtesting?",
        "expected_source": "backtesting/look_ahead_bias.md"
    },
    {
        "question": "What is Sharpe Ratio and how is it calculated?",
        "expected_source": "finance/sharpe.md"
    },
    {
        "question": "How does Simple Moving Average (SMA) crossover work?",
        "expected_source": "strategies/sma.md"
    },
    {
        "question": "What is maximum drawdown and how to calculate recovery gain?",
        "expected_source": "finance/drawdown.md"
    },
    {
        "question": "What causes overfitting in quantitative equity strategies?",
        "expected_source": "backtesting/overfitting.md"
    },
    {
        "question": "How does Exponential Moving Average (EMA) differ from SMA?",
        "expected_source": "strategies/ema.md"
    },
    {
        "question": "How do transaction costs and slippage affect backtests?",
        "expected_source": "finance/transaction_costs.md"
    },
    {
        "question": "What is Relative Strength Index (RSI) mean reversion?",
        "expected_source": "strategies/rsi.md"
    }
]


def evaluate_rag_retrieval(retriever: RAGRetriever = None) -> Dict[str, Any]:
    """
    Run empirical retrieval accuracy evaluation across the test dataset.

    Returns
    -------
    Dict[str, Any]
        {
          'total_questions': int,
          'top_1_hits': int,
          'top_3_hits': int,
          'top_1_accuracy': float,
          'top_3_accuracy': float,
          'detailed_results': List[Dict]
        }
    """
    if retriever is None:
        retriever = RAGRetriever()

    top_1_hits = 0
    top_3_hits = 0
    detailed_results = []

    for test_item in EVALUATION_DATASET:
        q = test_item["question"]
        expected = test_item["expected_source"]

        results = retriever.retrieve(q, top_k=3)
        retrieved_sources = [item[0]["source"] for item in results]

        is_top_1 = len(retrieved_sources) > 0 and retrieved_sources[0] == expected
        is_top_3 = expected in retrieved_sources

        if is_top_1:
            top_1_hits += 1
        if is_top_3:
            top_3_hits += 1

        detailed_results.append({
            "question": q,
            "expected_source": expected,
            "retrieved_sources": retrieved_sources,
            "top_1_hit": is_top_1,
            "top_3_hit": is_top_3
        })

    total = len(EVALUATION_DATASET)
    top_1_acc = (top_1_hits / total) * 100.0 if total > 0 else 0.0
    top_3_acc = (top_3_hits / total) * 100.0 if total > 0 else 0.0

    print(f"[RAG Evaluation] Evaluated {total} questions. Top-1 Accuracy: {top_1_acc:.1f}%, Top-3 Accuracy: {top_3_acc:.1f}%")

    return {
        "total_questions": total,
        "top_1_hits": top_1_hits,
        "top_3_hits": top_3_hits,
        "top_1_accuracy": top_1_acc,
        "top_3_accuracy": top_3_acc,
        "detailed_results": detailed_results
    }


if __name__ == "__main__":
    res = evaluate_rag_retrieval()
    print("RAG Retrieval Evaluation Results:")
    print(f"Top-1 Accuracy: {res['top_1_accuracy']:.2f}%")
    print(f"Top-3 Accuracy: {res['top_3_accuracy']:.2f}%")
