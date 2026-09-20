"""
token_counter.py

Token accounting engine measuring:
- Original Input Tokens
- MADRID Input Tokens
- RAG Context Tokens
- LLM Input Tokens
- LLM Output Tokens
- Final Translation Tokens
- Total Tokens
- Observed Token Difference (%)
"""

from typing import Dict, Any

HAS_TIKTOKEN = False
_ENCODING = None

try:
    import tiktoken
    _ENCODING = tiktoken.get_encoding("cl100k_base")
    HAS_TIKTOKEN = True
except Exception:
    HAS_TIKTOKEN = False


def count_tokens(text: str) -> int:
    """
    Count tokens for a given text string.
    Uses tiktoken cl100k_base if available, else standard word/subword approximation.
    """
    if not text:
        return 0
    if HAS_TIKTOKEN and _ENCODING is not None:
        try:
            return len(_ENCODING.encode(text))
        except Exception:
            pass

    # Approximation for non-English script / fallback
    words = text.strip().split()
    return max(1, int(len(words) * 1.3))


def calculate_token_stats(
    original_input: str,
    madrid_input: str,
    rag_context: str,
    llm_output_madrid: str,
    final_output: str
) -> Dict[str, Any]:
    """
    Computes full token breakdown across all RAG + MADRID pipeline stages.
    """
    orig_tokens = count_tokens(original_input)
    madrid_tokens = count_tokens(madrid_input)
    context_tokens = count_tokens(rag_context)
    llm_in_tokens = madrid_tokens + context_tokens
    llm_out_tokens = count_tokens(llm_output_madrid)
    final_tokens = count_tokens(final_output)

    total_tokens = orig_tokens + madrid_tokens + context_tokens + llm_out_tokens + final_tokens

    # Observed token difference between Original Input and MADRID Input
    if orig_tokens > 0:
        diff_pct = round(((madrid_tokens - orig_tokens) / orig_tokens) * 100.0, 2)
    else:
        diff_pct = 0.0

    return {
        "original_input_tokens": orig_tokens,
        "madrid_input_tokens": madrid_tokens,
        "rag_context_tokens": context_tokens,
        "llm_input_tokens": llm_in_tokens,
        "llm_output_tokens": llm_out_tokens,
        "final_translation_tokens": final_tokens,
        "total_tokens": total_tokens,
        "observed_token_difference_pct": diff_pct,
        "compression_summary": f"{'-' if diff_pct < 0 else '+'}{abs(diff_pct):.1f}% token difference vs original"
    }
