# AI Coding & LLM Engineering Report

This report documents the AI-assisted pair programming practices, prompt engineering techniques, code reviews, and hallucination management applied during the professional architectural upgrade of the **Equity Backtesting System**.

---

## 1. Cursor Prompts

### Effective System Prompts
```text
Role: Senior Quantitative Developer & AI Engineer.
Task: Upgrade backtesting engine with transaction costs, slippage, configurable position sizing (e.g. 0.2), stop loss (0.05), and take profit (0.10).
Constraint: Maintain modularity, preserve backward compatibility, write production-grade Python code.
```

### Key Interactive Prompts Used
1. **Strategy Factory Pattern**:
   > *"Refactor strategy generation into an abstract BaseStrategy class and a StrategyFactory supporting SMA, EMA, and RSI strategies, maintaining backward-compatible function signatures in strategy.py."*
2. **Quantitative Risk Metrics**:
   > *"Implement Win Rate, Profit Factor, Sortino Ratio, Calmar Ratio, Average Holding Period, Average Trade Return, and Trade Count in metrics.py without breaking existing dictionary keys."*
3. **Structured JSON LLM Output**:
   > *"Construct a prompt template in src/llm/prompts.py that forces the LLM to output valid JSON containing rating, risk, recommendation, and confidence score. Implement a robust fallback rule engine in src/llm/analyzer.py."*

---

## 2. GitHub Copilot Usage

### High-Value Auto-Completions
- **Numpy Vectorized Calculations**: Copilot assisted in autocompleting daily portfolio return calculations and cumulative maximum drawdowns:
  ```python
  running_max = portfolio_values.cummax()
  drawdowns = (portfolio_values - running_max) / running_max
  ```
- **Matplotlib Styling & Formatters**:
  ```python
  ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
  ```

### Developer Inspection Workflow
All Copilot-generated snippets were verified for zero look-ahead bias (e.g., verifying `.shift(1)` was properly applied prior to generating signals on historical prices).

---

## 3. Generated Code Review

### Areas Reviewed & Corrected
1. **Look-Ahead Bias Prevention**:
   - *Initial AI Suggestion*: Computed moving averages on `df["Adj Close"]` and checked `df["Adj Close"] > SMA` on the same day's close for execution.
   - *Refactored Correction*: Explicitly compared `SMA.shift(1)` against current price to guarantee trade execution occurs on available historical signals.
2. **Fractional Shares Handling**:
   - *Initial AI Suggestion*: Computed `shares = cash / price` returning floating-point shares.
   - *Refactored Correction*: Enforced `int(np.floor(allocated_capital / cost_per_share))` to align with stock exchange lot sizes (NSE whole shares).

---

## 4. Prompt Engineering Examples

### Example: Strict JSON Response Enforcement
- **Technique**: System prompt instructions + OpenAI `response_format={"type": "json_object"}` + schema specification.

```python
ANALYSIS_SYSTEM_PROMPT = """You are a Senior Quantitative Analyst.
Output strictly valid JSON:
{
  "rating": "<Strong Buy | Moderate Buy | Neutral | Underperform>",
  "risk": "<Low | Moderate | High>",
  "recommendation": "<Text>",
  "confidence": <float>
}
"""
```

---

## 5. Hallucination Examples & Mitigation

| Issue Type | AI Hallucination Example | Root Cause | Engineering Solution |
| :--- | :--- | :--- | :--- |
| **API Method Misnaming** | Generated `yfinance.download(..., adjust_close=True)` | Obsolete `yfinance` parameter name | Replaced with explicit `auto_adjust=False` and inspected `df.columns` level |
| **JSON Format Deviation** | Enclosed JSON response inside markdown backticks (```json ... ```) | LLM chat completion default formatting | Applied regex/json extraction and `json.loads()` error handling with fallback |
| **Import Deadlocks** | Suggested importing `langgraph` directly without fallback | Non-guaranteed environment package | Added `try...except ImportError` fallback block in `src/agent/workflow.py` |

---

*Report prepared by Senior Quantitative Developer & AI Engineer.*
