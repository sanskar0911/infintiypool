# LLM Foundations & Structured Output Report

**Milestone Target:** 01-Aug-2026  
**Topic:** LLM Foundations (GPT models, Claude, Gemini, Mistral, Prompt Engineering, Structured Output)  
**Project:** Equity Strategy Backtesting Engine (Shankh / Decuple Internship 2026-27 Project TE1)  

---

## 1. Architecture Overview

The LLM module (`src/llm/analyzer.py`) provides an institutional strategy analysis framework combining **deterministic financial metric calculation in Python** with **structured LLM qualitative evaluation**.

```text
  User Input / Backtest Simulation
                 │
                 ▼
Deterministic Python Engine (src/metrics.py)
[Calculates: Total Return, CAGR, Sharpe, Sortino, Max Drawdown, Win Rate, Profit Factor]
                 │
                 ▼
Prompt Engineering Engine (src/llm/prompts.py)
[Binds metrics into strict JSON response schema]
                 │
        ┌────────┴────────┐
        ▼                 ▼
   OpenAI SDK      Gemini API / Fallback Engine
  (gpt-4o-mini)   (gemini-1.5-flash / Rule-Based)
        │                 │
        └────────┬────────┘
                 ▼
    Structured JSON Output
[Rating, Executive Summary, Risk Analysis, Strengths, Weaknesses, Recommendations]
```

---

## 2. Hallucination Prevention Strategy

> **Core Financial Rule:** LLMs must never invent or recompute raw financial numbers. All performance metrics MUST be computed deterministically in Python before being supplied to the prompt.

### Why This Prevents Hallucination:
1. **Source of Truth**: Numerical values (`Sharpe = 0.85`, `CAGR = 14.2%`, `Max Drawdown = -12.4%`) are calculated via Pandas/NumPy in `src/metrics.py`.
2. **Instructional Guardrails**: The LLM prompt explicitly commands:
   - "Do not invent metrics."
   - "Explain the mathematical formula of the provided metric."
   - "Analyze what the provided metric value implies for live trading."
3. **Deterministic Fallback**: If no API key (`OPENAI_API_KEY` / `GEMINI_API_KEY`) is configured or network calls fail, `LLMStrategyAnalyzer` executes a local deterministic rule-based analysis engine that produces the exact same structured JSON schema.

---

## 3. Structured JSON Schema & Prompt Engineering

The system prompt in `src/llm/prompts.py` enforces strict JSON output without markdown wrapping:

```json
{
  "rating": "Strong Buy | Moderate Buy | Neutral | Underperform",
  "risk": "Low | Moderate | High",
  "confidence": 0.95,
  "executive_summary": "High-level summary of strategy performance...",
  "recommendation": "Detailed investment recommendation and deployment advice...",
  "return_analysis": "Explanation of Total Return and CAGR formulas and values...",
  "risk_analysis": "Annual volatility breakdown and comparison against benchmarks...",
  "sharpe_analysis": "Sharpe Ratio evaluation against hedge fund benchmarks...",
  "sortino_analysis": "Downside risk analysis and Sortino ratio evaluation...",
  "drawdown_analysis": "Peak-to-trough drawdown evaluation and capital recovery timeline...",
  "calmar_analysis": "Pain-to-reward ratio evaluation...",
  "trade_quality_analysis": "Win rate, profit factor, average trade return, and trade sample count...",
  "market_regime_suitability": "Analysis of bull, bear, and sideways regime suitability...",
  "strengths": ["List of quantitative strengths..."],
  "weaknesses": ["List of quantitative weaknesses..."],
  "risk_management_assessment": "Stop-loss and position sizing recommendations...",
  "suggested_param_adjustments": "Specific parameter adjustment guidance...",
  "position_sizing_advice": "Kelly criterion and position limit framework...",
  "live_trading_considerations": "Slippage, friction, and paper trading protocol..."
}
```

---

## 4. Evaluation & Example Output

### Sample Input (INFY - SMA Crossover Strategy):
- **CAGR**: `14.8%`
- **Sharpe Ratio**: `0.88`
- **Max Drawdown**: `-12.5%`
- **Win Rate**: `54.2%`
- **Profit Factor**: `1.42`

### Generated JSON Response (Excerpt):

```json
{
  "rating": "Moderate Buy",
  "risk": "Moderate",
  "confidence": 0.88,
  "executive_summary": "The SMA Crossover strategy applied to INFY generated an annualized return (CAGR) of 14.80% with a Sharpe ratio of 0.88 and a contained maximum drawdown of -12.50%. The strategy exhibits a statistically valid signal edge with a Profit Factor of 1.42 across executed trades. It is suitable for controlled deployment with ATR-based stop loss management.",
  "sharpe_analysis": "SHARPE RATIO ANALYSIS: The Sharpe Ratio measures excess return per unit of total risk using the formula: Sharpe = (Mean Daily Return / Std Dev of Daily Returns) * sqrt(252). This strategy achieved a Sharpe Ratio of 0.8800. Benchmarked against hedge fund standards (where >0.5 is acceptable and >1.0 is top-tier), this score indicates solid risk-adjusted compensation.",
  "strengths": [
    "Above-Benchmark CAGR (14.80%): Exceeds historical Nifty 50 average growth.",
    "Contained Drawdown (-12.50%): Limits capital loss during market corrections.",
    "Solid Profit Factor (1.42): Earns Rs.1.42 for every Rs.1.00 lost on losing trades."
  ],
  "weaknesses": [
    "Sensitivity to Whipsaws: SMA strategy may lag during sideways market regimes."
  ]
}
```
