# Compound Annual Growth Rate (CAGR)

## Definition
Compound Annual Growth Rate (CAGR) is the geometric mean growth rate of an investment over a specified period longer than one year, assuming profit is compounded annually.

## Why It Matters
Unlike simple arithmetic average returns, CAGR accurately captures compounding dynamics and provides a smoothed annual return rate for strategy comparison.

## Core Concept
CAGR measures the hypothetical constant annual return rate required to grow an initial capital amount to the final equity value over a given number of years.

## How It Works
1. Take initial portfolio starting value ($V_{\text{initial}}$).
2. Take final portfolio ending value ($V_{\text{final}}$).
3. Compute total number of years elapsed $N = \frac{\text{Total Days}}{365.25}$.
4. Compute geometric growth factor.

## Formula
$$\text{CAGR} = \left( \frac{V_{\text{final}}}{V_{\text{initial}}} \right)^{\frac{1}{N}} - 1$$

Where:
- $V_{\text{final}}$ = Ending Portfolio Value
- $V_{\text{initial}}$ = Starting Capital (e.g. ₹100,000)
- $N$ = Time period in years

## Example
If initial capital of ₹100,000 grows to ₹180,000 over 3 years:
$$\text{CAGR} = \left( \frac{180,000}{100,000} \right)^{\frac{1}{3}} - 1 = (1.80)^{0.3333} - 1 = 21.64\%$$

## Common Mistakes
- Confusing simple total return with CAGR for multi-year strategy evaluations.
- Calculating CAGR on periods shorter than 1 year without annualized scaling.

## Backtesting Considerations
CAGR should always be evaluated alongside Maximum Drawdown to ensure high growth rate isn't masking unacceptably high risk.

## QUANTLAB Implementation
Computed in `src/metrics.py` as part of the core quantitative evaluation metric suite.

## Related Concepts
- Total Return
- Sharpe Ratio
- Benchmark Comparison
