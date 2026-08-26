"""
analyzer.py

Structured LLM strategy analyzer module.
Produces structured JSON investment ratings and risk assessments from backtest metrics.
"""

import os
import sys
import json
from typing import Dict, Any

# Ensure path imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from llm.prompts import ANALYSIS_SYSTEM_PROMPT, ANALYSIS_USER_PROMPT_TEMPLATE


class LLMStrategyAnalyzer:
    """Analyzes quantitative backtest results using LLM or rule-based synthesis."""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("GEMINI_API_KEY")

    def format_prompt(self, metrics: Dict[str, Any], ticker: str, strategy: str) -> str:
        """Format the user prompt template with calculated metrics."""
        return ANALYSIS_USER_PROMPT_TEMPLATE.format(
            ticker=ticker,
            strategy=strategy,
            total_return=metrics.get("Total Return", 0.0) * 100.0,
            cagr=metrics.get("CAGR", 0.0) * 100.0,
            volatility=metrics.get("Annual Volatility", 0.0) * 100.0,
            sharpe=metrics.get("Sharpe Ratio", 0.0),
            sortino=metrics.get("Sortino Ratio", 0.0),
            calmar=metrics.get("Calmar Ratio", 0.0),
            max_drawdown=metrics.get("Maximum Drawdown", 0.0) * 100.0,
            win_rate=metrics.get("Win Rate", 0.0) * 100.0,
            profit_factor=metrics.get("Profit Factor", 0.0),
            holding_period=metrics.get("Average Holding Period", 0.0),
            num_trades=int(metrics.get("Number of Trades", 0))
        )

    def analyze_performance(self, metrics: Dict[str, Any], ticker: str = "INFY", strategy: str = "SMA") -> Dict[str, Any]:
        """
        Analyze strategy performance metrics and return structured JSON assessment.
        """
        user_prompt = self.format_prompt(metrics, ticker, strategy)

        # 1. Attempt LLM API call if key is present
        if self.api_key:
            # 1a. Try OpenAI SDK
            try:
                import openai
                client = openai.OpenAI(api_key=self.api_key)
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": ANALYSIS_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt}
                    ],
                    response_format={"type": "json_object"}
                )
                res_content = response.choices[0].message.content
                return json.loads(res_content)
            except Exception as e1:
                # 1b. Try Gemini API if available
                try:
                    import google.generativeai as genai
                    genai.configure(api_key=self.api_key)
                    model = genai.GenerativeModel("gemini-1.5-flash")
                    prompt = f"{ANALYSIS_SYSTEM_PROMPT}\n\n{user_prompt}"
                    res = model.generate_content(prompt)
                    clean_text = res.text.strip().replace("```json", "").replace("```", "").strip()
                    return json.loads(clean_text)
                except Exception as e2:
                    print(f"[LLM Analyzer Note] API calls unconfigured or failed ({e1} | {e2}). Using Next-Level Quantitative Rule Engine.")

        # 2. Advanced Quantitative Rule Engine Fallback
        return self._generate_rule_based_analysis(metrics, ticker, strategy)

    def _generate_rule_based_analysis(self, metrics: Dict[str, Any], ticker: str, strategy: str) -> Dict[str, Any]:
        sharpe = metrics.get("Sharpe Ratio", 0.0)
        sortino = metrics.get("Sortino Ratio", 0.0)
        calmar = metrics.get("Calmar Ratio", 0.0)
        cagr = metrics.get("CAGR", 0.0)
        total_return = metrics.get("Total Return", 0.0)
        max_dd = abs(metrics.get("Maximum Drawdown", 0.0))
        win_rate = metrics.get("Win Rate", 0.0)
        profit_factor = metrics.get("Profit Factor", 0.0)
        volatility = metrics.get("Annual Volatility", 0.0)
        avg_holding = metrics.get("Average Holding Period", 0.0)
        n_trades = int(metrics.get("Number of Trades", 0))
        avg_win = metrics.get("Average Win", 0.0)
        avg_loss = abs(metrics.get("Average Loss", 0.0))
        exposure = metrics.get("Exposure", 0.0)
        hit_ratio = metrics.get("Hit Ratio", 0.0)
        turnover = metrics.get("Turnover", 0.0)

        recovery_pct = ((1.0 / (1.0 - max_dd)) - 1.0) * 100.0 if max_dd < 1.0 else 999.0
        avg_rr = (avg_win / avg_loss) if avg_loss > 0.0 else 0.0
        kelly_full = max(0.0, win_rate - (1.0 - win_rate) / max(avg_rr, 0.001))

        # --- Rating Logic ---
        if sharpe > 1.2 and cagr > 0.15 and max_dd < 0.18:
            rating = "Strong Buy"
            confidence = 0.95
        elif sharpe > 0.5 and cagr > 0.06 and max_dd < 0.30:
            rating = "Moderate Buy"
            confidence = 0.88
        elif sharpe >= 0.0:
            rating = "Neutral"
            confidence = 0.78
        else:
            rating = "Underperform"
            confidence = 0.92

        # --- Risk Rating ---
        if max_dd > 0.30 or volatility > 0.25:
            risk = "High"
        elif max_dd > 0.15 or volatility > 0.15:
            risk = "Moderate"
        else:
            risk = "Low"

        # --- Executive Summary ---
        executive_summary = (
            f"The {strategy} strategy applied to {ticker} generated a total return of {total_return*100:.2f}% "
            f"with a CAGR of {cagr*100:.2f}%, placing it in the '{rating}' category with a {risk.lower()} risk profile. "
            f"The Sharpe ratio of {sharpe:.2f} indicates {'strong' if sharpe > 1.0 else 'moderate' if sharpe > 0.5 else 'weak'} "
            f"risk-adjusted performance, while the maximum drawdown of {max_dd*100:.2f}% reflects "
            f"{'contained' if max_dd < 0.15 else 'moderate' if max_dd < 0.25 else 'elevated'} tail risk. "
            f"With {n_trades} executed trades at a win rate of {win_rate*100:.1f}%, the strategy demonstrates "
            f"{'statistically meaningful' if n_trades >= 30 else 'limited but indicative'} signal quality. "
            f"Overall, the strategy {'is suitable for cautious live deployment with strict risk controls' if rating in ['Strong Buy', 'Moderate Buy'] else 'requires further optimization before live deployment'}."
        )

        # --- Return Analysis ---
        return_analysis = (
            f"TOTAL RETURN & CAGR ANALYSIS: Total Return is computed as (Final Portfolio Value - Initial Capital) / Initial Capital = "
            f"{total_return*100:.2f}%. This represents the raw cumulative gain over the entire backtest period without adjusting for time. "
            f"CAGR (Compound Annual Growth Rate) annualizes this return using the formula: CAGR = (Final Value / Initial Capital)^(1 / Years) - 1, "
            f"yielding {cagr*100:.2f}% per annum. A CAGR of {cagr*100:.2f}% should be benchmarked against the Nifty 50 historical average of ~12-14% CAGR "
            f"and risk-free alternatives such as Indian FD rates (~6-7%). "
            f"{'This strategy OUTPERFORMS the Nifty 50 benchmark, suggesting genuine alpha generation.' if cagr > 0.14 else 'This strategy MATCHES or slightly underperforms Nifty 50 returns, implying marginal alpha.' if cagr > 0.10 else 'This strategy UNDERPERFORMS Nifty 50, raising concerns about the strategy edge versus passive investing.'} "
            f"The compounding effect over multi-year periods is significant: at {cagr*100:.2f}% CAGR, an initial capital of Rs.1,00,000 "
            f"would grow to approximately Rs.{100000 * ((1 + cagr) ** 5):,.0f} in 5 years and Rs.{100000 * ((1 + cagr) ** 10):,.0f} in 10 years."
        )

        # --- Risk / Volatility Analysis ---
        risk_analysis = (
            f"ANNUAL VOLATILITY ANALYSIS: Annual Volatility is computed as the standard deviation of daily portfolio returns "
            f"multiplied by the square root of 252 (trading days per year): sigma_annual = std(daily_returns) x sqrt(252) = {volatility*100:.2f}%. "
            f"This annualization factor converts daily variance to an annual scale assuming i.i.d. returns. "
            f"The resulting {volatility*100:.2f}% annualized volatility implies that in a typical year, portfolio returns are expected to "
            f"fluctuate within approximately +/-{volatility*100:.2f}% of the mean (one standard deviation band). "
            f"{'This is BELOW the typical benchmark equity volatility of 15-20% for large-cap Indian equities, indicating a relatively stable portfolio.' if volatility < 0.15 else 'This ALIGNS with typical equity market volatility of 15-20% for Indian large-caps.' if volatility < 0.22 else 'This EXCEEDS normal equity volatility levels, suggesting high day-to-day P&L swings that may challenge investor psychology and margin requirements.'} "
            f"Market exposure stood at {exposure*100:.1f}% of trading days, while the daily hit ratio (% of profitable days) was {hit_ratio*100:.1f}%."
        )

        # --- Sharpe Analysis ---
        sharpe_analysis = (
            f"SHARPE RATIO ANALYSIS: The Sharpe Ratio measures excess return per unit of total risk. "
            f"Formula: Sharpe = (Mean Daily Return / Std Dev of Daily Returns) x sqrt(252), assuming a risk-free rate of 0% (Rf = 0). "
            f"This strategy achieved a Sharpe Ratio of {sharpe:.4f}. "
            f"Industry benchmarks: Sharpe > 1.0 is considered excellent (hedge fund grade), 0.5-1.0 is acceptable for systematic strategies, "
            f"and < 0.5 is considered poor. "
            f"{'VERDICT: EXCELLENT — This Sharpe ratio places the strategy in the top tier of systematic equity strategies.' if sharpe > 1.2 else 'VERDICT: ACCEPTABLE — The strategy earns moderate risk-adjusted returns suitable for deployment with position controls.' if sharpe > 0.5 else 'VERDICT: BELOW BENCHMARK — The strategy does not adequately compensate investors for risk taken. Optimization is required before deployment.'} "
            f"A Sharpe of {sharpe:.2f} means the strategy generates {sharpe:.2f} units of return per unit of risk annually, "
            f"{'which is significantly above market-equivalent risk levels.' if sharpe > 1.0 else 'which is modest and suggests the strategy edge is narrow.' if sharpe > 0.3 else 'suggesting the strategy may be no better than a randomized approach.'}"
        )

        # --- Sortino Analysis ---
        sortino_analysis = (
            f"SORTINO RATIO ANALYSIS: The Sortino Ratio is a refinement of the Sharpe Ratio that penalizes only downside (negative) volatility. "
            f"Formula: Sortino = (Mean Daily Return / Downside Std Dev) x sqrt(252), where Downside Std Dev = std(returns < 0). "
            f"This distinction is critical because investors are primarily harmed by losses, not by upside volatility. "
            f"This strategy achieved a Sortino Ratio of {sortino:.4f}. "
            f"{'The Sortino significantly exceeds the Sharpe (' + str(round(sharpe,2)) + '), confirming that most volatility is upside (favorable) — an excellent signal.' if sortino > sharpe else 'The Sortino is lower than the Sharpe, which indicates downside volatility is disproportionately large relative to upside moves — a warning sign for drawdown risk.'} "
            f"{'A Sortino above 1.0 indicates strong downside protection — this strategy qualifies.' if sortino > 1.0 else 'A Sortino below 1.0 indicates the strategy does not adequately shield capital during losing periods.'}"
        )

        # --- Drawdown Analysis ---
        recovery_gain = (1 / (1 - max_dd) - 1) * 100 if max_dd < 1.0 else 999.0
        drawdown_analysis = (
            f"MAXIMUM DRAWDOWN ANALYSIS: Maximum Drawdown (MDD) measures the largest peak-to-trough decline in portfolio value. "
            f"Formula: MDD = min((Portfolio_t - RunningPeak_t) / RunningPeak_t) across all time t. "
            f"This strategy experienced a Maximum Drawdown of -{max_dd*100:.2f}%. "
            f"In absolute capital terms, if trading with Rs.1,00,000 initial capital, the worst-case loss at any point would have been approximately Rs.{max_dd*100000:,.0f}. "
            f"Industry thresholds for systematic equity strategies: MDD < 10% is exceptional, 10-20% is acceptable, 20-30% is concerning, > 30% is generally unacceptable for institutional capital. "
            f"{'VERDICT: EXCELLENT drawdown control. This strategy maintains portfolio integrity even during adverse market regimes.' if max_dd < 0.10 else 'VERDICT: ACCEPTABLE drawdown within manageable limits for a disciplined investor.' if max_dd < 0.20 else 'VERDICT: ELEVATED drawdown risk. Investors must have strong risk tolerance and sufficient capital to withstand this level of peak-to-trough loss.' if max_dd < 0.30 else 'VERDICT: UNACCEPTABLE drawdown for most institutional mandates. Significant risk management improvements are required.'} "
            f"Recovery from a {max_dd*100:.2f}% drawdown requires a subsequent gain of {recovery_gain:.2f}% to return to the previous peak — "
            f"{'a modest recovery target.' if max_dd < 0.15 else 'a substantial gain that could take considerable time at the current CAGR of ' + str(round(cagr*100, 2)) + '%.'}"
        )

        # --- Calmar Analysis ---
        calmar_analysis = (
            f"CALMAR RATIO ANALYSIS: The Calmar Ratio measures the annual return per unit of maximum drawdown. "
            f"Formula: Calmar = CAGR / |Maximum Drawdown| = {cagr*100:.2f}% / {max_dd*100:.2f}% = {calmar:.4f}. "
            f"This ratio is particularly valued by CTA (commodity trading advisor) and hedge fund managers as it captures the 'pain-to-reward' trade-off. "
            f"Benchmarks: Calmar > 1.0 is excellent, 0.5-1.0 is good, < 0.5 is poor. "
            f"{'VERDICT: EXCELLENT Calmar ratio — the strategy generates strong annual returns relative to its worst historical drawdown.' if calmar > 1.0 else 'VERDICT: ACCEPTABLE Calmar ratio — the strategy offers reasonable return for the drawdown risk.' if calmar > 0.5 else 'VERDICT: POOR Calmar ratio — the strategy generates insufficient return to justify the drawdown risk endured.'} "
            f"At a Calmar of {calmar:.2f}, for every {max_dd*100:.1f}% of drawdown risk, the strategy earns approximately {calmar * max_dd * 100:.1f}% annually."
        )

        # --- Trade Quality Analysis ---
        avg_rr = (avg_win / avg_loss) if avg_loss > 0 else 0.0
        trade_quality_analysis = (
            f"TRADE QUALITY ANALYSIS: This section dissects the strategy's trade-level edge across {n_trades} round-trip trades. "
            f"WIN RATE: Computed as (Number of Winning Trades) / (Total Trades) = {win_rate*100:.2f}%. "
            f"{'A win rate above 50% indicates the strategy generates profitable signals more often than not.' if win_rate > 0.5 else 'A win rate below 50% is not inherently bad — it depends heavily on the average win vs. average loss magnitude.'} "
            f"PROFIT FACTOR: Computed as Gross Profit / Gross Loss = {profit_factor:.4f}. "
            f"A Profit Factor > 1.0 means the strategy is profitable overall. > 1.5 is considered robust; < 1.2 suggests a narrow edge vulnerable to transaction costs. "
            f"{'VERDICT: Robust edge confirmed — the strategy earns significantly more on winners than it loses on losers.' if profit_factor > 1.5 else 'VERDICT: Marginal edge — the strategy is profitable but the profit factor suggests fragility under live market conditions.' if profit_factor > 1.0 else 'VERDICT: Negative edge — the strategy loses more than it earns. Immediate review required.'} "
            f"AVERAGE WIN vs AVERAGE LOSS: Average winning trade returned {avg_win*100:.2f}%, while average losing trade gave back {avg_loss*100:.2f}%, "
            f"yielding a Reward-to-Risk ratio of {avg_rr:.2f}:1. "
            f"{'An R:R above 1.5 with solid win rate constitutes a high-quality edge.' if avg_rr > 1.5 else 'An R:R below 1.0 requires a win rate above 50% to remain profitable.'} "
            f"AVERAGE HOLDING PERIOD of {avg_holding:.1f} days classifies this as a "
            f"{'swing trading' if avg_holding > 5 else 'short-term'} strategy. "
            f"With {n_trades} trades, {'the sample size is statistically meaningful for strategy assessment.' if n_trades >= 30 else 'the sample size is limited; results should be interpreted cautiously and extended backtests are recommended.'}"
        )

        # --- Market Regime ---
        if strategy == "SMA":
            regime = (
                f"The Simple Moving Average (SMA) crossover strategy performs optimally during strong, sustained trending markets "
                f"where price momentum persists over multi-week or multi-month periods. The strategy generates Buy signals when the short-term SMA "
                f"crosses above the long-term SMA (Golden Cross) and Sell signals on the reverse (Death Cross). "
                f"It significantly underperforms in sideways, choppy, or mean-reverting markets where frequent false crossovers generate whipsaw losses. "
                f"The {volatility*100:.2f}% annual volatility and {avg_holding:.1f}-day average holding period suggest the strategy captured "
                f"{'sustained trending periods effectively' if avg_holding > 15 else 'short-term momentum but may be prone to choppy market whipsaws'}."
            )
            param_advice = (
                f"SMA PARAMETER OPTIMIZATION: (1) Test short window ranges of [10, 15, 20, 25] and long window ranges of [40, 50, 60, 75] "
                f"using a walk-forward validation framework to prevent overfitting. (2) Add an ADX (Average Directional Index) filter: "
                f"only take signals when ADX > 25 to ensure you trade only during trending conditions, eliminating false crossovers in ranging markets. "
                f"(3) Implement a volatility filter — reduce position size by 50% when 20-day realized volatility exceeds 25%. "
                f"(4) Consider a volume confirmation rule: require volume to be 1.5x the 20-day average on crossover days. "
                f"These adjustments are expected to improve Sharpe by 0.2-0.4 and reduce drawdown by 3-8%."
            )
        elif strategy == "EMA":
            regime = (
                f"The Exponential Moving Average (EMA) crossover strategy excels in momentum-driven, fast-trending environments. "
                f"EMA assigns exponentially greater weight to recent prices (multiplier = 2/(N+1)), making it more responsive than SMA to recent price changes. "
                f"This responsiveness is beneficial in fast-moving trends but increases whipsaw risk in volatile, choppy conditions. "
                f"The strategy struggles in low-momentum, range-bound markets where EMA lines converge and generate conflicting signals."
            )
            param_advice = (
                f"EMA PARAMETER OPTIMIZATION: (1) Combine with RSI(14) as a trend-strength filter: only enter long when EMA crossover occurs with RSI > 50, and exit when RSI < 45. "
                f"(2) Add volume-weighted entry confirmation — enter only if OBV (On-Balance Volume) is trending up. "
                f"(3) Test MACD alongside EMA crossovers for signal confirmation. "
                f"(4) Use ATR-based stop losses (2x ATR below entry) instead of fixed percentage stops to adapt dynamically to market volatility. "
                f"Expected improvement: Sharpe +0.15-0.30, reduced whipsaw losses by 20-30%."
            )
        elif strategy == "RSI":
            regime = (
                f"The RSI (Relative Strength Index) strategy thrives in mean-reverting, range-bound markets where price oscillates between "
                f"overbought and oversold boundaries. RSI is a momentum oscillator measuring the speed and magnitude of price changes: "
                f"RSI = 100 - (100 / (1 + (Avg Gain / Avg Loss))). The strategy generates Buy signals when RSI crosses below the oversold "
                f"threshold and Sell signals above the overbought threshold. This strategy fundamentally fails in strong trending markets "
                f"where RSI can remain overbought/oversold for extended periods, generating premature exit signals."
            )
            param_advice = (
                f"RSI PARAMETER OPTIMIZATION: (1) Add a 200-day SMA trend filter — only take RSI buy signals when price is above the 200-SMA (bullish regime). "
                f"(2) Experiment with asymmetric thresholds: 25/75 instead of 30/70 to reduce false signals. "
                f"(3) Use RSI(2) for ultra-short-term mean reversion signals in combination with RSI(14) for trend filtering. "
                f"(4) Implement a minimum holding period of 3-5 days to avoid exiting profitable trades too early. "
                f"Expected improvement: false signal reduction by 25-35%, Sharpe improvement of 0.2-0.5."
            )
        else:
            regime = (
                f"This systematic strategy is designed to capture algorithmic signals across multiple market regimes. "
                f"Performance will be strongest in the regime conditions for which the strategy rules were implicitly calibrated during backtesting. "
                f"The {volatility*100:.2f}% annual volatility and {sharpe:.2f} Sharpe ratio suggest the strategy was tested "
                f"during {'relatively favorable trending conditions' if sharpe > 0.5 else 'challenging mixed-regime conditions'}."
            )
            param_advice = (
                f"GENERAL OPTIMIZATION: (1) Conduct a grid search over primary parameter combinations using 5-year rolling walk-forward validation. "
                f"(2) Add risk overlays: volatility-adjusted position sizing, maximum sector concentration limits, and daily VaR monitoring. "
                f"(3) Implement a portfolio-level stop: halt trading if monthly portfolio loss exceeds 5%. "
                f"(4) Consider ensemble approaches combining 2-3 strategy variants to smooth return distributions."
            )

        # --- Strengths ---
        strengths = []
        if sharpe > 0.8:
            strengths.append(
                f"Superior Risk-Adjusted Returns: The Sharpe Ratio of {sharpe:.2f} demonstrates that the strategy generates "
                f"{sharpe:.2f} units of excess return for every unit of risk taken — above the 0.5 minimum threshold for systematic strategies. "
                f"This level of risk-adjusted performance is competitive with many actively managed equity funds."
            )
        if win_rate > 0.50:
            strengths.append(
                f"Consistent Signal Quality: A win rate of {win_rate*100:.1f}% indicates the strategy correctly identifies market direction "
                f"more than half the time, providing a statistically positive expectancy. Combined with a Profit Factor of {profit_factor:.2f}, "
                f"this confirms genuine directional edge."
            )
        if profit_factor > 1.3:
            strengths.append(
                f"Robust Profit Factor of {profit_factor:.2f}: For every Rs.1 lost on losing trades, the strategy earns Rs.{profit_factor:.2f} on winners. "
                f"This {((profit_factor-1)*100):.0f}% gross edge comfortably absorbs transaction costs, slippage, and market impact, "
                f"suggesting the strategy can remain profitable in live trading conditions."
            )
        if sortino > 1.0:
            strengths.append(
                f"Excellent Downside Risk Control: The Sortino Ratio of {sortino:.2f} (significantly above the 1.0 benchmark) confirms "
                f"that the strategy's volatility is predominantly upside. The downside standard deviation is well-controlled, "
                f"meaning losing days are less severe than winning days — a hallmark of superior risk management."
            )
        if max_dd < 0.15:
            strengths.append(
                f"Low Maximum Drawdown of {max_dd*100:.2f}%: The strategy successfully limits peak-to-trough losses, "
                f"requiring only a {recovery_gain:.2f}% recovery gain to reach new equity highs. "
                f"This drawdown level is well within institutional mandate tolerances and preserves compounding capital."
            )
        if cagr > 0.12:
            strengths.append(
                f"Above-Benchmark CAGR of {cagr*100:.2f}%: The strategy delivers annualized returns exceeding the Nifty 50 historical average "
                f"of ~12-14%, implying genuine alpha generation above market beta. Compounding at this rate, Rs.1 lakh grows to "
                f"Rs.{100000*(1+cagr)**5:,.0f} in 5 years."
            )
        if not strengths:
            strengths.append(
                "Systematic & Bias-Free Execution: The strategy generates fully algorithmic, rule-based "
                "signals that eliminate emotional decision-making and ensure consistent, repeatable execution "
                "discipline — a foundational advantage over discretionary trading approaches."
            )

        # --- Weaknesses ---
        weaknesses = []
        if max_dd > 0.20:
            weaknesses.append(
                "Elevated Maximum Drawdown ({:.2f}%): Worst peak-to-trough loss = "
                "Rs.{:,.0f} on a Rs.1,00,000 portfolio. "
                "Recovery requires {:.2f}% subsequent gain — approximately "
                "{:.1f} years at current CAGR of {:.2f}%. "
                "Investors often abandon strategies during drawdowns, causing live underperformance vs. backtest.".format(
                    max_dd * 100, max_dd * 100000, recovery_pct,
                    recovery_pct / 100.0 / max(cagr, 0.001), cagr * 100
                )
            )
        if profit_factor < 1.2:
            weaknesses.append(
                "Narrow Gross Edge (Profit Factor = {:.4f}): Only {:.1f}% gross margin. "
                "Realistic live costs (~0.2-0.3% per round-trip for brokerage + STT + exchange fees) "
                "could eliminate or even reverse the net edge. "
                "A Profit Factor >= 1.5 is strongly recommended before committing live capital.".format(
                    profit_factor, (profit_factor - 1) * 100
                )
            )
        if cagr < 0.08:
            weaknesses.append(
                "Subdued Annual Growth (CAGR = {:.2f}%): Barely exceeds risk-free FD rates (~6-7%). "
                "When accounting for the volatility ({:.2f}%) and drawdown ({:.2f}%) risk taken, "
                "the risk-adjusted case for this strategy over passive index investing is weak.".format(
                    cagr * 100, volatility * 100, max_dd * 100
                )
            )
        if sharpe < 0.5:
            weaknesses.append(
                "Below-Threshold Sharpe Ratio ({:.4f}): Falls below the 0.5 minimum required by "
                "institutional allocators. The strategy does not adequately compensate investors "
                "for the volatility risk taken. "
                "Signal filtering, regime detection, or parameter optimisation is required before deployment.".format(sharpe)
            )
        if n_trades < 20:
            weaknesses.append(
                "Insufficient Trade Sample ({} trades): Statistical reliability requires 30-50+ "
                "completed round-trip trades. With only {} trades, all metrics carry high sampling error. "
                "Extend the backtest period or increase signal frequency.".format(n_trades, n_trades)
            )
        if volatility > 0.22:
            weaknesses.append(
                "High Annual Volatility ({:.2f}%): Significantly above typical equity benchmark "
                "volatility (15-20%). Day-to-day P&L swings may challenge investor psychology and "
                "trigger margin calls. Volatility targeting is strongly recommended.".format(volatility * 100)
            )
        if n_trades > 0 and (n_trades * 0.002) > 0.03:
            weaknesses.append(
                "High Transaction Cost Drag: {} trades x ~0.2% round-trip friction = "
                "{:.2f}% annual drag. This reduces effective live CAGR from {:.2f}% "
                "to approximately {:.2f}%.".format(
                    n_trades, n_trades * 0.002 * 100,
                    cagr * 100, max(0, (cagr - n_trades * 0.002) * 100)
                )
            )
        if not weaknesses:
            weaknesses.append(
                "Backtest-to-Live Performance Gap: Backtest assumes ideal execution at closing prices. "
                "Live trading introduces slippage (0.05-0.15% per trade), spread costs, and execution latency. "
                "With {} trades, estimated live friction reduces returns by ~{:.2f}% annually. "
                "3-6 months paper trading is strongly recommended.".format(
                    n_trades, n_trades * 0.001 * 100
                )
            )

        # --- Risk Management Assessment ---
        if risk == "High":
            risk_mgmt = (
                "HIGH RISK STRATEGY ({:.2f}% MDD, {:.2f}% annual vol): "
                "Switch to ATR-based dynamic stops (2x ATR below entry) — more adaptive than fixed % stops. "
                "Reduce live position size by 40-50% for the first 3 months. "
                "Hard circuit breaker: if portfolio drawdown exceeds 15% from peak, "
                "halt all new positions and conduct a full strategy review. "
                "Daily 95% VaR limit: 2% of total portfolio value. "
                "Volatility overlay: Scale position = Base Size x (Target 15% vol / Realised 20d Vol).".format(
                    max_dd * 100, volatility * 100
                )
            )
            sizing_advice = (
                "HIGH RISK SIZING PROTOCOL: Max 5-10% capital per trade. "
                "Full Kelly fraction = {:.2%} (Win Rate {:.1f}%, R:R {:.2f}:1). "
                "Apply 25% Kelly = {:.2%} to avoid ruin risk. "
                "Max 2-3 simultaneous open positions (concentration control). "
                "Monthly portfolio loss limit: 5% — halt new trades if breached.".format(
                    kelly_full, win_rate * 100, avg_rr, kelly_full * 0.25
                )
            )
        elif risk == "Moderate":
            risk_mgmt = (
                "MODERATE RISK STRATEGY ({:.2f}% MDD, {:.2f}% annual vol): "
                "ATR-based stops (1.5-2x ATR) preferred over fixed % stops. "
                "Volatility-adjusted sizing: reduce position 25% when 20-day realised vol > 20%. "
                "Rolling 3-month Sharpe monitor: if < 0.3, halve all position sizes until recovery. "
                "Monitor inter-position correlation: avoid >0.7 correlated simultaneous positions.".format(
                    max_dd * 100, volatility * 100
                )
            )
            sizing_advice = (
                "MODERATE RISK SIZING PROTOCOL: 15-20% of capital per trade. "
                "Full Kelly = {:.2%}. Apply Half Kelly = {:.2%} for live deployment. "
                "Max 3-5 simultaneous open positions. "
                "5% trailing stop-loss on all profitable open positions. "
                "Monthly portfolio loss limit: 5%.".format(kelly_full, kelly_full * 0.5)
            )
        else:
            risk_mgmt = (
                "LOW RISK STRATEGY ({:.2f}% MDD, {:.2f}% annual vol): "
                "Both metrics within institutional tolerance thresholds. "
                "60-day rolling Sharpe monitor: if drops below 0, halve position sizes immediately. "
                "Regime change alert: if 3-month drawdown exceeds 2x historical average, "
                "pause and review strategy logic. "
                "This strategy is suitable for inclusion in a diversified multi-strategy portfolio.".format(
                    max_dd * 100, volatility * 100
                )
            )
            sizing_advice = (
                "LOW RISK SIZING PROTOCOL: 25-30% per trade is appropriate given the low risk profile. "
                "Full Kelly = {:.2%}. Apply Half Kelly = {:.2%} for optimal capital efficiency. "
                "Monthly rebalancing to prevent exposure drift. "
                "Gradual live scale-up: 50% target month 1, 75% month 2, 100% month 3.".format(
                    kelly_full, kelly_full * 0.5
                )
            )

        # --- Live Trading Considerations ---
        friction_annual = n_trades * 0.002
        live_cagr = max(0.0, cagr - friction_annual)
        live_trading = (
            "LIVE TRADING DEPLOYMENT CONSIDERATIONS:\n"
            "(1) SLIPPAGE & FRICTION: {} trades x ~0.2% round-trip = {:.2f}% annual cost drag. "
            "Effective live CAGR: {:.2f}% -> ~{:.2f}%.\n"
            "(2) OVERFITTING: Out-of-sample degradation of 20-40% is typical. "
            "Validate with 2-year walk-forward out-of-sample test before live capital.\n"
            "(3) REGIME CHANGE: Pause all trading if live portfolio drawdown exceeds 12%. "
            "Conduct full strategy review before resuming.\n"
            "(4) INFRASTRUCTURE: Ensure low-latency broker API (<100ms), automated order management, "
            "and backup connectivity failsafes.\n"
            "(5) PAPER TRADING: Minimum 3-6 months paper trading before live capital allocation. "
            "Monitor daily: Win Rate, Drawdown, Sharpe. "
            "If any metric deviates >20% from backtest, pause and diagnose.".format(
                n_trades, friction_annual * 100, cagr * 100, live_cagr * 100
            )
        )

        # --- Full Recommendation ---
        if rating == "Strong Buy":
            deploy_text = "STRONGLY APPROVED for full deployment after paper trading validation."
        elif rating == "Moderate Buy":
            deploy_text = "APPROVED — start at 50% position size for first 3 months, scale to full after consistent live performance."
        elif rating == "Neutral":
            deploy_text = "CONDITIONAL — optimise parameters and re-evaluate. Do not commit full capital."
        else:
            deploy_text = "NOT APPROVED for live deployment in current form. Fundamental redesign required."

        recommendation = (
            "INVESTMENT RECOMMENDATION — {} | RISK: {} | CONFIDENCE: {:.0f}%\n\n"
            "The {} strategy on {}: CAGR {:.2f}%, Sharpe {:.4f}, MDD {:.2f}%, "
            "Win Rate {:.1f}%, Profit Factor {:.4f}, {} executed trades.\n\n"
            "EDGE ASSESSMENT: {}\n\n"
            "DEPLOYMENT DECISION: {}\n\n"
            "RISK BUDGET: Max 2% daily VaR | Max 12% portfolio drawdown trigger | "
            "Monthly loss limit: 5% before pausing new trades.".format(
                rating.upper(), risk.upper(), confidence * 100,
                strategy, ticker, cagr * 100, sharpe, max_dd * 100,
                win_rate * 100, profit_factor, n_trades,
                "Statistically confirmed positive edge." if profit_factor > 1.3 and n_trades >= 20
                else "Marginal or unconfirmed edge — more data or parameter tuning required.",
                deploy_text
            )
        )

        return {
            "rating":                      rating,
            "investment_rating":           rating,
            "risk":                        risk,
            "risk_score":                  7.5 if risk == "High" else (5.0 if risk == "Moderate" else 2.5),
            "confidence":                  confidence,
            "executive_summary":           executive_summary,
            "recommendation":              recommendation,
            "return_analysis":             return_analysis,
            "risk_analysis":               risk_analysis,
            "sharpe_analysis":             sharpe_analysis,
            "sortino_analysis":            sortino_analysis,
            "drawdown_analysis":           drawdown_analysis,
            "calmar_analysis":             calmar_analysis,
            "trade_quality_analysis":      trade_quality_analysis,
            "market_regime_suitability":   regime,
            "market_regime_fit":           regime,
            "strengths":                   strengths,
            "weaknesses":                  weaknesses,
            "risk_management_assessment":  risk_mgmt,
            "suggested_param_adjustments": param_advice,
            "position_sizing_advice":      sizing_advice,
            "live_trading_considerations": live_trading
        }


if __name__ == "__main__":
    print("Testing LLMStrategyAnalyzer...")
    analyzer = LLMStrategyAnalyzer()
    mock_metrics = {
        "Total Return": 0.25,
        "CAGR": 0.08,
        "Annual Volatility": 0.18,
        "Sharpe Ratio": 0.65,
        "Sortino Ratio": 0.82,
        "Calmar Ratio": 0.45,
        "Maximum Drawdown": -0.18,
        "Win Rate": 0.55,
        "Profit Factor": 1.45,
        "Average Holding Period": 12.5,
        "Average Trade Return": 0.022,
        "Number of Trades": 14,
        "Average Win": 0.04,
        "Average Loss": -0.025,
        "Exposure": 0.62,
        "Hit Ratio": 0.54,
        "Turnover": 2.1
    }
    result = analyzer.analyze_performance(mock_metrics, "INFY", "SMA")
    print("Generated Structured Analysis:")
    print(json.dumps(result, indent=2))
