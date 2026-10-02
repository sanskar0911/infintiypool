"""Performance Metrics Calculation Engine for QuantLab Backtesting Engine."""

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats

from .config import BacktestConfig
from .drawdown import DrawdownMetrics, calculate_drawdown_metrics


@dataclass
class BacktestMetrics:
    """Comprehensive performance metrics matching QuantLab specification."""
    # Trade Statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    even_trades: int
    win_rate: float
    loss_rate: float

    # Returns
    initial_capital: float
    final_equity: float
    total_pnl: float
    realized_pnl: float
    total_return_pct: float
    avg_pnl_per_trade: float
    median_pnl: float
    cagr: float
    years: float

    # Profitability
    gross_profit: float
    gross_loss: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    risk_reward_ratio: float
    expectancy: float

    # Risk Metrics
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    max_drawdown_pct: float
    max_drawdown_duration: int
    drawdown_peak_date: str
    drawdown_trough_date: str
    recovery_date: str
    volatility: float  # annualized volatility of daily equity returns

    # Skewness & Distribution
    skewness: float
    mean_pnl: float
    pnl_std: float

    # Streaks
    max_win_streak: int
    max_loss_streak: int
    max_consecutive_profit: float
    max_consecutive_loss: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def calculate_skewness(pnl_series: pd.Series) -> float:
    """
    Sample-adjusted Fisher-Pearson skewness:
    n / ((n-1)(n-2)) * sum(((xi - mean) / s)^3)
    """
    n = len(pnl_series)
    if n < 3:
        return 0.0

    mean_val = pnl_series.mean()
    s = pnl_series.std(ddof=1)
    if s == 0 or np.isnan(s):
        return 0.0

    m3 = np.sum(((pnl_series - mean_val) / s) ** 3)
    factor = n / ((n - 1) * (n - 2))
    return float(factor * m3)


def calculate_streaks(pnl_series: pd.Series) -> Tuple[int, int, float, float]:
    """
    Calculate maximum winning streak, losing streak,
    and maximum consecutive profit / loss in currency.
    """
    max_win_streak = 0
    max_loss_streak = 0
    max_consec_profit = 0.0
    max_consec_loss = 0.0

    cur_win_streak = 0
    cur_loss_streak = 0
    cur_profit_sum = 0.0
    cur_loss_sum = 0.0

    for pnl in pnl_series:
        if pnl > 0:
            cur_win_streak += 1
            cur_profit_sum += pnl
            if cur_win_streak > max_win_streak:
                max_win_streak = cur_win_streak
            if cur_profit_sum > max_consec_profit:
                max_consec_profit = cur_profit_sum

            cur_loss_streak = 0
            cur_loss_sum = 0.0
        elif pnl < 0:
            cur_loss_streak += 1
            cur_loss_sum += pnl  # negative
            if cur_loss_streak > max_loss_streak:
                max_loss_streak = cur_loss_streak
            if cur_loss_sum < max_consec_loss:
                max_consec_loss = cur_loss_sum

            cur_win_streak = 0
            cur_profit_sum = 0.0
        else:
            # PnL == 0 resets active streaks
            cur_win_streak = 0
            cur_loss_streak = 0
            cur_profit_sum = 0.0
            cur_loss_sum = 0.0

    return max_win_streak, max_loss_streak, max_consec_profit, max_consec_loss


def calculate_metrics(
    trades: pd.DataFrame,
    daily_df: pd.DataFrame,
    config: BacktestConfig
) -> BacktestMetrics:
    """Calculates all QuantLab backtest metrics."""
    n_trades = len(trades)
    if n_trades == 0:
        raise ValueError("Cannot calculate metrics on empty trade log.")

    pnl_series = trades["pnl"]
    winning_mask = pnl_series > 0
    losing_mask = pnl_series < 0
    even_mask = pnl_series == 0

    winning_trades = int(winning_mask.sum())
    losing_trades = int(losing_mask.sum())
    even_trades = int(even_mask.sum())

    win_rate = winning_trades / n_trades if n_trades > 0 else 0.0
    loss_rate = losing_trades / n_trades if n_trades > 0 else 0.0

    # Returns
    initial_cap = config.initial_capital
    total_pnl = float(pnl_series.sum())
    final_equity = initial_cap + total_pnl
    realized_pnl = total_pnl
    total_return_pct = (total_pnl / initial_cap) * 100.0

    avg_pnl = total_pnl / n_trades if n_trades > 0 else 0.0
    median_pnl = float(pnl_series.median())

    # Time span & CAGR
    earliest_entry = trades["entry_date"].min()
    latest_exit = trades["exit_date"].max()
    days_span = (latest_exit - earliest_entry).days
    years = days_span / 365.25 if days_span > 0 else 1.0

    cagr = ((final_equity / initial_cap) ** (1.0 / years) - 1.0) if (years > 0 and final_equity > 0) else 0.0

    # Profitability
    gross_profit = float(pnl_series[winning_mask].sum()) if winning_trades > 0 else 0.0
    gross_loss = float(pnl_series[losing_mask].sum()) if losing_trades > 0 else 0.0  # negative

    profit_factor = (gross_profit / abs(gross_loss)) if abs(gross_loss) > 0 else 999.0
    avg_win = (gross_profit / winning_trades) if winning_trades > 0 else 0.0
    avg_loss = (gross_loss / losing_trades) if losing_trades > 0 else 0.0  # negative
    risk_reward_ratio = (avg_win / abs(avg_loss)) if abs(avg_loss) > 0 else 999.0

    # Expectancy: (Win Rate * Avg Win) + (Loss Rate * Avg Loss)
    # Average Loss is negative
    expectancy = (win_rate * avg_win) + (loss_rate * avg_loss)

    # Drawdown metrics
    dd_metrics = calculate_drawdown_metrics(daily_df)

    # Calmar Ratio: CAGR / Max Drawdown % (using positive magnitude)
    abs_max_dd_pct = abs(dd_metrics.max_drawdown_pct)
    calmar = (cagr / (abs_max_dd_pct / 100.0)) if abs_max_dd_pct > 0 else 0.0

    # Skewness
    skew = calculate_skewness(pnl_series)
    mean_pnl_val = float(pnl_series.mean())
    std_pnl_val = float(pnl_series.std(ddof=1)) if n_trades > 1 else 0.0

    # Streaks
    max_win_streak, max_loss_streak, max_consec_profit, max_consec_loss = calculate_streaks(pnl_series)

    # Risk-adjusted return annualization
    trades_per_year = n_trades / years if years > 0 else n_trades
    trade_returns = (trades["calc_return_pct"] / 100.0)

    trade_ann_return = trade_returns.mean() * trades_per_year
    trade_ann_std = trade_returns.std(ddof=1) * np.sqrt(trades_per_year) if n_trades > 1 else 1.0

    rf = config.risk_free_rate
    sharpe = (trade_ann_return - rf) / trade_ann_std if trade_ann_std > 0 else 0.0

    # Downside deviation for Sortino (target = 0 or negative trades)
    downside_trades = trade_returns.apply(lambda x: x if x < 0 else 0.0)
    ann_downside_std = np.sqrt(np.sum(downside_trades ** 2) / (n_trades - 1)) * np.sqrt(trades_per_year) if n_trades > 1 else 1.0
    sortino = (trade_ann_return - rf) / ann_downside_std if ann_downside_std > 0 else 0.0

    # Daily equity return volatility
    daily_returns = daily_df["equity"].pct_change().dropna()
    daily_vol_ann = float(daily_returns.std(ddof=1) * np.sqrt(252)) if len(daily_returns) > 1 else 0.0

    return BacktestMetrics(
        total_trades=n_trades,
        winning_trades=winning_trades,
        losing_trades=losing_trades,
        even_trades=even_trades,
        win_rate=win_rate,
        loss_rate=loss_rate,
        initial_capital=initial_cap,
        final_equity=final_equity,
        total_pnl=total_pnl,
        realized_pnl=realized_pnl,
        total_return_pct=total_return_pct,
        avg_pnl_per_trade=avg_pnl,
        median_pnl=median_pnl,
        cagr=cagr,
        years=years,
        gross_profit=gross_profit,
        gross_loss=gross_loss,
        profit_factor=profit_factor,
        avg_win=avg_win,
        avg_loss=avg_loss,
        risk_reward_ratio=risk_reward_ratio,
        expectancy=expectancy,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        calmar_ratio=calmar,
        max_drawdown=dd_metrics.max_drawdown,
        max_drawdown_pct=dd_metrics.max_drawdown_pct,
        max_drawdown_duration=dd_metrics.max_drawdown_duration,
        drawdown_peak_date=dd_metrics.peak_date,
        drawdown_trough_date=dd_metrics.trough_date,
        recovery_date=dd_metrics.recovery_date,
        volatility=daily_vol_ann,
        skewness=skew,
        mean_pnl=mean_pnl_val,
        pnl_std=std_pnl_val,
        max_win_streak=max_win_streak,
        max_loss_streak=max_loss_streak,
        max_consecutive_profit=max_consec_profit,
        max_consecutive_loss=max_consec_loss
    )
