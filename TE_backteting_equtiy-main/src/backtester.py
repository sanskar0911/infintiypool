"""
backtester.py

Clean equity backtesting engine for historical stock trading simulation.
Supports configurable transaction costs, slippage, position sizing,
stop loss, take profit triggers, and detailed trade log generation.
"""

import os
import sys
from typing import Tuple, List, Dict, Any, Optional
import numpy as np
import pandas as pd


def run_backtest(
    df: pd.DataFrame,
    initial_capital: float = 100000.0,
    txn_cost_rate: float = 0.001,      # 0.1% transaction cost / commission
    slippage_rate: float = 0.0005,     # 0.05% execution slippage
    position_size: float = 1.0,        # Capital allocation fraction per position (0.0 to 1.0)
    stop_loss: Optional[float] = None, # e.g. 0.05 for 5% stop loss
    take_profit: Optional[float] = None, # e.g. 0.10 for 10% take profit
    **kwargs                           # Accept extra kwargs gracefully for compatibility
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run a historical backtest simulation of a trading strategy.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing price data and strategy 'Position' / 'Signal' columns.
    initial_capital : float
        Starting portfolio cash in INR.
    txn_cost_rate : float
        Transaction fee / commission percentage.
    slippage_rate : float
        Slippage percentage per execution.
    position_size : float
        Fraction of available cash allocated per trade (0.0 to 1.0).
    stop_loss : float, optional
        Percentage drawdown from entry price to trigger exit.
    take_profit : float, optional
        Percentage gain from entry price to trigger exit.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (portfolio_df, trade_log_df)
    """
    price_col = "Adj Close" if "Adj Close" in df.columns else "Close"
    required_cols = [price_col, "Position", "Signal"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"DataFrame must contain column: {col}")

    dates = df.index
    n = len(df)

    cash_arr = np.zeros(n)
    shares_arr = np.zeros(n)
    port_val_arr = np.zeros(n)

    current_cash = float(initial_capital)
    current_shares = 0
    entry_price = 0.0

    trades: List[Dict[str, Any]] = []

    positions = df["Position"].values
    signals = df["Signal"].values
    close_prices = df[price_col].values

    for i in range(n):
        date = dates[i]
        close_price = close_prices[i]

        prev_pos = 0 if i == 0 else positions[i - 1]
        curr_pos = positions[i]

        is_signal_buy = (prev_pos == 0 and curr_pos == 1) or (signals[i] == 1 and current_shares == 0)
        is_signal_sell = (prev_pos == 1 and curr_pos == 0) or (signals[i] == -1 and current_shares > 0)

        sl_triggered = False
        tp_triggered = False

        if current_shares > 0 and entry_price > 0:
            current_return = (close_price - entry_price) / entry_price
            if stop_loss is not None and current_return <= -abs(stop_loss):
                sl_triggered = True
            elif take_profit is not None and current_return >= abs(take_profit):
                tp_triggered = True

        # Buy Execution
        if is_signal_buy and current_cash > 0 and current_shares == 0:
            execution_price = close_price * (1.0 + slippage_rate)
            allocated_capital = current_cash * min(max(position_size, 0.01), 1.0)
            cost_per_share = execution_price * (1.0 + txn_cost_rate)
            shares_to_buy = int(np.floor(allocated_capital / cost_per_share))

            if shares_to_buy > 0:
                trade_value = shares_to_buy * execution_price
                commission = trade_value * txn_cost_rate
                total_cost = trade_value + commission

                current_cash -= total_cost
                current_shares += shares_to_buy
                entry_price = execution_price

                port_val = current_cash + (current_shares * close_price)

                trades.append({
                    "Date": date,
                    "BUY/SELL": "BUY",
                    "Type": "BUY",
                    "Price": close_price,
                    "Execution Price": execution_price,
                    "Shares": shares_to_buy,
                    "Value": trade_value,
                    "Commission": commission,
                    "Transaction Cost": commission,
                    "Cash": current_cash,
                    "Remaining Cash": current_cash,
                    "Portfolio Value": port_val,
                    "Reason": "Signal BUY"
                })

        # Sell Execution
        elif (is_signal_sell or sl_triggered or tp_triggered) and current_shares > 0:
            execution_price = close_price * (1.0 - slippage_rate)
            trade_value = current_shares * execution_price
            commission = trade_value * txn_cost_rate
            net_proceeds = trade_value - commission

            current_cash += net_proceeds
            shares_sold = current_shares
            current_shares = 0
            entry_price = 0.0

            reason = "Stop Loss" if sl_triggered else ("Take Profit" if tp_triggered else "Signal SELL")
            port_val = current_cash

            trades.append({
                "Date": date,
                "BUY/SELL": "SELL",
                "Type": "SELL",
                "Price": close_price,
                "Execution Price": execution_price,
                "Shares": shares_sold,
                "Value": trade_value,
                "Commission": commission,
                "Transaction Cost": commission,
                "Cash": current_cash,
                "Remaining Cash": current_cash,
                "Portfolio Value": port_val,
                "Reason": reason
            })

        cash_arr[i] = current_cash
        shares_arr[i] = current_shares
        port_val_arr[i] = current_cash + (current_shares * close_price)

    portfolio_df = pd.DataFrame(index=dates)
    portfolio_df["Cash"] = cash_arr
    portfolio_df["Shares"] = shares_arr
    portfolio_df["Portfolio Value"] = port_val_arr
    portfolio_df["Portfolio Return"] = portfolio_df["Portfolio Value"].pct_change().fillna(0.0)

    if trades:
        trade_log_df = pd.DataFrame(trades)
        trade_log_df.set_index("Date", inplace=True)
    else:
        trade_log_df = pd.DataFrame(columns=[
            "BUY/SELL", "Type", "Price", "Execution Price", "Shares", "Value",
            "Commission", "Transaction Cost", "Cash", "Remaining Cash", "Portfolio Value", "Reason"
        ])
        trade_log_df.index.name = "Date"

    return portfolio_df, trade_log_df


if __name__ == "__main__":
    print("Testing clean backtester.py engine...")
    try:
        from strategy import generate_signals

        sample_path = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "RELIANCE.csv")
        if os.path.exists(sample_path):
            df_sample = pd.read_csv(sample_path, index_col="Date")
            df_sample.index = pd.to_datetime(df_sample.index)

            df_signals = generate_signals(df_sample)
            portfolio_df, trade_log_df = run_backtest(
                df_signals,
                position_size=0.2,
                stop_loss=0.05,
                take_profit=0.10
            )
            print(f"Engine test successful! Trades executed: {len(trade_log_df)}")
            print(trade_log_df.head(4))
    except Exception as e:
        print(f"Self-test notification: {e}")
