"""
engine.py

Core Backtesting Engine for QUANTLAB.
Implements bar-by-bar event simulation:
SIGNAL -> POSITION SIZING -> ORDER -> EXECUTION -> FILL -> PORTFOLIO STATE -> METRICS
"""

import os
import sys
import math
from typing import Tuple, List, Dict, Any, Optional
import numpy as np
import pandas as pd

from .config import BacktestConfig, ExecutionTiming, InsufficientCashPolicy
from .order import Order, OrderSide, OrderStatus, OrderType
from .position_sizing import PositionSizer
from .execution import ExecutionModel, SlippageModel
from .cost_model import TransactionCostModel
from .fill import Fill
from .portfolio import PortfolioManager, PositionTracker
from .trade_log import TradeLogger
from .validation import DataValidator
from .result import BacktestResult

# Import metrics calculator from root src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from metrics import calculate_metrics


class BacktestEngine:
    """
    Realistic Order-Driven Quantitative Backtesting Engine.
    """

    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()
        self.validator = DataValidator()
        self.position_sizer = PositionSizer(self.config)
        self.execution_model = ExecutionModel(self.config)
        self.cost_model = TransactionCostModel(commission_bps=self.config.commission_bps)

    def run(self, df: pd.DataFrame, symbol: str = "ASSET") -> BacktestResult:
        """
        Execute full bar-by-bar backtest simulation.

        Parameters
        ----------
        df : pd.DataFrame
            Stock price DataFrame with DatetimeIndex and required OHLC/Signal columns.
        symbol : str
            Ticker symbol.

        Returns
        -------
        BacktestResult
            Container object with portfolio time series, orders, fills, metrics, and trace.
        """
        # 1. Data Validation
        self.validator.validate(df)

        price_col = "Adj Close" if "Adj Close" in df.columns else self.config.price_column
        if price_col not in df.columns:
            price_col = "Close"

        dates = df.index
        n_bars = len(df)

        portfolio_mgr = PortfolioManager(
            initial_capital=self.config.initial_capital,
            allow_short=self.config.allow_short_selling
        )
        trade_logger = TradeLogger()

        all_orders: List[Order] = []
        all_fills: List[Fill] = []
        pending_orders: List[Order] = []
        execution_trace: List[str] = []
        portfolio_history: List[Dict[str, Any]] = []

        positions_arr = df["Position"].values if "Position" in df.columns else np.zeros(n_bars)
        signals_arr = df["Signal"].values if "Signal" in df.columns else np.zeros(n_bars)
        close_prices = df[price_col].values

        entry_price = 0.0

        for i in range(n_bars):
            date = dates[i]
            close_price = float(close_prices[i])

            # ------------------------------------------------------------------
            # STEP 1: Process and Execute Pending Orders from Previous Bar(s)
            # ------------------------------------------------------------------
            orders_to_remove = []
            for order in pending_orders:
                base_price, exec_time, timing_note = self.execution_model.resolve_execution_price(order, i, df)

                if base_price is not None:
                    # Calculate adverse slippage fill price
                    fill_price, slip_per_share = self.execution_model.slippage_model.calculate_fill_price(
                        base_price=base_price,
                        side=order.side
                    )

                    total_slippage_cost = slip_per_share * order.quantity

                    # Calculate transaction costs
                    cost_info = self.cost_model.calculate_cost(order.quantity, fill_price)
                    txn_cost = cost_info["total_cost"]

                    fill_qty = order.quantity
                    net_value = 0.0

                    # Apply cash constraints for BUY orders
                    if order.is_buy():
                        net_value = cost_info["gross_value"] + txn_cost
                        if net_value > portfolio_mgr.cash:
                            if self.config.insufficient_cash_policy == InsufficientCashPolicy.REJECT:
                                order.status = OrderStatus.REJECTED
                                execution_trace.append(
                                    f"BAR {i} ({date.strftime('%Y-%m-%d')}): REJECTED BUY Order {order.order_id} - "
                                    f"Required ₹{net_value:,.2f} > Available Cash ₹{portfolio_mgr.cash:,.2f}"
                                )
                                orders_to_remove.append(order)
                                continue
                            else:  # REDUCE_QUANTITY
                                # Adjust quantity to affordable amount
                                cost_per_share = fill_price * (1.0 + self.config.commission_bps / 10000.0)
                                fill_qty = math.floor(portfolio_mgr.cash / cost_per_share)
                                if fill_qty <= 0:
                                    order.status = OrderStatus.REJECTED
                                    execution_trace.append(
                                        f"BAR {i} ({date.strftime('%Y-%m-%d')}): REJECTED BUY Order {order.order_id} - "
                                        f"Insufficient cash for 1 share"
                                    )
                                    orders_to_remove.append(order)
                                    continue
                                # Recalculate cost with reduced quantity
                                cost_info = self.cost_model.calculate_cost(fill_qty, fill_price)
                                txn_cost = cost_info["total_cost"]
                                total_slippage_cost = slip_per_share * fill_qty
                                net_value = cost_info["gross_value"] + txn_cost

                    elif order.is_sell():
                        pos = portfolio_mgr.get_position(symbol)
                        if fill_qty > pos.quantity:
                            fill_qty = pos.quantity
                        if fill_qty <= 0:
                            order.status = OrderStatus.REJECTED
                            orders_to_remove.append(order)
                            continue
                        cost_info = self.cost_model.calculate_cost(fill_qty, fill_price)
                        txn_cost = cost_info["total_cost"]
                        total_slippage_cost = slip_per_share * fill_qty
                        net_value = cost_info["gross_value"] - txn_cost

                    # Generate Fill Object
                    fill = Fill(
                        timestamp=exec_time or date,
                        symbol=symbol,
                        side=order.side,
                        quantity=fill_qty,
                        reference_price=base_price,
                        fill_price=fill_price,
                        slippage_cost=total_slippage_cost,
                        transaction_cost=txn_cost,
                        gross_value=cost_info["gross_value"],
                        net_value=net_value,
                        order_id=order.order_id,
                        reason=order.reason
                    )

                    # Update Portfolio State
                    cash_before, cash_after = portfolio_mgr.apply_fill(fill)
                    order.status = OrderStatus.FILLED
                    order.quantity = fill_qty

                    # Log Trade
                    port_val = portfolio_mgr.get_total_portfolio_value()
                    trade_logger.log_trade(
                        fill=fill,
                        cash_before=cash_before,
                        cash_after=cash_after,
                        portfolio_value=port_val,
                        reason=order.reason
                    )

                    all_fills.append(fill)
                    orders_to_remove.append(order)

                    if fill.side == OrderSide.BUY:
                        entry_price = fill.fill_price
                    else:
                        entry_price = 0.0

                    execution_trace.append(
                        f"BAR {i} ({date.strftime('%Y-%m-%d')}): EXECUTED {fill.side} {fill.quantity} shares of {symbol} "
                        f"@ ₹{fill_price:,.2f} [Ref: ₹{base_price:,.2f}, Slip: ₹{total_slippage_cost:,.2f}, Fee: ₹{txn_cost:,.2f}] "
                        f"Timing: {timing_note}"
                    )

            for order in orders_to_remove:
                if order in pending_orders:
                    pending_orders.remove(order)

            # ------------------------------------------------------------------
            # STEP 2: Update Mark-to-Market Valuation for Bar i
            # ------------------------------------------------------------------
            portfolio_mgr.update_market_prices({symbol: close_price})
            snapshot = portfolio_mgr.snapshot(date)
            portfolio_history.append(snapshot)

            # ------------------------------------------------------------------
            # STEP 3: Signal Generation & Risk Trigger Evaluation
            # ------------------------------------------------------------------
            pos_tracker = portfolio_mgr.get_position(symbol)
            curr_shares = pos_tracker.quantity

            prev_pos = 0 if i == 0 else positions_arr[i - 1]
            curr_pos = positions_arr[i]

            is_signal_buy = (prev_pos == 0 and curr_pos == 1) or (signals_arr[i] == 1 and curr_shares == 0)
            is_signal_sell = (prev_pos == 1 and curr_pos == 0) or (signals_arr[i] == -1 and curr_shares > 0)

            sl_triggered = False
            tp_triggered = False

            if curr_shares > 0 and entry_price > 0:
                unrealized_return = (close_price - entry_price) / entry_price
                if self.config.stop_loss is not None and unrealized_return <= -abs(self.config.stop_loss):
                    sl_triggered = True
                elif self.config.take_profit is not None and unrealized_return >= abs(self.config.take_profit):
                    tp_triggered = True

            # Generate BUY Order
            if is_signal_buy and curr_shares == 0 and not any(o.is_buy() for o in pending_orders):
                buy_qty, sizer_note = self.position_sizer.calculate_buy_quantity(
                    current_cash=portfolio_mgr.cash,
                    price=close_price,
                    position_size=self.config.position_size
                )
                if buy_qty > 0:
                    buy_order = Order(
                        timestamp=date,
                        symbol=symbol,
                        side=OrderSide.BUY,
                        quantity=buy_qty,
                        order_type=OrderType.MARKET,
                        reference_price=close_price,
                        signal_bar_idx=i,
                        reason="Signal BUY"
                    )
                    all_orders.append(buy_order)
                    pending_orders.append(buy_order)
                    execution_trace.append(
                        f"BAR {i} ({date.strftime('%Y-%m-%d')}): GENERATED ORDER {buy_order.order_id} - "
                        f"BUY {buy_qty} shares @ Ref Price ₹{close_price:,.2f}"
                    )

            # Generate SELL Order (Signal or Risk Trigger)
            elif (is_signal_sell or sl_triggered or tp_triggered) and curr_shares > 0 and not any(o.is_sell() for o in pending_orders):
                sell_qty, _ = self.position_sizer.calculate_sell_quantity(curr_shares)
                if sell_qty > 0:
                    reason = "Stop Loss" if sl_triggered else ("Take Profit" if tp_triggered else "Signal SELL")
                    sell_order = Order(
                        timestamp=date,
                        symbol=symbol,
                        side=OrderSide.SELL,
                        quantity=sell_qty,
                        order_type=OrderType.MARKET,
                        reference_price=close_price,
                        signal_bar_idx=i,
                        reason=reason
                    )
                    all_orders.append(sell_order)
                    pending_orders.append(sell_order)
                    execution_trace.append(
                        f"BAR {i} ({date.strftime('%Y-%m-%d')}): GENERATED ORDER {sell_order.order_id} - "
                        f"SELL {sell_qty} shares @ Ref Price ₹{close_price:,.2f} [Reason: {reason}]"
                    )

            # Immediate execution check for SAME_BAR_CLOSE mode
            if self.config.execution_timing == ExecutionTiming.SAME_BAR_CLOSE and pending_orders:
                for order in list(pending_orders):
                    base_price, exec_time, timing_note = self.execution_model.resolve_execution_price(order, i, df)
                    if base_price is not None:
                        fill_price, slip_per_share = self.execution_model.slippage_model.calculate_fill_price(base_price, order.side)
                        total_slippage_cost = slip_per_share * order.quantity
                        cost_info = self.cost_model.calculate_cost(order.quantity, fill_price)
                        txn_cost = cost_info["total_cost"]
                        fill_qty = order.quantity
                        
                        if order.is_buy():
                            net_value = cost_info["gross_value"] + txn_cost
                        else:
                            net_value = cost_info["gross_value"] - txn_cost

                        fill = Fill(
                            timestamp=exec_time or date,
                            symbol=symbol,
                            side=order.side,
                            quantity=fill_qty,
                            reference_price=base_price,
                            fill_price=fill_price,
                            slippage_cost=total_slippage_cost,
                            transaction_cost=txn_cost,
                            gross_value=cost_info["gross_value"],
                            net_value=net_value,
                            order_id=order.order_id,
                            reason=order.reason
                        )
                        cash_before, cash_after = portfolio_mgr.apply_fill(fill)
                        order.status = OrderStatus.FILLED
                        port_val = portfolio_mgr.get_total_portfolio_value()
                        trade_logger.log_trade(fill, cash_before, cash_after, port_val, order.reason)
                        all_fills.append(fill)
                        pending_orders.remove(order)

        # Build final portfolio DataFrame
        portfolio_df = pd.DataFrame(portfolio_history)
        portfolio_df.set_index("Date", inplace=True)
        portfolio_df["Portfolio Return"] = portfolio_df["Portfolio Value"].pct_change().fillna(0.0)

        trade_log_df = trade_logger.to_dataframe()

        # Compute Metrics
        metrics = calculate_metrics(portfolio_df, trade_log_df, initial_capital=self.config.initial_capital)

        # Compute Execution Statistics
        total_slippage = sum(f.slippage_cost for f in all_fills)
        total_commission = sum(f.transaction_cost for f in all_fills)
        buy_fills = [f for f in all_fills if f.side == OrderSide.BUY]
        sell_fills = [f for f in all_fills if f.side == OrderSide.SELL]
        avg_trade_val = np.mean([f.gross_value for f in all_fills]) if all_fills else 0.0
        avg_slippage = np.mean([f.slippage_cost for f in all_fills]) if all_fills else 0.0

        execution_stats = {
            "Total Orders": len(all_orders),
            "Total Fills": len(all_fills),
            "Buy Fills": len(buy_fills),
            "Sell Fills": len(sell_fills),
            "Total Transaction Costs": total_commission,
            "Total Slippage Cost": total_slippage,
            "Average Slippage": avg_slippage,
            "Average Trade Value": avg_trade_val,
            "Turnover": metrics.get("Turnover", 0.0),
            "Average Exposure": metrics.get("Exposure", 0.0)
        }

        return BacktestResult(
            portfolio_history=portfolio_df,
            orders=all_orders,
            fills=all_fills,
            trade_log_df=trade_log_df,
            metrics=metrics,
            benchmark={},
            execution_statistics=execution_stats,
            configuration=self.config,
            execution_trace=execution_trace
        )


def run_backtest(
    df: pd.DataFrame,
    initial_capital: float = 100000.0,
    txn_cost_rate: float = 0.001,
    slippage_rate: float = 0.0005,
    position_size: float = 1.0,
    stop_loss: Optional[float] = None,
    take_profit: Optional[float] = None,
    execution_timing: str = "NEXT_BAR_OPEN",
    insufficient_cash_policy: str = "REDUCE_QUANTITY",
    allow_short_selling: bool = False,
    **kwargs
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Backward-compatible wrapper function executing the upgraded BacktestEngine.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (portfolio_df, trade_log_df)
    """
    config = BacktestConfig(
        initial_capital=initial_capital,
        position_size=position_size,
        commission_bps=txn_cost_rate * 10000.0,
        slippage_bps=slippage_rate * 10000.0,
        execution_timing=ExecutionTiming(execution_timing) if isinstance(execution_timing, str) else execution_timing,
        insufficient_cash_policy=InsufficientCashPolicy(insufficient_cash_policy) if isinstance(insufficient_cash_policy, str) else insufficient_cash_policy,
        allow_short_selling=allow_short_selling,
        stop_loss=stop_loss,
        take_profit=take_profit
    )

    engine = BacktestEngine(config)
    res = engine.run(df)
    return res.portfolio_history, res.trade_log_df
