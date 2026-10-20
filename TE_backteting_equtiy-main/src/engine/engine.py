"""
engine.py

Core Event-Driven Execution Engine for QUANTLAB v1.1.
Implements the authoritative order-driven simulation lifecycle:
SIGNAL -> ORDER SEQUENCING (SELL-FIRST, ALPHABETICAL BUYS) -> AFFORDABILITY SIZING
-> LOOKAHEAD-FREE EXECUTION -> INDIAN EQUITY COSTS -> AUDITABLE LEDGER -> TERMINAL LIQUIDATION -> METRICS
"""

import os
import sys
import math
from typing import Tuple, List, Dict, Any, Optional
import numpy as np
import pandas as pd

from .config import BacktestConfig, ExecutionTiming, InsufficientCashPolicy, get_default_baseline_config
from .order import Order, OrderSide, OrderStatus, OrderType
from .position_sizing import PositionSizer
from .execution import ExecutionModel, SlippageModel
from .cost_model import TransactionCostModel, CostSchedule
from .fill import Fill
from .portfolio import PortfolioManager, PositionTracker
from .trade_log import TradeLogger
from .validation import DataValidator
from .result import BacktestResult

# Import metrics calculator
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from metrics import calculate_metrics


class BacktestEngine:
    """
    QUANTLAB Unified Event-Driven Quantitative Backtesting Engine v1.1.
    """

    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or get_default_baseline_config()
        self.validator = DataValidator()
        self.cost_model = TransactionCostModel(
            schedule=CostSchedule(
                brokerage_rate=self.config.brokerage_bps / 10000.0,
                stt_rate_sell=self.config.stt_rate_sell,
                stamp_duty_rate_buy=self.config.stamp_duty_rate_buy,
                gst_rate=self.config.gst_rate,
                exchange_charge_rate=self.config.exchange_charge_rate,
                sebi_charge_rate=self.config.sebi_charge_rate
            )
        )
        self.position_sizer = PositionSizer(self.config, cost_model=self.cost_model)
        self.execution_model = ExecutionModel(self.config)

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
        exited_symbols_this_bar = set()

        for i in range(n_bars):
            date = dates[i]
            close_price = float(close_prices[i])
            exited_symbols_this_bar.clear()

            # ------------------------------------------------------------------
            # STEP 1: Process Pending Orders from Previous Bar(s)
            # Order Sequencing Rule: Process SELL orders first, credit cash,
            # then process BUY orders sorted alphabetically by symbol!
            # ------------------------------------------------------------------
            executable_pending = []
            for order in pending_orders:
                base_price, exec_time, timing_note = self.execution_model.resolve_execution_price(order, i, df)
                if base_price is not None:
                    executable_pending.append((order, base_price, exec_time, timing_note))

            # Separate into sells and buys
            sell_pending = [x for x in executable_pending if x[0].is_sell()]
            buy_pending = [x for x in executable_pending if x[0].is_buy()]

            # Sort buys alphabetically by symbol
            buy_pending.sort(key=lambda x: x[0].symbol)

            # Process SELLS first
            for order, base_price, exec_time, timing_note in sell_pending:
                fill_price, slip_per_share = self.execution_model.slippage_model.calculate_fill_price(
                    base_price=base_price,
                    side=order.side
                )
                total_slippage_cost = slip_per_share * order.quantity
                cost_info = self.cost_model.calculate_cost(order.quantity, fill_price, side=OrderSide.SELL)

                pos = portfolio_mgr.get_position(order.symbol)
                fill_qty = min(order.quantity, pos.quantity)
                if fill_qty <= 0:
                    order.status = OrderStatus.REJECTED
                    trade_logger.log_rejection(
                        timestamp=exec_time or date,
                        symbol=order.symbol,
                        side=order.side,
                        quantity=order.quantity,
                        price=base_price,
                        cash_available=portfolio_mgr.cash,
                        portfolio_value=portfolio_mgr.get_total_portfolio_value(),
                        rejection_reason="No open shares held to sell",
                        order_id=order.order_id
                    )
                    pending_orders.remove(order)
                    continue

                if fill_qty != order.quantity:
                    # Recalculate with adjusted quantity
                    cost_info = self.cost_model.calculate_cost(fill_qty, fill_price, side=OrderSide.SELL)
                    total_slippage_cost = slip_per_share * fill_qty

                # Net value = gross value minus all sell-side costs
                net_value = cost_info["gross_value"] - cost_info["total_cost"]

                fill = Fill(
                    timestamp=exec_time or date,
                    symbol=order.symbol,
                    side=order.side,
                    quantity=fill_qty,
                    reference_price=base_price,
                    fill_price=fill_price,
                    slippage_cost=total_slippage_cost,
                    brokerage=cost_info["brokerage"],
                    stt=cost_info["stt"],
                    stamp_duty=cost_info["stamp_duty"],
                    gst=cost_info["gst"],
                    exchange_charges=cost_info["exchange_charges"],
                    transaction_cost=cost_info["total_cost"],
                    gross_value=cost_info["gross_value"],
                    net_value=net_value,
                    order_id=order.order_id,
                    reason=order.reason
                )

                pos_before = pos.quantity
                cash_before, cash_after, realized_pnl = portfolio_mgr.apply_fill(fill)
                order.status = OrderStatus.FILLED
                order.quantity = fill_qty

                trade_logger.log_trade(
                    fill=fill,
                    cash_before=cash_before,
                    cash_after=cash_after,
                    portfolio_value=portfolio_mgr.get_total_portfolio_value(),
                    position_before=pos_before,
                    position_after=pos.quantity,
                    realized_pnl=realized_pnl,
                    unrealized_pnl=portfolio_mgr.get_unrealized_pnl(),
                    order_status="FILLED",
                    reason=order.reason
                )

                all_fills.append(fill)
                pending_orders.remove(order)
                exited_symbols_this_bar.add(order.symbol)
                entry_price = 0.0

                execution_trace.append(
                    f"BAR {i} ({date.strftime('%Y-%m-%d')}): EXECUTED SELL {fill.quantity} shares of {order.symbol} "
                    f"@ ₹{fill_price:,.2f} [Net Proceeds: ₹{net_value:,.2f}, Realized PnL: ₹{realized_pnl:,.2f}]"
                )

            # Process BUYS second (alphabetically)
            for order, base_price, exec_time, timing_note in buy_pending:
                # Duplicate position check: Ignore BUY if symbol already held
                if portfolio_mgr.has_open_position(order.symbol):
                    order.status = OrderStatus.REJECTED
                    trade_logger.log_rejection(
                        timestamp=exec_time or date,
                        symbol=order.symbol,
                        side=order.side,
                        quantity=order.quantity,
                        price=base_price,
                        cash_available=portfolio_mgr.cash,
                        portfolio_value=portfolio_mgr.get_total_portfolio_value(),
                        rejection_reason="Duplicate Position Prevention: Symbol already held",
                        order_id=order.order_id
                    )
                    pending_orders.remove(order)
                    continue

                # Same-bar re-entry suppression
                if order.symbol in exited_symbols_this_bar:
                    order.status = OrderStatus.REJECTED
                    trade_logger.log_rejection(
                        timestamp=exec_time or date,
                        symbol=order.symbol,
                        side=order.side,
                        quantity=order.quantity,
                        price=base_price,
                        cash_available=portfolio_mgr.cash,
                        portfolio_value=portfolio_mgr.get_total_portfolio_value(),
                        rejection_reason="Same-Event Re-Entry Suppressed: Position exited in same event",
                        order_id=order.order_id
                    )
                    pending_orders.remove(order)
                    continue

                fill_price, slip_per_share = self.execution_model.slippage_model.calculate_fill_price(
                    base_price=base_price,
                    side=order.side
                )

                # Re-verify affordability with current post-sell cash and equity
                port_equity = portfolio_mgr.get_total_portfolio_value()
                fill_qty, sizer_note = self.position_sizer.calculate_buy_quantity(
                    current_cash=portfolio_mgr.cash,
                    price=base_price,
                    portfolio_equity=port_equity,
                    position_size=self.config.position_size
                )

                if fill_qty <= 0:
                    order.status = OrderStatus.REJECTED
                    trade_logger.log_rejection(
                        timestamp=exec_time or date,
                        symbol=order.symbol,
                        side=order.side,
                        quantity=order.quantity,
                        price=base_price,
                        cash_available=portfolio_mgr.cash,
                        portfolio_value=port_equity,
                        rejection_reason=sizer_note,
                        order_id=order.order_id
                    )
                    pending_orders.remove(order)
                    continue

                cost_info = self.cost_model.calculate_cost(fill_qty, fill_price, side=OrderSide.BUY)
                net_debit = cost_info["gross_value"] + cost_info["total_cost"]
                total_slippage_cost = slip_per_share * fill_qty

                # Strict cash invariant check: debit must not exceed cash
                if net_debit > portfolio_mgr.cash:
                    order.status = OrderStatus.REJECTED
                    trade_logger.log_rejection(
                        timestamp=exec_time or date,
                        symbol=order.symbol,
                        side=order.side,
                        quantity=fill_qty,
                        price=base_price,
                        cash_available=portfolio_mgr.cash,
                        portfolio_value=port_equity,
                        rejection_reason=f"Insufficient Cash: Required ₹{net_debit:,.2f} > Cash ₹{portfolio_mgr.cash:,.2f}",
                        order_id=order.order_id
                    )
                    pending_orders.remove(order)
                    continue

                fill = Fill(
                    timestamp=exec_time or date,
                    symbol=order.symbol,
                    side=order.side,
                    quantity=fill_qty,
                    reference_price=base_price,
                    fill_price=fill_price,
                    slippage_cost=total_slippage_cost,
                    brokerage=cost_info["brokerage"],
                    stt=cost_info["stt"],
                    stamp_duty=cost_info["stamp_duty"],
                    gst=cost_info["gst"],
                    exchange_charges=cost_info["exchange_charges"],
                    transaction_cost=cost_info["total_cost"],
                    gross_value=cost_info["gross_value"],
                    net_value=net_debit,
                    order_id=order.order_id,
                    reason=order.reason
                )

                pos = portfolio_mgr.get_position(order.symbol)
                pos_before = pos.quantity
                cash_before, cash_after, _ = portfolio_mgr.apply_fill(fill)
                order.status = OrderStatus.FILLED
                order.quantity = fill_qty

                trade_logger.log_trade(
                    fill=fill,
                    cash_before=cash_before,
                    cash_after=cash_after,
                    portfolio_value=portfolio_mgr.get_total_portfolio_value(),
                    position_before=pos_before,
                    position_after=pos.quantity,
                    realized_pnl=0.0,
                    unrealized_pnl=portfolio_mgr.get_unrealized_pnl(),
                    order_status="FILLED",
                    reason=order.reason
                )

                all_fills.append(fill)
                pending_orders.remove(order)
                entry_price = fill.fill_price

                execution_trace.append(
                    f"BAR {i} ({date.strftime('%Y-%m-%d')}): EXECUTED BUY {fill.quantity} shares of {order.symbol} "
                    f"@ ₹{fill_price:,.2f} [Debit: ₹{net_debit:,.2f}, Remaining Cash: ₹{cash_after:,.2f}]"
                )

            # ------------------------------------------------------------------
            # STEP 2: Update Mark-to-Market Valuation for Bar i
            # ------------------------------------------------------------------
            portfolio_mgr.update_market_prices({symbol: close_price})
            snapshot = portfolio_mgr.snapshot(date)
            portfolio_history.append(snapshot)

            # ------------------------------------------------------------------
            # STEP 3: Completed-Bar Signal Generation & Risk Rules
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

            # Conflicting signals resolution rule: Prioritize SELL, suppress BUY
            if is_signal_buy and is_signal_sell:
                is_signal_buy = False

            # Generate SELL Order (Priority)
            if (is_signal_sell or sl_triggered or tp_triggered) and curr_shares > 0 and not any(o.is_sell() for o in pending_orders):
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

            # Generate BUY Order
            elif is_signal_buy and curr_shares == 0 and not any(o.is_buy() for o in pending_orders):
                port_equity = portfolio_mgr.get_total_portfolio_value()
                buy_qty, sizer_note = self.position_sizer.calculate_buy_quantity(
                    current_cash=portfolio_mgr.cash,
                    price=close_price,
                    portfolio_equity=port_equity,
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

        # ----------------------------------------------------------------------
        # STEP 4: Terminal Policy (Force-close remaining open positions)
        # Force-close open positions at final available price with adverse slippage and costs
        # ----------------------------------------------------------------------
        if self.config.terminal_force_close and n_bars > 0:
            final_date = dates[-1]
            final_close = float(close_prices[-1])
            pos = portfolio_mgr.get_position(symbol)

            if pos.quantity > 0:
                fill_price, slip_per_share = self.execution_model.slippage_model.calculate_fill_price(
                    base_price=final_close,
                    side=OrderSide.SELL
                )
                cost_info = self.cost_model.calculate_cost(pos.quantity, fill_price, side=OrderSide.SELL)
                total_slippage_cost = slip_per_share * pos.quantity
                net_value = cost_info["gross_value"] - cost_info["total_cost"]

                term_fill = Fill(
                    timestamp=final_date,
                    symbol=symbol,
                    side=OrderSide.SELL,
                    quantity=pos.quantity,
                    reference_price=final_close,
                    fill_price=fill_price,
                    slippage_cost=total_slippage_cost,
                    brokerage=cost_info["brokerage"],
                    stt=cost_info["stt"],
                    stamp_duty=cost_info["stamp_duty"],
                    gst=cost_info["gst"],
                    exchange_charges=cost_info["exchange_charges"],
                    transaction_cost=cost_info["total_cost"],
                    gross_value=cost_info["gross_value"],
                    net_value=net_value,
                    order_id="ORD-TERMINAL",
                    reason="Terminal Liquidation"
                )

                pos_before = pos.quantity
                cash_before, cash_after, realized_pnl = portfolio_mgr.apply_fill(term_fill)

                trade_logger.log_trade(
                    fill=term_fill,
                    cash_before=cash_before,
                    cash_after=cash_after,
                    portfolio_value=portfolio_mgr.get_total_portfolio_value(),
                    position_before=pos_before,
                    position_after=0,
                    realized_pnl=realized_pnl,
                    unrealized_pnl=0.0,
                    order_status="FILLED",
                    reason="Terminal Liquidation"
                )
                all_fills.append(term_fill)

                # Update final snapshot
                if portfolio_history:
                    portfolio_mgr.update_market_prices({symbol: final_close})
                    portfolio_history[-1] = portfolio_mgr.snapshot(final_date)

        # Build final portfolio DataFrame
        portfolio_df = pd.DataFrame(portfolio_history)
        if not portfolio_df.empty:
            portfolio_df.set_index("Date", inplace=True)
            portfolio_df["Portfolio Return"] = portfolio_df["Portfolio Value"].pct_change().fillna(0.0)

        trade_log_df = trade_logger.to_dataframe()

        # Compute authoritative Metrics
        metrics = calculate_metrics(
            portfolio_df=portfolio_df,
            trade_log_df=trade_log_df,
            initial_capital=self.config.initial_capital
        )

        total_slippage = sum(f.slippage_cost for f in all_fills)
        total_commission = sum(f.transaction_cost for f in all_fills)
        buy_fills = [f for f in all_fills if f.side == OrderSide.BUY]
        sell_fills = [f for f in all_fills if f.side == OrderSide.SELL]
        avg_trade_val = float(np.mean([f.gross_value for f in all_fills])) if all_fills else 0.0
        avg_slippage = float(np.mean([f.slippage_cost for f in all_fills])) if all_fills else 0.0

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
    initial_capital: float = 10000000.0,
    txn_cost_rate: Optional[float] = None,
    slippage_rate: Optional[float] = None,
    position_size: float = 0.05,
    stop_loss: Optional[float] = None,
    take_profit: Optional[float] = None,
    execution_timing: str = "NEXT_BAR_OPEN",
    insufficient_cash_policy: str = "REJECT",
    allow_short_selling: bool = False,
    **kwargs
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Backward-compatible wrapper executing the upgraded QUANTLAB BacktestEngine.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (portfolio_df, trade_log_df)
    """
    config = get_default_baseline_config()
    config.initial_capital = initial_capital
    config.position_size = position_size
    config.allow_short_selling = allow_short_selling
    config.stop_loss = stop_loss
    config.take_profit = take_profit

    if txn_cost_rate is not None:
        config.brokerage_bps = txn_cost_rate * 10000.0
    if slippage_rate is not None:
        config.slippage_bps = slippage_rate * 10000.0

    if isinstance(execution_timing, str):
        config.execution_timing = ExecutionTiming(execution_timing)
    if isinstance(insufficient_cash_policy, str):
        config.insufficient_cash_policy = InsufficientCashPolicy(insufficient_cash_policy)

    engine = BacktestEngine(config)
    res = engine.run(df)
    return res.portfolio_history, res.trade_log_df
