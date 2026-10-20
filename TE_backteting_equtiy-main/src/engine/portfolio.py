"""
portfolio.py

Portfolio State and Position Accounting Engine for QUANTLAB v1.1.
Enforces strict long-only cash invariants, tracks mark-to-market valuations,
net realized P&L with entry/exit transaction cost deductions, and daily equity curves.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
from .fill import Fill
from .order import OrderSide


@dataclass
class PositionTracker:
    """
    Tracks state for a single held stock symbol.

    Attributes
    ----------
    symbol : str
        Stock ticker symbol.
    quantity : int
        Number of open integer shares held.
    avg_entry_price : float
        Weighted average purchase price per share (after slippage).
    current_price : float
        Latest mark-to-market closing price.
    total_buy_fees : float
        Accumulated transaction fees paid to enter current open position.
    """
    symbol: str
    quantity: int = 0
    avg_entry_price: float = 0.0
    current_price: float = 0.0
    total_buy_fees: float = 0.0

    @property
    def market_value(self) -> float:
        return float(self.quantity) * self.current_price

    @property
    def unrealized_pnl(self) -> float:
        if self.quantity <= 0 or self.avg_entry_price <= 0:
            return 0.0
        # MTM value minus cost basis (excluding exit fee estimate or with exit fee)
        return (self.current_price - self.avg_entry_price) * self.quantity

    @property
    def unrealized_pnl_pct(self) -> float:
        if self.quantity <= 0 or self.avg_entry_price <= 0:
            return 0.0
        return (self.current_price - self.avg_entry_price) / self.avg_entry_price

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "quantity": self.quantity,
            "avg_entry_price": self.avg_entry_price,
            "current_price": self.current_price,
            "market_value": self.market_value,
            "unrealized_pnl": self.unrealized_pnl,
            "unrealized_pnl_pct": self.unrealized_pnl_pct
        }


class PortfolioManager:
    """
    Authoritative Portfolio Accounting Manager.
    Guarantees cash invariants, tracks fills, realized P&L, and daily MTM equity.
    """

    def __init__(self, initial_capital: float = 10000000.0, allow_short: bool = False):
        self.initial_capital = float(initial_capital)
        self.cash = float(initial_capital)
        self.allow_short = allow_short
        self.positions: Dict[str, PositionTracker] = {}
        self.realized_pnl: float = 0.0

    def get_position(self, symbol: str) -> PositionTracker:
        if symbol not in self.positions:
            self.positions[symbol] = PositionTracker(symbol=symbol)
        return self.positions[symbol]

    def has_open_position(self, symbol: str) -> bool:
        """Check if symbol currently has open non-zero holdings."""
        if symbol in self.positions:
            return self.positions[symbol].quantity > 0
        return False

    def update_market_prices(self, current_prices: Dict[str, float]):
        """Update mark-to-market prices for all open positions."""
        for symbol, price in current_prices.items():
            if symbol in self.positions:
                self.positions[symbol].current_price = float(price)

    def apply_fill(self, fill: Fill) -> Tuple[float, float, float]:
        """
        Process executed trade fill, updating cash, position quantity, average entry price,
        and realized P&L.

        Returns
        -------
        Tuple[float, float, float]
            (cash_before, cash_after, trade_realized_pnl)
        """
        cash_before = self.cash
        pos = self.get_position(fill.symbol)
        trade_realized_pnl = 0.0

        is_buy = (fill.side == OrderSide.BUY or str(fill.side).upper() == "BUY")

        if is_buy:
            # Cash outflow = (Q * P_buy) + Buy_costs = fill.net_value
            if fill.net_value > self.cash + 1e-6 and not self.allow_short:
                raise ValueError(
                    f"Cash Invariant Violation: Buy order requires ₹{fill.net_value:,.2f} "
                    f"but available cash is ₹{self.cash:,.2f}"
                )

            self.cash -= fill.net_value
            # Prevent negative floating-point epsilon
            if abs(self.cash) < 1e-7:
                self.cash = 0.0

            old_qty = pos.quantity
            new_qty = old_qty + fill.quantity

            if old_qty == 0:
                pos.avg_entry_price = fill.fill_price
                pos.total_buy_fees = fill.transaction_cost
            else:
                total_cost_basis = (old_qty * pos.avg_entry_price) + (fill.quantity * fill.fill_price)
                pos.avg_entry_price = total_cost_basis / new_qty
                pos.total_buy_fees += fill.transaction_cost

            pos.quantity = new_qty
            pos.current_price = fill.fill_price

        else:  # SELL
            if fill.quantity > pos.quantity and not self.allow_short:
                raise ValueError(
                    f"Execution Error: Cannot sell {fill.quantity} shares of {fill.symbol}. "
                    f"Currently held: {pos.quantity} shares."
                )

            # Cash inflow = (Q * P_sell) - Sell_costs = fill.net_value
            self.cash += fill.net_value

            # Completed trade net P&L:
            # Net_PnL = Q * P_sell - Q * P_buy - Buy_costs - Sell_costs
            cost_basis_sold = fill.quantity * pos.avg_entry_price
            buy_fee_portion = (fill.quantity / pos.quantity) * pos.total_buy_fees if pos.quantity > 0 else 0.0

            # fill.net_value = (Q * P_sell) - Sell_costs
            # trade_realized_pnl = fill.net_value - cost_basis_sold - buy_fee_portion
            trade_realized_pnl = fill.net_value - cost_basis_sold - buy_fee_portion
            self.realized_pnl += trade_realized_pnl

            pos.quantity -= fill.quantity
            pos.total_buy_fees = max(0.0, pos.total_buy_fees - buy_fee_portion)
            pos.current_price = fill.fill_price

            if pos.quantity == 0:
                pos.avg_entry_price = 0.0

        cash_after = self.cash
        return cash_before, cash_after, trade_realized_pnl

    def get_market_value(self) -> float:
        """Sum of market value of all open holdings."""
        return sum(pos.market_value for pos in self.positions.values() if pos.quantity > 0)

    def get_unrealized_pnl(self) -> float:
        """Sum of unrealized P&L across all open positions."""
        return sum(pos.unrealized_pnl for pos in self.positions.values() if pos.quantity > 0)

    def get_total_portfolio_value(self) -> float:
        """Mark-to-market portfolio equity = Available cash + Holdings market value."""
        return self.cash + self.get_market_value()

    def get_exposure(self) -> float:
        """Portfolio market exposure fraction."""
        port_val = self.get_total_portfolio_value()
        if port_val <= 0:
            return 0.0
        return self.get_market_value() / port_val

    def snapshot(self, timestamp: pd.Timestamp) -> Dict[str, Any]:
        """Generate a complete state snapshot at a valuation timestamp."""
        mkt_val = self.get_market_value()
        unrealized = self.get_unrealized_pnl()
        port_val = self.cash + mkt_val
        total_pnl = port_val - self.initial_capital
        total_shares = sum(pos.quantity for pos in self.positions.values() if pos.quantity > 0)
        open_count = sum(1 for pos in self.positions.values() if pos.quantity > 0)

        return {
            "Date": timestamp,
            "Cash": self.cash,
            "Holdings Value": mkt_val,
            "Portfolio Value": port_val,
            "Realized PnL": self.realized_pnl,
            "Unrealized PnL": unrealized,
            "Total PnL": total_pnl,
            "Shares": total_shares,
            "Open Positions": open_count,
            "Exposure": (mkt_val / port_val) if port_val > 0 else 0.0
        }
