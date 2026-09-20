"""
portfolio.py

Portfolio State and Position Tracking Module for QUANTLAB Backtesting Engine.
Tracks mark-to-market valuations, open positions, average entry prices,
realized P&L, unrealized P&L, exposure, and portfolio value time-series.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple
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
        Number of open shares held.
    avg_entry_price : float
        Weighted average purchase price per share.
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
    Manages complete portfolio state: cash balance, open positions, realized/unrealized P&L.
    """

    def __init__(self, initial_capital: float = 100000.0, allow_short: bool = False):
        self.initial_capital = float(initial_capital)
        self.cash = float(initial_capital)
        self.allow_short = allow_short
        self.positions: Dict[str, PositionTracker] = {}
        self.realized_pnl: float = 0.0

    def get_position(self, symbol: str) -> PositionTracker:
        if symbol not in self.positions:
            self.positions[symbol] = PositionTracker(symbol=symbol)
        return self.positions[symbol]

    def update_market_prices(self, current_prices: Dict[str, float]):
        """Update mark-to-market prices for all open positions."""
        for symbol, price in current_prices.items():
            if symbol in self.positions:
                self.positions[symbol].current_price = float(price)

    def apply_fill(self, fill: Fill) -> Tuple[float, float]:
        """
        Process executed trade fill, updating cash, position quantity, average entry price,
        and realized P&L.

        Returns
        -------
        Tuple[float, float]
            (cash_before, cash_after)
        """
        cash_before = self.cash
        pos = self.get_position(fill.symbol)

        if fill.side == OrderSide.BUY:
            # Cash outflow = gross_value + transaction_cost = fill.net_value
            self.cash -= fill.net_value

            # Update position quantity & weighted avg entry price
            old_qty = pos.quantity
            new_qty = old_qty + fill.quantity

            if old_qty == 0:
                pos.avg_entry_price = fill.fill_price
                pos.total_buy_fees = fill.transaction_cost
            else:
                # Weighted average entry price
                total_cost_basis = (old_qty * pos.avg_entry_price) + (fill.quantity * fill.fill_price)
                pos.avg_entry_price = total_cost_basis / new_qty
                pos.total_buy_fees += fill.transaction_cost

            pos.quantity = new_qty
            pos.current_price = fill.fill_price

        elif fill.side == OrderSide.SELL:
            # Verify position constraint
            if fill.quantity > pos.quantity and not self.allow_short:
                raise ValueError(
                    f"Execution Error: Cannot sell {fill.quantity} shares of {fill.symbol}. "
                    f"Current position: {pos.quantity} shares."
                )

            # Cash inflow = gross_value - transaction_cost = fill.net_value
            self.cash += fill.net_value

            # Realized P&L calculation: Net Proceeds - Cost Basis
            cost_basis_sold = fill.quantity * pos.avg_entry_price
            # Proportionate buy fee for sold shares
            buy_fee_portion = (fill.quantity / pos.quantity) * pos.total_buy_fees if pos.quantity > 0 else 0.0
            
            trade_realized_pnl = fill.net_value - cost_basis_sold - buy_fee_portion
            self.realized_pnl += trade_realized_pnl

            # Update remaining position
            pos.quantity -= fill.quantity
            pos.total_buy_fees = max(0.0, pos.total_buy_fees - buy_fee_portion)
            pos.current_price = fill.fill_price

            if pos.quantity == 0:
                pos.avg_entry_price = 0.0

        cash_after = self.cash
        return cash_before, cash_after

    def get_market_value(self) -> float:
        return sum(pos.market_value for pos in self.positions.values() if pos.quantity > 0)

    def get_unrealized_pnl(self) -> float:
        return sum(pos.unrealized_pnl for pos in self.positions.values() if pos.quantity > 0)

    def get_total_portfolio_value(self) -> float:
        return self.cash + self.get_market_value()

    def get_exposure(self) -> float:
        port_val = self.get_total_portfolio_value()
        if port_val <= 0:
            return 0.0
        return self.get_market_value() / port_val

    def snapshot(self, timestamp: pd.Timestamp) -> Dict[str, Any]:
        """Generate a complete state snapshot dictionary at a given bar timestamp."""
        mkt_val = self.get_market_value()
        unrealized = self.get_unrealized_pnl()
        port_val = self.cash + mkt_val
        total_pnl = (port_val - self.initial_capital)
        total_shares = sum(pos.quantity for pos in self.positions.values())

        return {
            "Date": timestamp,
            "Cash": self.cash,
            "Holdings Value": mkt_val,
            "Shares": total_shares,
            "Portfolio Value": port_val,
            "Realized PnL": self.realized_pnl,
            "Unrealized PnL": unrealized,
            "Total PnL": total_pnl,
            "Exposure": self.get_exposure()
        }
