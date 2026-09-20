"""
position_sizing.py

Position Sizing Module for QUANTLAB.
Computes order quantity based on available capital, position allocation fraction,
reference price, and cost estimates.
"""

import math
from typing import Tuple
from .config import BacktestConfig, InsufficientCashPolicy
from .order import OrderSide


class PositionSizer:
    """
    Computes trade quantity for BUY and SELL orders under capital and position constraints.
    """

    def __init__(self, config: BacktestConfig):
        self.config = config

    def calculate_buy_quantity(
        self,
        current_cash: float,
        price: float,
        position_size: float = None
    ) -> Tuple[int, str]:
        """
        Calculate target share quantity to BUY.

        Formula:
            Allocated Capital = Cash * Position Size Fraction
            Shares = floor(Allocated Capital / Estimated Cost Per Share)

        Returns
        -------
        Tuple[int, str]
            (shares_to_buy, note/reason)
        """
        if current_cash <= 0 or price <= 0:
            return 0, "Insufficient cash or invalid price"

        pos_fraction = position_size if position_size is not None else self.config.position_size
        pos_fraction = min(max(pos_fraction, 0.0), 1.0)

        allocated_capital = current_cash * pos_fraction

        # Estimate cost per share incorporating expected slippage and commission
        slippage_rate = self.config.slippage_bps / 10000.0
        commission_rate = self.config.commission_bps / 10000.0

        est_fill_price = price * (1.0 + slippage_rate)
        est_cost_per_share = est_fill_price * (1.0 + commission_rate)

        if est_cost_per_share <= 0:
            return 0, "Invalid cost estimation"

        shares = int(math.floor(allocated_capital / est_cost_per_share))

        if shares <= 0:
            if self.config.insufficient_cash_policy == InsufficientCashPolicy.REJECT:
                return 0, "Rejected: Position size too small for available capital"
            # Try to afford at least 1 share if total cash allows
            if current_cash >= est_cost_per_share:
                shares = 1
            else:
                return 0, "Insufficient cash to buy even 1 share"

        # Verify strict cash limit
        total_estimated_cost = shares * est_cost_per_share
        if total_estimated_cost > current_cash:
            if self.config.insufficient_cash_policy == InsufficientCashPolicy.REJECT:
                return 0, "Rejected: Required cash exceeds available cash"
            shares = int(math.floor(current_cash / est_cost_per_share))

        shares = max(0, shares)
        return shares, "OK"

    def calculate_sell_quantity(self, current_shares: int) -> Tuple[int, str]:
        """
        Calculate quantity to SELL (liquidate entire long position).

        Returns
        -------
        Tuple[int, str]
            (shares_to_sell, note)
        """
        if current_shares <= 0:
            return 0, "No position to sell"
        return max(0, current_shares), "OK"
