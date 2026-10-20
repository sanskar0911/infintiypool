"""
position_sizing.py

Position Sizing and Affordability Engine for QUANTLAB v1.1.
Enforces whole-share integer quantities, exact cash affordability constraints,
and fixed-fraction portfolio equity allocation:
- A = E * f (where E is current portfolio equity, f = 0.05 baseline)
- V_max = min(A, C / (1 + r_buy)) (where C is available cash, r_buy is proportional buy cost rate)
- Q = floor(V_max / P_buy)
- Actual buy debit = Q * P_buy + actual buy costs <= C
"""

import math
from typing import Tuple, Optional
from .config import BacktestConfig, InsufficientCashPolicy, SizingBasis
from .cost_model import TransactionCostModel, CostSchedule


class PositionSizer:
    """
    Computes integer whole-share quantities under strict affordability and exposure invariants.
    """

    def __init__(self, config: BacktestConfig, cost_model: Optional[TransactionCostModel] = None):
        self.config = config
        self.cost_model = cost_model or TransactionCostModel(
            schedule=CostSchedule(
                brokerage_rate=config.brokerage_bps / 10000.0,
                stt_rate_sell=config.stt_rate_sell,
                stamp_duty_rate_buy=config.stamp_duty_rate_buy,
                gst_rate=config.gst_rate,
                exchange_charge_rate=config.exchange_charge_rate,
                sebi_charge_rate=config.sebi_charge_rate
            )
        )

    def calculate_buy_quantity(
        self,
        current_cash: float,
        price: float,
        portfolio_equity: Optional[float] = None,
        position_size: Optional[float] = None
    ) -> Tuple[int, str]:
        """
        Calculate whole-share target quantity to BUY based on current portfolio equity.

        Parameters
        ----------
        current_cash : float
            Currently available uncommitted cash balance C.
        price : float
            Reference market price P_reference.
        portfolio_equity : float, optional
            Current total mark-to-market portfolio equity E.
            Defaults to current_cash if None.
        position_size : float, optional
            Target allocation fraction f (default: 0.05).

        Returns
        -------
        Tuple[int, str]
            (shares_to_buy, note/reason)
        """
        if current_cash <= 0:
            return 0, "Rejected: Available cash is zero or negative (no borrowing permitted)"
        if price <= 0:
            return 0, "Rejected: Invalid reference price"

        f = position_size if position_size is not None else self.config.position_size
        f = max(0.0, min(1.0, f))

        # Base sizing equity: current equity E or cash C (compatibility mode)
        sizing_basis = getattr(self.config, "sizing_basis", "equity")
        if sizing_basis == "cash" or portfolio_equity is None or portfolio_equity <= 0:
            base_capital = current_cash
        else:
            base_capital = portfolio_equity

        # Target allocation A = E * f
        target_allocation = base_capital * f

        # Estimated fill price with adverse slippage
        slippage_rate = self.config.slippage_bps / 10000.0
        p_buy = price * (1.0 + slippage_rate)

        # Baseline buy cost rate r_buy
        r_buy = self.cost_model.buy_cost_rate

        # Maximum affordable gross value: V_max = min(A, C / (1 + r_buy))
        max_affordable_by_cash = current_cash / (1.0 + r_buy)
        v_max = min(target_allocation, max_affordable_by_cash)

        if v_max <= 0 or p_buy <= 0:
            return 0, "Rejected: Target allocation or price invalid"

        # Integer whole-share quantity: Q = floor(V_max / P_buy)
        q = math.floor(v_max / p_buy)

        if q < 1:
            if self.config.insufficient_cash_policy == InsufficientCashPolicy.REJECT:
                return 0, f"Rejected: Quantity < 1 share (V_max ₹{v_max:,.2f} < P_buy ₹{p_buy:,.2f})"
            # If reduce quantity mode, check if we can afford exactly 1 share
            one_share_cost = p_buy * (1.0 + r_buy)
            if current_cash >= one_share_cost:
                q = 1
            else:
                return 0, "Rejected: Insufficient cash for 1 share"

        # Exact actual cost validation
        cost_info = self.cost_model.calculate_cost(quantity=q, fill_price=p_buy, side="BUY")
        actual_debit = cost_info["gross_value"] + cost_info["total_cost"]

        # Loop decrement if exact costs slightly exceed available cash due to rounding
        while actual_debit > current_cash and q > 0:
            q -= 1
            if q > 0:
                cost_info = self.cost_model.calculate_cost(quantity=q, fill_price=p_buy, side="BUY")
                actual_debit = cost_info["gross_value"] + cost_info["total_cost"]
            else:
                return 0, "Rejected: Total debit exceeds available cash"

        if q < 1:
            return 0, "Rejected: Net quantity less than 1 whole share"

        return q, "OK"

    def calculate_sell_quantity(self, current_shares: int) -> Tuple[int, str]:
        """
        Calculate quantity to SELL (liquidates open position).

        Returns
        -------
        Tuple[int, str]
            (shares_to_sell, note)
        """
        if current_shares <= 0:
            return 0, "No position to sell"
        return int(max(0, current_shares)), "OK"
