"""
cost_model.py

Transaction Cost Model for QUANTLAB Backtesting Engine.
Calculates dynamic brokerage, commissions, and transaction costs for executed trades.
"""

from typing import Dict, Any


class TransactionCostModel:
    """
    Configurable transaction cost model.

    Parameters
    ----------
    commission_bps : float
        Commission rate in basis points (5 bps = 0.05%).
    fixed_fee : float
        Optional fixed fee per execution in INR (default: 0.0).
    """

    def __init__(self, commission_bps: float = 5.0, fixed_fee: float = 0.0):
        self.commission_bps = float(commission_bps)
        self.rate = self.commission_bps / 10000.0
        self.fixed_fee = float(fixed_fee)

    def calculate_cost(self, quantity: int, fill_price: float) -> Dict[str, float]:
        """
        Calculate total transaction cost for a trade fill.

        Returns
        -------
        Dict[str, float]
            {
                'gross_value': gross_value,
                'commission': commission_cost,
                'fixed_fee': fixed_fee,
                'total_cost': total_transaction_cost
            }
        """
        gross_value = float(quantity) * float(fill_price)
        commission = gross_value * self.rate
        total_cost = commission + self.fixed_fee

        return {
            "gross_value": gross_value,
            "commission": commission,
            "fixed_fee": self.fixed_fee,
            "total_cost": total_cost
        }
