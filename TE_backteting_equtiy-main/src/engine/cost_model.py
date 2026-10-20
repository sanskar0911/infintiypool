"""
cost_model.py

Indian-Equity Transaction-Cost Engine for QUANTLAB Backtesting Engine v1.1.
Calculates independent buy-side and sell-side transaction costs, taxes, and statutory levies:
- Brokerage (default 0.03% / 3 bps per side)
- Securities Transaction Tax (STT): 0% buy-side, 0.10% (10 bps) sell-side
- Stamp Duty: 0.015% (1.5 bps) buy-side, 0% sell-side
- Goods and Services Tax (GST): 18% of (Brokerage + Exchange Charges)
- Exchange Transaction Charges: 0% baseline (configurable)
- SEBI Charges: 0% baseline (configurable)
- DP Charges: 0.0 baseline (configurable per sell)
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass
from .order import OrderSide


@dataclass
class CostSchedule:
    """Configurable statutory rates and fee schedules for Indian Cash Equities."""
    brokerage_rate: float = 0.0003          # 0.03% (3 bps) per side
    stt_rate_buy: float = 0.0               # 0% on buy
    stt_rate_sell: float = 0.0010           # 0.10% (10 bps) on sell
    stamp_duty_rate_buy: float = 0.00015    # 0.015% (1.5 bps) on buy
    stamp_duty_rate_sell: float = 0.0       # 0% on sell
    gst_rate: float = 0.18                  # 18% on (brokerage + exchange charges)
    exchange_charge_rate: float = 0.0       # 0% baseline (configurable, e.g. 0.0000325 for NSE)
    sebi_charge_rate: float = 0.0           # 0% baseline (configurable, e.g. 0.000001)
    dp_charges_per_sell: float = 0.0        # ₹0 baseline per sell execution
    min_brokerage: float = 0.0              # Optional minimum brokerage in INR
    max_brokerage_per_order: Optional[float] = None  # Optional cap (e.g. ₹20 flat discount broker)


class TransactionCostModel:
    """
    Indian-Equity Transaction-Cost Engine.
    Computes exact, transparent statutory deductions on buy and sell executions.
    """

    def __init__(
        self,
        commission_bps: Optional[float] = None,
        fixed_fee: float = 0.0,
        schedule: Optional[CostSchedule] = None,
        pure_commission_mode: bool = False
    ):
        if schedule is not None:
            self.schedule = schedule
        elif pure_commission_mode or (commission_bps is not None and schedule is None):
            # Pure commission test mode: only commission applied unless statutory rates configured
            self.schedule = CostSchedule(
                brokerage_rate=float(commission_bps) / 10000.0 if commission_bps is not None else 0.0,
                stt_rate_buy=0.0,
                stt_rate_sell=0.0,
                stamp_duty_rate_buy=0.0,
                stamp_duty_rate_sell=0.0,
                gst_rate=0.0,
                exchange_charge_rate=0.0,
                sebi_charge_rate=0.0
            )
        else:
            self.schedule = CostSchedule()

        self.fixed_fee = float(fixed_fee)

    @property
    def commission_bps(self) -> float:
        return self.schedule.brokerage_rate * 10000.0

    @property
    def rate(self) -> float:
        return self.schedule.brokerage_rate

    @property
    def buy_cost_rate(self) -> float:
        """
        Baseline buy-side proportional cost rate.
        r_buy = brokerage + stamp_duty + (brokerage + exchange_charge) * gst + exchange_charge + sebi_charge
        For baseline (0.0003 brokerage, 0.00015 stamp duty, 18% GST):
        0.0003 + 0.00015 + (0.0003 * 0.18) = 0.000504 (0.0504%).
        """
        b = self.schedule.brokerage_rate
        sd = self.schedule.stamp_duty_rate_buy
        ex = self.schedule.exchange_charge_rate
        sebi = self.schedule.sebi_charge_rate
        gst = (b + ex) * self.schedule.gst_rate
        return b + sd + ex + sebi + gst

    @property
    def sell_cost_rate(self) -> float:
        """
        Baseline sell-side proportional cost rate.
        r_sell = brokerage + stt + (brokerage + exchange_charge) * gst + exchange_charge + sebi_charge
        For baseline (0.0003 brokerage, 0.0010 STT, 18% GST):
        0.0003 + 0.0010 + (0.0003 * 0.18) = 0.001354 (0.1354%).
        """
        b = self.schedule.brokerage_rate
        stt = self.schedule.stt_rate_sell
        ex = self.schedule.exchange_charge_rate
        sebi = self.schedule.sebi_charge_rate
        gst = (b + ex) * self.schedule.gst_rate
        return b + stt + ex + sebi + gst

    def calculate_cost(
        self,
        quantity: int,
        fill_price: float,
        side: OrderSide = OrderSide.BUY
    ) -> Dict[str, float]:
        """
        Calculate auditable breakdown of transaction costs for an execution.

        Parameters
        ----------
        quantity : int
            Whole-share execution quantity.
        fill_price : float
            Execution price after slippage.
        side : OrderSide
            OrderSide.BUY or OrderSide.SELL.

        Returns
        -------
        Dict[str, float]
            Comprehensive breakdown of gross value, brokerage, STT, stamp duty,
            GST, exchange charges, and total cost.
        """
        gross_value = float(quantity) * float(fill_price)
        if gross_value <= 0:
            return {
                "gross_value": 0.0,
                "brokerage": 0.0,
                "stt": 0.0,
                "stamp_duty": 0.0,
                "gst": 0.0,
                "exchange_charges": 0.0,
                "sebi_charges": 0.0,
                "dp_charges": 0.0,
                "fixed_fee": 0.0,
                "commission": 0.0,
                "total_cost": 0.0
            }

        # 1. Brokerage
        brokerage = gross_value * self.schedule.brokerage_rate
        if self.schedule.min_brokerage > 0:
            brokerage = max(brokerage, self.schedule.min_brokerage)
        if self.schedule.max_brokerage_per_order is not None:
            brokerage = min(brokerage, self.schedule.max_brokerage_per_order)

        # 2. Exchange Charges & SEBI Turnover Charges
        exchange_charges = gross_value * self.schedule.exchange_charge_rate
        sebi_charges = gross_value * self.schedule.sebi_charge_rate

        # 3. GST (18% on Brokerage + Exchange Charges)
        gst = self.schedule.gst_rate * (brokerage + exchange_charges)

        is_buy = (side == OrderSide.BUY or str(side).upper() == "BUY")

        if is_buy:
            stt = gross_value * self.schedule.stt_rate_buy
            stamp_duty = gross_value * self.schedule.stamp_duty_rate_buy
            dp_charges = 0.0
        else:
            stt = gross_value * self.schedule.stt_rate_sell
            stamp_duty = gross_value * self.schedule.stamp_duty_rate_sell
            dp_charges = self.schedule.dp_charges_per_sell

        total_statutory = brokerage + stt + stamp_duty + gst + exchange_charges + sebi_charges + dp_charges + self.fixed_fee

        return {
            "gross_value": gross_value,
            "brokerage": brokerage,
            "stt": stt,
            "stamp_duty": stamp_duty,
            "gst": gst,
            "exchange_charges": exchange_charges,
            "sebi_charges": sebi_charges,
            "dp_charges": dp_charges,
            "fixed_fee": self.fixed_fee,
            "commission": brokerage,  # backward compatibility alias
            "total_cost": total_statutory
        }
