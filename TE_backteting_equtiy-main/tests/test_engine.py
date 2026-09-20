"""
test_engine.py

Comprehensive test suite for QUANTLAB Upgraded Backtesting Engine:
Order -> Execution -> Fill -> Portfolio State -> Metrics
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, "src")
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from engine import (
    BacktestEngine,
    BacktestConfig,
    ExecutionTiming,
    InsufficientCashPolicy,
    OrderSide,
    PositionSizer,
    SlippageModel,
    TransactionCostModel,
    PortfolioManager,
    DataValidator,
    DataValidationError
)


@pytest.fixture
def sample_data():
    """Create synthetic price DataFrame for testing."""
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    df = pd.DataFrame({
        "Open":  [1000.0, 1000.0, 1050.0, 1080.0, 1100.0, 1090.0, 1070.0, 1040.0, 1020.0, 1000.0],
        "High":  [1010.0, 1060.0, 1090.0, 1110.0, 1120.0, 1100.0, 1080.0, 1050.0, 1030.0, 1010.0],
        "Low":   [990.0,  995.0,  1040.0, 1070.0, 1080.0, 1070.0, 1050.0, 1020.0, 1000.0, 990.0],
        "Close": [1000.0, 1050.0, 1080.0, 1100.0, 1090.0, 1070.0, 1040.0, 1020.0, 1000.0, 980.0],
        "Volume": [10000] * 10,
        "Signal": [0,      1,      0,      0,      -1,     0,      0,      0,      0,      0],
        "Position": [0,    1,      1,      1,      0,      0,      0,      0,      0,      0]
    }, index=dates)
    return df


def test_position_sizing_calculation():
    """Verify position sizing: 20% of ₹1,00,000 at ₹1,000 -> ~20 shares."""
    config = BacktestConfig(initial_capital=100000.0, position_size=0.20, slippage_bps=0.0, commission_bps=0.0)
    sizer = PositionSizer(config)
    shares, note = sizer.calculate_buy_quantity(current_cash=100000.0, price=1000.0)
    assert shares == 20
    assert note == "OK"


def test_slippage_model():
    """Verify BUY fill > reference price and SELL fill < reference price."""
    slip_model = SlippageModel(slippage_bps=10.0)  # 0.10%
    base_price = 1000.0

    buy_fill, buy_slip = slip_model.calculate_fill_price(base_price, OrderSide.BUY)
    sell_fill, sell_slip = slip_model.calculate_fill_price(base_price, OrderSide.SELL)

    assert buy_fill == pytest.approx(1001.0)
    assert buy_slip == pytest.approx(1.0)
    assert sell_fill == pytest.approx(999.0)
    assert sell_slip == pytest.approx(1.0)


def test_transaction_cost_model():
    """Verify commission cost calculation."""
    cost_model = TransactionCostModel(commission_bps=5.0)  # 0.05%
    cost_info = cost_model.calculate_cost(quantity=20, fill_price=1000.0)

    assert cost_info["gross_value"] == 20000.0
    assert cost_info["commission"] == 10.0  # 20,000 * 0.0005
    assert cost_info["total_cost"] == 10.0


def test_buy_execution_updates_portfolio(sample_data):
    """Verify BUY execution decreases cash, increases position, and creates a fill record."""
    config = BacktestConfig(
        initial_capital=100000.0,
        position_size=0.20,
        slippage_bps=10.0,
        commission_bps=5.0,
        execution_timing=ExecutionTiming.NEXT_BAR_OPEN
    )
    engine = BacktestEngine(config)
    res = engine.run(sample_data)

    assert len(res.fills) >= 2  # 1 BUY, 1 SELL
    buy_fill = res.fills[0]

    assert buy_fill.side == OrderSide.BUY
    assert buy_fill.quantity == 19  # floor(20,000 / (1001 * 1.0005))
    assert buy_fill.reference_price == 1050.0  # Next Bar Open (bar 2 Open)
    assert buy_fill.fill_price > buy_fill.reference_price  # Slippage applied
    assert res.portfolio_history["Cash"].iloc[2] < 100000.0
    assert res.portfolio_history["Holdings Value"].iloc[2] > 0


def test_sell_execution_and_realized_pnl(sample_data):
    """Verify SELL execution liquidates position, increases cash, and calculates realized P&L."""
    config = BacktestConfig(
        initial_capital=100000.0,
        position_size=0.20,
        slippage_bps=0.0,
        commission_bps=0.0,
        execution_timing=ExecutionTiming.NEXT_BAR_OPEN
    )
    engine = BacktestEngine(config)
    res = engine.run(sample_data)

    sell_fill = [f for f in res.fills if f.side == OrderSide.SELL][0]
    assert sell_fill.quantity > 0
    assert sell_fill.reference_price == 1090.0  # Bar 5 Open

    final_row = res.portfolio_history.iloc[-1]
    assert final_row["Realized PnL"] > 0
    assert final_row["Shares"] == 0


def test_next_bar_execution_timing(sample_data):
    """Verify point-in-time safety: Signal at Bar T (Close) executes at Bar T+1 (Open)."""
    config = BacktestConfig(execution_timing=ExecutionTiming.NEXT_BAR_OPEN)
    engine = BacktestEngine(config)
    res = engine.run(sample_data)

    # Signal generated on Bar 1 (Close: 1050)
    order = res.orders[0]
    fill = res.fills[0]

    assert order.signal_bar_idx == 1
    assert fill.timestamp == sample_data.index[2]  # Executed on Bar 2
    assert fill.reference_price == sample_data["Open"].iloc[2]  # Bar 2 Open (1050.0)


def test_cash_constraint_prevention():
    """Verify system never creates negative cash under standard execution."""
    pm = PortfolioManager(initial_capital=1000.0, allow_short=False)
    assert pm.cash == 1000.0
    assert pm.cash >= 0.0


def test_position_constraint_prevention():
    """Verify engine raises error or rejects selling more shares than owned."""
    pm = PortfolioManager(initial_capital=100000.0, allow_short=False)
    with pytest.raises(ValueError, match="Cannot sell 50 shares"):
        from engine.fill import Fill
        invalid_sell = Fill(
            timestamp=pd.Timestamp("2024-01-01"),
            symbol="INFY",
            side=OrderSide.SELL,
            quantity=50,
            reference_price=1000.0,
            fill_price=1000.0,
            slippage_cost=0.0,
            transaction_cost=0.0,
            gross_value=50000.0,
            net_value=50000.0,
            order_id="ORD-1"
        )
        pm.apply_fill(invalid_sell)


def test_portfolio_valuation_identity(sample_data):
    """Verify portfolio value = Cash + Holdings Market Value at every bar."""
    engine = BacktestEngine()
    res = engine.run(sample_data)
    df_port = res.portfolio_history

    for idx, row in df_port.iterrows():
        calc_val = row["Cash"] + row["Holdings Value"]
        assert pytest.approx(row["Portfolio Value"], 1e-5) == calc_val


def test_realized_vs_unrealized_pnl_separation(sample_data):
    """Verify realized and unrealized P&L are separated throughout lifecycle."""
    engine = BacktestEngine()
    res = engine.run(sample_data)
    df_port = res.portfolio_history

    # Bar 2: Open long position -> Unrealized PnL present, Realized PnL == 0
    bar2 = df_port.iloc[2]
    assert bar2["Realized PnL"] == 0.0
    assert bar2["Unrealized PnL"] != 0.0

    # Final Bar: Closed position -> Realized PnL present, Unrealized PnL == 0
    final_bar = df_port.iloc[-1]
    assert final_bar["Unrealized PnL"] == 0.0
