"""
signal_ingestion.py

Signal File Ingestion & Execution Engine for QUANTLAB v1.1.
Enables running backtests directly from CSV (or Excel) signal files:
- Automated column header detection and standardization
- Active trade handling ("Active(on Date)")
- Full event-driven simulation under QUANTLAB v1.1 rules:
  * Initial capital: ₹10,000,000 (₹1 Crore)
  * Sizing: 5% of current portfolio equity (whole shares, integer Q)
  * Adverse slippage: 0.05% per side
  * Indian equity transaction costs: Brokerage 0.03%, STT 0.10% sell, Stamp Duty 0.015% buy, GST 18%
  * Order sequencing: Sell-first, alphabetical buys, duplicate prevention
  * Auditable trade ledger with cumulative P&L and cost breakdown
  * Daily mark-to-market equity curve
  * Terminal liquidation at final date
  * Authoritative performance metrics
"""

import os
import re
import io
import sys
import math
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import BacktestConfig, get_default_baseline_config
from .cost_model import TransactionCostModel, CostSchedule
from .execution import SlippageModel
from .order import OrderSide
from .fill import Fill
from .trade_log import TradeLogger
from .portfolio import PortfolioManager

# Metrics library
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from metrics import calculate_metrics

COLUMN_KEYWORDS = {
    "stock": ["stock", "symbol", "ticker", "instrument", "name", "stock name", "company", "security"],
    "entry_date": ["entry date", "entrydate", "buy date", "start date", "trade date", "buy_date"],
    "entry_price": ["entry price", "entryprice", "buy price", "purchase price"],
    "exit_date": ["exit date", "exitdate", "sell date", "end date", "close date", "sell_date"],
    "exit_price": ["close price", "closeprice", "exit price", "exitprice", "sell price", "closing price"],
    "provided_return": ["returns delivered (%)", "returns delivered", "return", "returns", "p&l %", "return %", "ret %"]
}


def load_and_standardize_signals(file_input: Any) -> pd.DataFrame:
    """
    Load signal CSV or Excel file (path, file-like, or DataFrame) and standardize columns.
    """
    if isinstance(file_input, pd.DataFrame):
        raw_df = file_input.copy()
    else:
        file_path = str(file_input)
        p = Path(file_path)
        if not p.exists():
            # Check in validation/, src/validation_signal/, or root
            alt_candidates = [
                Path("validation") / p.name,
                Path("src/validation_signal") / p.name,
                Path("..") / "validation" / p.name,
                Path(__file__).resolve().parent.parent / "validation_signal" / p.name,
                Path(__file__).resolve().parent.parent.parent / "validation" / p.name
            ]
            for c in alt_candidates:
                if c.exists():
                    p = c
                    break

        if p.suffix.lower() == ".csv" or ".csv" in p.name.lower():
            try:
                raw_df = pd.read_csv(p, header=None)
            except Exception:
                raw_df = pd.read_csv(p, header=None, encoding="latin1")
        else:
            try:
                raw_df = pd.read_excel(p, header=None)
            except Exception:
                raw_df = pd.read_csv(p, header=None)

    # Detect header row
    best_row_idx = -1
    col_mapping: Dict[int, str] = {}
    max_matches = 0

    for r_idx in range(min(15, len(raw_df))):
        temp_map: Dict[int, str] = {}
        for c_idx, cell_val in enumerate(raw_df.iloc[r_idx]):
            if pd.isna(cell_val):
                continue
            norm = str(cell_val).strip().lower()
            for std_col, syns in COLUMN_KEYWORDS.items():
                if any(syn in norm for syn in syns) and std_col not in temp_map.values():
                    temp_map[c_idx] = std_col
                    break
        if len(temp_map) > max_matches:
            max_matches = len(temp_map)
            best_row_idx = r_idx
            col_mapping = temp_map

    if best_row_idx == -1 or max_matches < 3:
        # Fallback to direct column names if already standard
        if isinstance(raw_df.columns[0], str):
            clean_cols = {}
            for c in raw_df.columns:
                c_norm = c.strip().lower()
                for std_col, syns in COLUMN_KEYWORDS.items():
                    if any(syn in c_norm for syn in syns):
                        clean_cols[c] = std_col
                        break
            if len(clean_cols) >= 3:
                df = raw_df.rename(columns=clean_cols).copy()
                data_rows = df
            else:
                raise ValueError("Could not automatically map headers in signal file.")
        else:
            raise ValueError("Could not identify header row in signal file.")
    else:
        data_rows = raw_df.iloc[best_row_idx + 1:].copy()
        cleaned_data = {}
        for c_idx, col_name in col_mapping.items():
            cleaned_data[col_name] = data_rows[c_idx]
        df = pd.DataFrame(cleaned_data).reset_index(drop=True)

    # Clean fields
    df["stock"] = df["stock"].astype(str).str.strip()
    df = df[df["stock"].ne("") & df["stock"].ne("nan")].copy()

    # Handle active trades
    def clean_date_str(val: Any) -> str:
        if pd.isna(val):
            return ""
        s = str(val).strip()
        if "active" in s.lower() and "(" in s:
            m = re.search(r"\((?:on\s+)?([^)]+)\)", s)
            if m:
                s = m.group(1).strip()
        return s

    raw_exit_dates = [clean_date_str(x) for x in df["exit_date"]]
    df["entry_date"] = pd.to_datetime(df["entry_date"], errors="coerce")
    df["exit_date"] = pd.to_datetime(raw_exit_dates, errors="coerce")
    df["exit_date"] = df["exit_date"].fillna(df["entry_date"])

    df["entry_price"] = pd.to_numeric(df["entry_price"], errors="coerce")
    df["exit_price"] = pd.to_numeric(df["exit_price"], errors="coerce")
    df["exit_price"] = df["exit_price"].fillna(df["entry_price"])

    if "provided_return" in df.columns:
        df["provided_return"] = pd.to_numeric(df["provided_return"], errors="coerce")
    else:
        df["provided_return"] = np.nan

    # Keep valid positive entry prices and dates
    valid_mask = (df["entry_price"] > 0) & (pd.notna(df["entry_date"]))
    df = df[valid_mask].copy()

    # Drop exact duplicates
    df.drop_duplicates(subset=["stock", "entry_date", "entry_price", "exit_date"], inplace=True)
    df.sort_values(by=["entry_date", "stock"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


def run_signal_file_backtest(
    file_input: Any,
    config: Optional[BacktestConfig] = None,
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Execute full QUANTLAB v1.1 backtesting engine on a signal CSV/Excel file.

    Parameters
    ----------
    file_input : str, Path, or pd.DataFrame
        Path to the signal CSV file or loaded DataFrame.
    config : BacktestConfig, optional
        Configuration settings (defaults to baseline config).
    output_dir : str, optional
        Directory to export output CSVs and chart images.

    Returns
    -------
    Dict[str, Any]
        Dictionary with:
        - "trade_log_df": Auditable ledger DataFrame
        - "portfolio_df": Daily mark-to-market equity curve DataFrame
        - "metrics": Comprehensive KPI dictionary
        - "raw_signals_df": Standardized signal inputs
        - "charts": Dict of chart image paths
    """
    cfg = config or get_default_baseline_config()
    signals_df = load_and_standardize_signals(file_input)

    cost_model = TransactionCostModel(
        schedule=CostSchedule(
            brokerage_rate=cfg.brokerage_bps / 10000.0,
            stt_rate_sell=cfg.stt_rate_sell,
            stamp_duty_rate_buy=cfg.stamp_duty_rate_buy,
            gst_rate=cfg.gst_rate,
            exchange_charge_rate=cfg.exchange_charge_rate,
            sebi_charge_rate=cfg.sebi_charge_rate
        )
    )
    slippage_model = SlippageModel(cfg.slippage_bps)

    portfolio_mgr = PortfolioManager(
        initial_capital=cfg.initial_capital,
        allow_short=cfg.allow_short_selling
    )
    trade_logger = TradeLogger()

    # Build discrete event schedule
    # Events: (timestamp, 'SELL', trade_idx) and (timestamp, 'BUY', trade_idx)
    events = []
    for idx, row in signals_df.iterrows():
        events.append((row["entry_date"], "BUY", idx, row["stock"], row["entry_price"]))
        events.append((row["exit_date"], "SELL", idx, row["stock"], row["exit_price"]))

    # Sort events:
    # 1. By Date ascending
    # 2. On same date, SELL before BUY
    # 3. On same date and same side, BUYS alphabetically by symbol
    def event_sort_key(ev):
        date_ts, side, t_idx, stock, px = ev
        side_order = 0 if side == "SELL" else 1
        return (date_ts, side_order, stock, t_idx)

    events.sort(key=event_sort_key)

    # Active held positions map: stock -> {qty, entry_px, trade_idx, ...}
    open_trades: Dict[str, Dict[str, Any]] = {}
    all_dates = sorted(list(set(ev[0] for ev in events)))
    daily_history = []

    # Map events by date
    events_by_date: Dict[pd.Timestamp, List[Any]] = {}
    for ev in events:
        d = ev[0]
        if d not in events_by_date:
            events_by_date[d] = []
        events_by_date[d].append(ev)

    for current_date in all_dates:
        day_events = events_by_date.get(current_date, [])
        sells_today = [e for e in day_events if e[1] == "SELL"]
        buys_today = [e for e in day_events if e[1] == "BUY"]

        # Sort buys alphabetically by symbol
        buys_today.sort(key=lambda x: x[3])

        # ----------------------------------------------------------------------
        # 1. Process SELLS First
        # ----------------------------------------------------------------------
        for _, side, trade_idx, stock, ref_price in sells_today:
            if stock not in open_trades:
                continue  # Trade was not entered or already closed

            trade_info = open_trades[stock]
            sell_qty = trade_info["quantity"]
            if sell_qty <= 0:
                continue

            fill_price, slip_per_share = slippage_model.calculate_fill_price(ref_price, OrderSide.SELL)
            cost_info = cost_model.calculate_cost(sell_qty, fill_price, side=OrderSide.SELL)
            total_slippage = slip_per_share * sell_qty
            net_proceeds = cost_info["gross_value"] - cost_info["total_cost"]

            fill = Fill(
                timestamp=current_date,
                symbol=stock,
                side=OrderSide.SELL,
                quantity=sell_qty,
                reference_price=ref_price,
                fill_price=fill_price,
                slippage_cost=total_slippage,
                brokerage=cost_info["brokerage"],
                stt=cost_info["stt"],
                stamp_duty=cost_info["stamp_duty"],
                gst=cost_info["gst"],
                exchange_charges=cost_info["exchange_charges"],
                transaction_cost=cost_info["total_cost"],
                gross_value=cost_info["gross_value"],
                net_value=net_proceeds,
                order_id=f"ORD-S-{trade_idx}",
                reason="Signal Exit"
            )

            pos = portfolio_mgr.get_position(stock)
            pos_before = pos.quantity
            cash_before, cash_after, realized_pnl = portfolio_mgr.apply_fill(fill)

            trade_logger.log_trade(
                fill=fill,
                cash_before=cash_before,
                cash_after=cash_after,
                portfolio_value=portfolio_mgr.get_total_portfolio_value(),
                position_before=pos_before,
                position_after=0,
                realized_pnl=realized_pnl,
                unrealized_pnl=portfolio_mgr.get_unrealized_pnl(),
                order_status="FILLED",
                reason="Signal Exit"
            )

            del open_trades[stock]

        # ----------------------------------------------------------------------
        # 2. Process BUYS Second (alphabetically)
        # ----------------------------------------------------------------------
        for _, side, trade_idx, stock, ref_price in buys_today:
            # Duplicate prevention: ignore BUY if symbol is already held
            if portfolio_mgr.has_open_position(stock):
                trade_logger.log_rejection(
                    timestamp=current_date,
                    symbol=stock,
                    side=OrderSide.BUY,
                    quantity=0,
                    price=ref_price,
                    cash_available=portfolio_mgr.cash,
                    portfolio_value=portfolio_mgr.get_total_portfolio_value(),
                    rejection_reason="Duplicate Position Prevention: Symbol already held",
                    order_id=f"ORD-B-{trade_idx}"
                )
                continue

            fill_price, slip_per_share = slippage_model.calculate_fill_price(ref_price, OrderSide.BUY)

            # Sizing: Target allocation 5% of current equity E
            port_equity = portfolio_mgr.get_total_portfolio_value()
            sizing_basis = getattr(cfg, "sizing_basis", "equity")
            base_capital = port_equity if sizing_basis == "equity" else portfolio_mgr.cash
            target_alloc = base_capital * cfg.position_size

            # Proportional buy cost rate
            r_buy = cost_model.buy_cost_rate
            max_affordable = portfolio_mgr.cash / (1.0 + r_buy)
            v_max = min(target_alloc, max_affordable)

            if v_max <= 0 or fill_price <= 0:
                continue

            buy_qty = math.floor(v_max / fill_price)
            if buy_qty < 1:
                trade_logger.log_rejection(
                    timestamp=current_date,
                    symbol=stock,
                    side=OrderSide.BUY,
                    quantity=0,
                    price=ref_price,
                    cash_available=portfolio_mgr.cash,
                    portfolio_value=port_equity,
                    rejection_reason=f"Insufficient Cash: Target alloc ₹{target_alloc:,.2f} affords < 1 share @ ₹{fill_price:,.2f}",
                    order_id=f"ORD-B-{trade_idx}"
                )
                continue

            cost_info = cost_model.calculate_cost(buy_qty, fill_price, side=OrderSide.BUY)
            total_debit = cost_info["gross_value"] + cost_info["total_cost"]

            # Guard against tiny rounding differences
            while total_debit > portfolio_mgr.cash and buy_qty > 0:
                buy_qty -= 1
                if buy_qty > 0:
                    cost_info = cost_model.calculate_cost(buy_qty, fill_price, side=OrderSide.BUY)
                    total_debit = cost_info["gross_value"] + cost_info["total_cost"]

            if buy_qty < 1:
                continue

            total_slippage = slip_per_share * buy_qty
            fill = Fill(
                timestamp=current_date,
                symbol=stock,
                side=OrderSide.BUY,
                quantity=buy_qty,
                reference_price=ref_price,
                fill_price=fill_price,
                slippage_cost=total_slippage,
                brokerage=cost_info["brokerage"],
                stt=cost_info["stt"],
                stamp_duty=cost_info["stamp_duty"],
                gst=cost_info["gst"],
                exchange_charges=cost_info["exchange_charges"],
                transaction_cost=cost_info["total_cost"],
                gross_value=cost_info["gross_value"],
                net_value=total_debit,
                order_id=f"ORD-B-{trade_idx}",
                reason="Signal Entry"
            )

            pos = portfolio_mgr.get_position(stock)
            pos_before = pos.quantity
            cash_before, cash_after, _ = portfolio_mgr.apply_fill(fill)

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
                reason="Signal Entry"
            )

            open_trades[stock] = {
                "quantity": buy_qty,
                "entry_price": fill_price,
                "entry_date": current_date,
                "trade_idx": trade_idx
            }

        # ----------------------------------------------------------------------
        # 3. Daily Mark-to-Market Snapshot
        # ----------------------------------------------------------------------
        snap = portfolio_mgr.snapshot(current_date)
        daily_history.append(snap)

    # --------------------------------------------------------------------------
    # 4. Terminal Liquidation of Remaining Open Positions
    # --------------------------------------------------------------------------
    if cfg.terminal_force_close and open_trades and all_dates:
        final_date = all_dates[-1]
        for stock in list(open_trades.keys()):
            pos = portfolio_mgr.get_position(stock)
            if pos.quantity <= 0:
                continue

            final_ref_price = pos.current_price if pos.current_price > 0 else pos.avg_entry_price
            fill_price, slip_per_share = slippage_model.calculate_fill_price(final_ref_price, OrderSide.SELL)
            cost_info = cost_model.calculate_cost(pos.quantity, fill_price, side=OrderSide.SELL)
            total_slippage = slip_per_share * pos.quantity
            net_proceeds = cost_info["gross_value"] - cost_info["total_cost"]

            term_fill = Fill(
                timestamp=final_date,
                symbol=stock,
                side=OrderSide.SELL,
                quantity=pos.quantity,
                reference_price=final_ref_price,
                fill_price=fill_price,
                slippage_cost=total_slippage,
                brokerage=cost_info["brokerage"],
                stt=cost_info["stt"],
                stamp_duty=cost_info["stamp_duty"],
                gst=cost_info["gst"],
                exchange_charges=cost_info["exchange_charges"],
                transaction_cost=cost_info["total_cost"],
                gross_value=cost_info["gross_value"],
                net_value=net_proceeds,
                order_id="ORD-S-TERMINAL",
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

        if daily_history:
            daily_history[-1] = portfolio_mgr.snapshot(final_date)

    # Build Output DataFrames
    portfolio_df = pd.DataFrame(daily_history)
    if not portfolio_df.empty:
        portfolio_df.set_index("Date", inplace=True)
        portfolio_df["Portfolio Return"] = portfolio_df["Portfolio Value"].pct_change().fillna(0.0)

    trade_log_df = trade_logger.to_dataframe()

    # Calculate authoritative Metrics
    metrics = calculate_metrics(
        portfolio_df=portfolio_df,
        trade_log_df=trade_log_df,
        initial_capital=cfg.initial_capital
    )

    # Export CSVs if output_dir specified
    out_dir = Path(output_dir) if output_dir else Path(__file__).resolve().parent.parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    trade_log_path = out_dir / "trade_ledger.csv"
    trade_log_df.to_csv(trade_log_path)

    equity_curve_path = out_dir / "daily_equity_curve.csv"
    portfolio_df.to_csv(equity_curve_path)

    metrics_df = pd.DataFrame([
        {"Metric Name": k, "Calculated Value": v} for k, v in metrics.items()
    ])
    metrics_path = out_dir / "engine_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    # Generate high-resolution visual charts
    chart_paths = {}
    try:
        # Chart 1: Equity Curve
        fig, ax = plt.subplots(figsize=(10, 4.5), dpi=150)
        ax.plot(portfolio_df.index, portfolio_df["Portfolio Value"] / 10000000.0, color="#10b981", lw=2, label="Portfolio Equity (₹ Cr)")
        ax.axhline(cfg.initial_capital / 10000000.0, color="#64748b", ls="--", label="Initial Capital (₹1.00 Cr)")
        ax.set_title("QUANTLAB v1.1 - Equity Growth Curve", color="white", fontsize=12, fontweight="bold")
        ax.set_ylabel("Equity (₹ Crore)", color="#94a3b8")
        ax.set_facecolor("#111827")
        fig.patch.set_facecolor("#0b0f19")
        ax.tick_params(colors="#94a3b8")
        for spine in ax.spines.values():
            spine.set_color("#334155")
        ax.legend(facecolor="#1e293b", edgecolor="#475569", labelcolor="white")
        plt.tight_layout()
        eq_img_path = out_dir / "equity_growth_curve.png"
        fig.savefig(eq_img_path)
        plt.close(fig)
        chart_paths["equity"] = str(eq_img_path)

        # Chart 2: Underwater Drawdown
        running_max = portfolio_df["Portfolio Value"].cummax()
        dd_pct = ((portfolio_df["Portfolio Value"] - running_max) / running_max) * 100.0

        fig, ax = plt.subplots(figsize=(10, 3.5), dpi=150)
        ax.fill_between(portfolio_df.index, dd_pct, 0, color="#ef4444", alpha=0.35)
        ax.plot(portfolio_df.index, dd_pct, color="#f87171", lw=1.5, label="Drawdown %")
        ax.set_title("QUANTLAB v1.1 - Underwater Drawdown Curve", color="white", fontsize=12, fontweight="bold")
        ax.set_ylabel("Drawdown (%)", color="#94a3b8")
        ax.set_facecolor("#111827")
        fig.patch.set_facecolor("#0b0f19")
        ax.tick_params(colors="#94a3b8")
        for spine in ax.spines.values():
            spine.set_color("#334155")
        ax.legend(facecolor="#1e293b", edgecolor="#475569", labelcolor="white")
        plt.tight_layout()
        dd_img_path = out_dir / "underwater_drawdown.png"
        fig.savefig(dd_img_path)
        plt.close(fig)
        chart_paths["drawdown"] = str(dd_img_path)
    except Exception as e:
        print(f"Chart generation note: {e}")

    return {
        "trade_log_df": trade_log_df,
        "portfolio_df": portfolio_df,
        "metrics": metrics,
        "raw_signals_df": signals_df,
        "charts": chart_paths
    }
