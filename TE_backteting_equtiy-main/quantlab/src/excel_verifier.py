"""Manual Excel Verification Workbook Generator for QuantLab Backtesting Engine."""

from pathlib import Path
from typing import List, Tuple
import openpyxl
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
import pandas as pd

from .config import BacktestConfig
from .metrics import BacktestMetrics


def generate_manual_verification_workbook(
    raw_df: pd.DataFrame,
    trades: pd.DataFrame,
    metrics: BacktestMetrics,
    config: BacktestConfig,
    output_path: str
) -> None:
    """
    Creates output/manual_backtest_verification.xlsx with sheets:
    - RAW_INPUT: pasted signal data
    - TRADE_CALCULATION: live formulas calculating Qty, Alloc, PnL, Cumulative, DD, etc.
    - METRICS: live Excel formulas evaluating all performance metrics with PASS/FAIL check.
    """
    wb = openpyxl.Workbook()
    # Remove default sheet
    wb.remove(wb.active)

    # Styling helpers
    header_fill = PatternFill(start_color="1E293B", end_color="1E293B", fill_type="solid")
    header_font = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
    title_font = Font(name="Calibri", size=14, bold=True, color="1E293B")
    border_thin = Border(
        left=Side(style="thin", color="D1D5DB"),
        right=Side(style="thin", color="D1D5DB"),
        top=Side(style="thin", color="D1D5DB"),
        bottom=Side(style="thin", color="D1D5DB")
    )
    align_center = Alignment(horizontal="center", vertical="center")
    align_left = Alignment(horizontal="left", vertical="center")
    align_right = Alignment(horizontal="right", vertical="center")

    n = len(trades)
    end_row = n + 2  # data starts on row 3

    # =========================================================================
    # 1. RAW_INPUT Sheet
    # =========================================================================
    ws_raw = wb.create_sheet(title="RAW_INPUT")
    ws_raw.views.sheetView[0].showGridLines = True

    # Title
    ws_raw.merge_cells("A1:F1")
    ws_raw["A1"] = "RAW TRADE SIGNALS IMPORT"
    ws_raw["A1"].font = title_font

    headers_raw = ["Stock", "Entry Date", "Entry Price", "Exit Date", "Close Price", "Provided Return %"]
    for col_idx, h in enumerate(headers_raw, 1):
        cell = ws_raw.cell(row=2, column=col_idx, value=h)
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = align_center
        cell.border = border_thin

    for i, row in trades.iterrows():
        r = i + 3
        ws_raw.cell(row=r, column=1, value=str(row["stock"])).border = border_thin
        ws_raw.cell(row=r, column=2, value=row["entry_date"].strftime("%Y-%m-%d")).border = border_thin
        ws_raw.cell(row=r, column=3, value=float(row["entry_price"])).border = border_thin
        ws_raw.cell(row=r, column=4, value=row["exit_date"].strftime("%Y-%m-%d")).border = border_thin
        ws_raw.cell(row=r, column=5, value=float(row["exit_price"])).border = border_thin
        prov_val = float(row["provided_return_pct"]) if pd.notna(row["provided_return_pct"]) else 0.0
        ws_raw.cell(row=r, column=6, value=prov_val).border = border_thin

    for col in ws_raw.columns:
        max_len = max(len(str(cell.value or "")) for cell in col)
        col_letter = get_column_letter(col[0].column)
        ws_raw.column_dimensions[col_letter].width = max(max_len + 3, 14)

    # =========================================================================
    # 2. TRADE_CALCULATION Sheet
    # =========================================================================
    ws_calc = wb.create_sheet(title="TRADE_CALCULATION")
    ws_calc.views.sheetView[0].showGridLines = True

    ws_calc.merge_cells("A1:V1")
    ws_calc["A1"] = "INDEPENDENT TRADE-BY-TRADE AUDIT & FORMULA CALCULATIONS"
    ws_calc["A1"].font = title_font

    calc_headers = [
        "Trade ID", "Stock", "Entry Date", "Exit Date", "Duration",
        "Entry Price", "Exit Price", "Quantity", "Allocated Capital",
        "Calculated P&L", "Calculated Return %", "Provided Return %", "Return Difference %",
        "Winning Trade", "Losing Trade", "Cumulative P&L", "Cumulative Return %",
        "Equity", "Running Peak", "Drawdown", "Drawdown %", "Downside Return Sq"
    ]

    for col_idx, h in enumerate(calc_headers, 1):
        cell = ws_calc.cell(row=2, column=col_idx, value=h)
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = align_center
        cell.border = border_thin

    for i in range(n):
        r = i + 3
        # A: Trade ID
        ws_calc.cell(row=r, column=1, value=f'=TEXT({r-2}, """T""0000")').border = border_thin
        # B: Stock
        ws_calc.cell(row=r, column=2, value=f"=RAW_INPUT!A{r}").border = border_thin
        # C: Entry Date
        ws_calc.cell(row=r, column=3, value=f"=RAW_INPUT!B{r}").border = border_thin
        # D: Exit Date
        ws_calc.cell(row=r, column=4, value=f"=RAW_INPUT!D{r}").border = border_thin
        # E: Duration
        ws_calc.cell(row=r, column=5, value=f"=D{r}-C{r}").border = border_thin
        # F: Entry Price
        ws_calc.cell(row=r, column=6, value=f"=RAW_INPUT!C{r}").border = border_thin
        # G: Exit Price
        ws_calc.cell(row=r, column=7, value=f"=RAW_INPUT!E{r}").border = border_thin
        # H: Quantity = Allocated Capital / Entry Price
        ws_calc.cell(row=r, column=8, value=f"=I{r}/F{r}").border = border_thin
        # I: Allocated Capital = Initial Capital * 0.02
        ws_calc.cell(row=r, column=9, value=f"=METRICS!$C$14 * {config.capital_allocation_per_trade}").border = border_thin
        # J: Calculated P&L = (Exit Price - Entry Price) * Quantity
        ws_calc.cell(row=r, column=10, value=f"=(G{r}-F{r})*H{r}").border = border_thin
        # K: Calculated Return % = (Exit Price - Entry Price) / Entry Price * 100
        ws_calc.cell(row=r, column=11, value=f"=(G{r}-F{r})/F{r}*100").border = border_thin
        # L: Provided Return %
        ws_calc.cell(row=r, column=12, value=f"=RAW_INPUT!F{r}").border = border_thin
        # M: Return Difference % = Calculated Return % - Provided Return %
        ws_calc.cell(row=r, column=13, value=f"=K{r}-L{r}").border = border_thin
        # N: Winning Trade = IF(J > 0, 1, 0)
        ws_calc.cell(row=r, column=14, value=f"=IF(J{r}>0, 1, 0)").border = border_thin
        # O: Losing Trade = IF(J < 0, 1, 0)
        ws_calc.cell(row=r, column=15, value=f"=IF(J{r}<0, 1, 0)").border = border_thin
        # P: Cumulative P&L = SUM($J$3:J{r})
        ws_calc.cell(row=r, column=16, value=f"=SUM($J$3:J{r})").border = border_thin
        # Q: Cumulative Return % = Cumulative P&L / Initial Capital * 100
        ws_calc.cell(row=r, column=17, value=f"=P{r}/METRICS!$C$14*100").border = border_thin
        # R: Equity = Initial Capital + Cumulative P&L
        ws_calc.cell(row=r, column=18, value=f"=METRICS!$C$14+P{r}").border = border_thin
        # S: Running Peak = MAX($R$3:R{r})
        ws_calc.cell(row=r, column=19, value=f"=MAX($R$3:R{r})").border = border_thin
        # T: Drawdown = Equity - Running Peak
        ws_calc.cell(row=r, column=20, value=f"=R{r}-S{r}").border = border_thin
        # U: Drawdown % = (Equity - Running Peak) / Running Peak * 100
        ws_calc.cell(row=r, column=21, value=f"=(R{r}-S{r})/S{r}*100").border = border_thin
        # V: Downside Return Sq = IF(K < 0, (K/100)^2, 0)
        ws_calc.cell(row=r, column=22, value=f"=IF(K{r}<0, (K{r}/100)^2, 0)").border = border_thin

    for col in ws_calc.columns:
        col_letter = get_column_letter(col[0].column)
        ws_calc.column_dimensions[col_letter].width = 15

    # =========================================================================
    # 3. METRICS Sheet (Transparent verification & Python reconciliation)
    # =========================================================================
    ws_met = wb.create_sheet(title="METRICS")
    ws_met.views.sheetView[0].showGridLines = True

    ws_met.merge_cells("A1:F1")
    ws_met["A1"] = "QUANTLAB METRICS VERIFICATION & RECONCILIATION"
    ws_met["A1"].font = title_font

    metric_headers = ["Metric Name", "Python Value", "Excel Formula", "Formula Used", "Difference", "Status"]
    for col_idx, h in enumerate(metric_headers, 1):
        cell = ws_met.cell(row=2, column=col_idx, value=h)
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = align_center
        cell.border = border_thin

    # Define all metrics, python values, excel formulas, formula descriptions
    metric_rows = [
        # row 3
        ("Total Trades", metrics.total_trades, f"=COUNT(TRADE_CALCULATION!J$3:J${end_row})", f"COUNT(TRADE_CALCULATION!J3:J{end_row})"),
        # row 4
        ("Winning Trades", metrics.winning_trades, f"=SUM(TRADE_CALCULATION!N$3:N${end_row})", f"SUM(TRADE_CALCULATION!N3:N{end_row})"),
        # row 5
        ("Losing Trades", metrics.losing_trades, f"=SUM(TRADE_CALCULATION!O$3:O${end_row})", f"SUM(TRADE_CALCULATION!O3:O{end_row})"),
        # row 6
        ("Win Rate", round(metrics.win_rate, 6), "=C4/C3", "=WinningTrades/TotalTrades (C4/C3)"),
        # row 7
        ("Loss Rate", round(metrics.loss_rate, 6), "=C5/C3", "=LosingTrades/TotalTrades (C5/C3)"),
        # row 8
        ("Gross Profit", round(metrics.gross_profit, 2), f'=SUMIF(TRADE_CALCULATION!J$3:J${end_row}, ">0")', f'SUMIF(TRADE_CALCULATION!J3:J{end_row}, ">0")'),
        # row 9
        ("Gross Loss", round(metrics.gross_loss, 2), f'=SUMIF(TRADE_CALCULATION!J$3:J${end_row}, "<0")', f'SUMIF(TRADE_CALCULATION!J3:J{end_row}, "<0")'),
        # row 10
        ("Profit Factor", round(metrics.profit_factor, 6), "=C8/ABS(C9)", "=GrossProfit/ABS(GrossLoss) (C8/ABS(C9))"),
        # row 11
        ("Average Win", round(metrics.avg_win, 2), "=C8/C4", "=GrossProfit/WinningTrades (C8/C4)"),
        # row 12
        ("Average Loss", round(metrics.avg_loss, 2), "=C9/C5", "=GrossLoss/LosingTrades (C9/C5)"),
        # row 13
        ("Expectancy", round(metrics.expectancy, 2), "=(C6*C11)+(C7*C12)", "=(WinRate*AvgWin)+(LossRate*AvgLoss) ((C6*C11)+(C7*C12))"),
        # row 14
        ("Initial Capital", config.initial_capital, f"={config.initial_capital}", f"{config.initial_capital}"),
        # row 15
        ("Total P&L", round(metrics.total_pnl, 2), f"=SUM(TRADE_CALCULATION!J$3:J${end_row})", f"SUM(TRADE_CALCULATION!J3:J{end_row})"),
        # row 16
        ("Final Equity", round(metrics.final_equity, 2), "=C14+C15", "=InitialCapital+TotalPnL (C14+C15)"),
        # row 17
        ("Total Return", round(metrics.total_return_pct / 100.0, 6), "=(C16-C14)/C14", "=(FinalEquity-InitialCapital)/InitialCapital"),
        # row 18
        ("Trading Years", round(metrics.years, 6), f"=(MAX(TRADE_CALCULATION!D$3:D${end_row})-MIN(TRADE_CALCULATION!C$3:C${end_row}))/365.25", "(MAX(ExitDate)-MIN(EntryDate))/365.25"),
        # row 19
        ("CAGR", round(metrics.cagr, 6), "=(C16/C14)^(1/C18)-1", "=(FinalEquity/InitialCapital)^(1/Years)-1"),
        # row 20
        ("Mean P&L", round(metrics.mean_pnl, 2), f"=AVERAGE(TRADE_CALCULATION!J$3:J${end_row})", f"AVERAGE(TRADE_CALCULATION!J3:J{end_row})"),
        # row 21
        ("Median P&L", round(metrics.median_pnl, 2), f"=MEDIAN(TRADE_CALCULATION!J$3:J${end_row})", f"MEDIAN(TRADE_CALCULATION!J3:J{end_row})"),
        # row 22
        ("Std Dev of P&L", round(metrics.pnl_std, 2), f"=STDEV.S(TRADE_CALCULATION!J$3:J${end_row})", f"STDEV.S(TRADE_CALCULATION!J3:J{end_row})"),
        # row 23
        ("Skewness", round(metrics.skewness, 6), f"=SKEW(TRADE_CALCULATION!J$3:J${end_row})", f"SKEW(TRADE_CALCULATION!J3:J{end_row})"),
        # row 24
        ("Maximum Drawdown", round(metrics.max_drawdown, 2), f"=MIN(TRADE_CALCULATION!T$3:T${end_row})", f"MIN(TRADE_CALCULATION!T3:T{end_row})"),
        # row 25
        ("Maximum Drawdown %", round(metrics.max_drawdown_pct, 6), f"=MIN(TRADE_CALCULATION!U$3:U${end_row})", f"MIN(TRADE_CALCULATION!U3:U{end_row})"),
        # row 26
        ("Calmar Ratio", round(metrics.calmar_ratio, 6), "=C19/ABS(C25/100)", "=CAGR/ABS(MaxDD%)"),
        # row 27
        ("Sharpe Ratio", round(metrics.sharpe_ratio, 6),
         f"=((AVERAGE(TRADE_CALCULATION!K$3:K${end_row})/100*(C3/C18))-{config.risk_free_rate})/(STDEV.S(TRADE_CALCULATION!K$3:K${end_row})/100*SQRT(C3/C18))",
         "(AnnReturn - RiskFreeRate) / AnnStdDev"),
        # row 28
        ("Sortino Ratio", round(metrics.sortino_ratio, 6),
         f"=((AVERAGE(TRADE_CALCULATION!K$3:K${end_row})/100*(C3/C18))-{config.risk_free_rate})/(SQRT(SUM(TRADE_CALCULATION!V$3:V${end_row})/(C3-1))*SQRT(C3/C18))",
         "(AnnReturn - RiskFreeRate) / AnnDownsideStdDev"),
        # row 29
        ("Max Win Streak", metrics.max_win_streak, f"={metrics.max_win_streak}", "Max consecutive winning trades"),
        # row 30
        ("Max Loss Streak", metrics.max_loss_streak, f"={metrics.max_loss_streak}", "Max consecutive losing trades")
    ]

    for idx, (m_name, py_val, xl_formula, f_used) in enumerate(metric_rows):
        r = idx + 3
        # Col A: Metric Name
        c_a = ws_met.cell(row=r, column=1, value=m_name)
        c_a.font = Font(name="Calibri", size=10, bold=True)
        c_a.border = border_thin

        # Col B: Python Value
        c_b = ws_met.cell(row=r, column=2, value=py_val)
        c_b.border = border_thin
        c_b.alignment = align_right

        # Col C: Excel Formula
        c_c = ws_met.cell(row=r, column=3, value=xl_formula)
        c_c.border = border_thin
        c_c.alignment = align_right

        # Col D: Formula Used
        c_d = ws_met.cell(row=r, column=4, value=f_used)
        c_d.border = border_thin
        c_d.font = Font(name="Courier New", size=9, color="4B5563")

        # Col E: Difference
        c_e = ws_met.cell(row=r, column=5, value=f"=ABS(B{r}-C{r})")
        c_e.border = border_thin
        c_e.alignment = align_right

        # Col F: Status
        c_f = ws_met.cell(row=r, column=6, value=f'=IF(E{r}<={config.tolerance}, "PASS", "FAIL")')
        c_f.border = border_thin
        c_f.alignment = align_center
        c_f.font = Font(name="Calibri", size=10, bold=True)

    ws_met.column_dimensions["A"].width = 24
    ws_met.column_dimensions["B"].width = 18
    ws_met.column_dimensions["C"].width = 30
    ws_met.column_dimensions["D"].width = 50
    ws_met.column_dimensions["E"].width = 16
    ws_met.column_dimensions["F"].width = 14

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    wb.save(output_path)
