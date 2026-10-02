"""QuantLab Backtesting System - Main CLI and Pipeline Orchestrator."""

import argparse
from pathlib import Path
import sys
import os

# Add parent directory to sys.path so quantlab can be executed from anywhere
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from src.config import load_config
from src.data_loader import load_signal_file
from src.backtest_engine import run_backtest, generate_trade_log_excel
from src.drawdown import (
    build_daily_equity_curve,
    export_drawdown_data,
    plot_equity_and_drawdown
)
from src.returns import (
    calculate_monthly_and_yearly_returns,
    export_returns_reports,
    plot_yearly_return_histogram
)
from src.metrics import calculate_metrics
from src.excel_verifier import generate_manual_verification_workbook
from src.llm_validator import run_llm_validation
from src.reconciliation import reconcile_metrics, print_reconciliation_table
from src.report_generator import generate_master_backtest_report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="QuantLab Quantitative Backtesting Engine"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to Excel signal/trade file (e.g. 'trade logbook-Signal file.xlsx')"
    )
    parser.add_argument(
        "--config", "-c",
        default="config.yaml",
        help="Path to config.yaml (default: 'config.yaml')"
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=None,
        help="Initial capital in INR (default: 10,000,000)"
    )
    parser.add_argument(
        "--allocation",
        type=float,
        default=None,
        help="Capital allocated per trade as fraction (default: 0.02)"
    )
    parser.add_argument(
        "--risk-free",
        type=float,
        default=None,
        help="Risk-free rate (default: 0.06)"
    )
    parser.add_argument(
        "--mode",
        default=None,
        choices=["long_only", "short_only", "long_short"],
        help="Strategy mode (default: long_only)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="output",
        help="Directory to store generated reports (default: 'output')"
    )

    args = parser.parse_args()

    # Resolve input path
    input_path = Path(args.input)
    if not input_path.exists():
        # Check in input/ or validation_signal/ or current folder
        alt_paths = [
            CURRENT_DIR / "input" / input_path.name,
            CURRENT_DIR / input_path,
            CURRENT_DIR.parent / "validation_signal" / input_path.name,
            CURRENT_DIR / "validation_signal" / input_path.name
        ]
        found = False
        for p in alt_paths:
            if p.exists():
                input_path = p
                found = True
                break
        if not found:
            print(f"Error: Signal file '{args.input}' not found.", file=sys.stderr)
            sys.exit(1)

    # Load configuration
    cfg_path = args.config if Path(args.config).exists() else str(CURRENT_DIR / "config.yaml")
    config = load_config(
        config_path=cfg_path,
        initial_capital=args.capital,
        capital_allocation_per_trade=args.allocation,
        risk_free_rate=args.risk_free,
        strategy_mode=args.mode,
        output_dir=args.output_dir
    )

    # Ensure output directory exists
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Input Excel Processing
    raw_df, val_report = load_signal_file(str(input_path))
    val_report.print_summary()

    # 2. Position and P&L Engine
    trades = run_backtest(raw_df, config)

    # 3. Drawdown Engine & Daily Series
    daily_df = build_daily_equity_curve(trades, config)
    export_drawdown_data(daily_df, str(out_dir))

    # 4. Performance Metrics
    metrics = calculate_metrics(trades, daily_df, config)

    # 5. Charts Generation
    plot_equity_and_drawdown(daily_df, metrics, str(out_dir))

    # 6. Monthly & Yearly Return Logs
    monthly_df, yearly_df = calculate_monthly_and_yearly_returns(daily_df)
    export_returns_reports(monthly_df, yearly_df, str(out_dir))
    plot_yearly_return_histogram(yearly_df, str(out_dir))

    # 7. Generate Trade Log Excel
    generate_trade_log_excel(trades, str(out_dir / "trade_log.xlsx"))

    # 8. Manual Excel Verification Workbook
    verification_path = str(out_dir / "manual_backtest_verification.xlsx")
    generate_manual_verification_workbook(raw_df, trades, metrics, config, verification_path)

    # 9. LLM Validation
    llm_audit, llm_text = run_llm_validation(trades, metrics, config, str(out_dir))

    # 10. Python vs Excel Reconciliation
    reconciliation_df, matches, total_reconciled = reconcile_metrics(metrics, config, llm_audit)
    print_reconciliation_table(reconciliation_df, matches, total_reconciled)

    # 11. Master Backtest Report (10 sheets)
    report_path = str(out_dir / "backtest_report.xlsx")
    generate_master_backtest_report(
        trades=trades,
        daily_df=daily_df,
        monthly_df=monthly_df,
        yearly_df=yearly_df,
        metrics=metrics,
        val_report=val_report,
        reconciliation_df=reconciliation_df,
        llm_audit=llm_audit,
        llm_text=llm_text,
        config=config,
        output_path=report_path
    )

    # Print final user-friendly summary matching Section 27
    llm_status_str = "COMPLETED" if llm_audit.get("status") == "COMPLETED" else "NOT CONFIGURED"
    if config.openai_api_key:
        llm_status_str += f" ({config.openai_model})"
    else:
        llm_status_str += " (Deterministic Auditor)"

    print("====================================================")
    print("QUANTLAB BACKTEST COMPLETE")
    print("====================================================")
    print(f"\nInput:\n{input_path.name}")
    print(f"\nInitial Capital:\n₹{metrics.initial_capital:,.0f}")
    print(f"\nFinal Equity:\n₹{metrics.final_equity:,.2f}")
    print(f"\nTotal P&L:\n₹{metrics.total_pnl:,.2f}")
    print(f"\nTotal Return:\n{metrics.total_return_pct:.2f}%")
    print(f"\nCAGR:\n{metrics.cagr * 100:.2f}%")
    print(f"\nTotal Trades:\n{metrics.total_trades}")
    print(f"\nWin Rate:\n{metrics.win_rate * 100:.2f}%")
    print(f"\nProfit Factor:\n{metrics.profit_factor:.2f}")
    print(f"\nExpectancy:\n₹{metrics.expectancy:,.2f}")
    print(f"\nSharpe:\n{metrics.sharpe_ratio:.2f}")
    print(f"\nSortino:\n{metrics.sortino_ratio:.2f}")
    print(f"\nCalmar:\n{metrics.calmar_ratio:.2f}")
    print(f"\nMaximum Drawdown:\n₹{metrics.max_drawdown:,.2f}")
    print(f"\nMaximum Drawdown %:\n{metrics.max_drawdown_pct:.2f}%")
    print(f"\nSkewness:\n{metrics.skewness:.4f}")
    print(f"\nMax Win Streak:\n{metrics.max_win_streak}")
    print(f"\nMax Loss Streak:\n{metrics.max_loss_streak}")
    print("\n====================================================")
    print("VALIDATION")
    print("====================================================")
    print(f"\nPython vs Excel:\n{matches}/{total_reconciled} metrics matched")
    print(f"\nLLM Validation:\n{llm_status_str}")
    print("\n====================================================")
    print("FILES GENERATED")
    print("====================================================")
    print(f"\n{out_dir / 'backtest_report.xlsx'}")
    print(f"{out_dir / 'manual_backtest_verification.xlsx'}")
    print(f"{out_dir / 'trade_log.xlsx'}")
    print(f"{out_dir / 'monthly_returns.xlsx'}")
    print(f"{out_dir / 'yearly_returns.xlsx'}")
    print(f"{out_dir / 'equity_curve.png'}")
    print(f"{out_dir / 'drawdown_curve.png'}")
    print(f"{out_dir / 'yearly_return_histogram.png'}")
    print(f"{out_dir / 'llm_validation_report.json'}")
    print(f"{out_dir / 'llm_validation_report.txt'}\n")


if __name__ == "__main__":
    main()
