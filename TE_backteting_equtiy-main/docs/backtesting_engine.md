# QUANTLAB — Backtesting Engine Architecture

The **QUANTLAB Upgraded Backtesting Engine** (`src/engine/`) is an event-driven, point-in-time safe simulation architecture for historical equity strategy evaluation.

## 1. Architecture Flow

```text
HISTORICAL DATA
       ↓
DATA VALIDATION (DataValidator)
       ↓
STRATEGY & SIGNAL GENERATION (BUY / SELL / HOLD)
       ↓
POSITION SIZING (PositionSizer)
       ↓
ORDER GENERATION (Order)
       ↓
EXECUTION MODEL (ExecutionModel & SlippageModel)
       ↓
FILL GENERATION (Fill)
       ↓
PORTFOLIO STATE (PortfolioManager)
       ↓
METRICS ENGINE (calculate_metrics)
       ↓
BENCHMARK & RESULTS
```

## 2. Distinction: Signal vs Trade

In QUANTLAB, **SIGNAL ≠ TRADE**.

1. **Signal**: Output of strategy rules evaluating historical bar indicators at Close time $T$.
2. **Order**: Intent to execute trade with specified side, quantity, reference price, and order type (`MARKET`).
3. **Execution**: Resolution of base price at execution bar (e.g. Bar $T+1$ Open under `NEXT_BAR_OPEN` mode).
4. **Fill**: Executed trade record after applying adverse price slippage and transaction brokerage fees.
5. **Portfolio State**: Dynamic mark-to-market updates for cash balance, held positions, realized P&L, unrealized P&L, and total portfolio value.

## 3. Package Structure (`src/engine/`)

- `config.py`: `BacktestConfig` dataclass and timing/policy enums.
- `order.py`: `Order` dataclass for order lifecycle management.
- `position_sizing.py`: `PositionSizer` for capital allocation calculation $\lfloor \text{Capital} / \text{Price} \rfloor$.
- `execution.py`: `ExecutionModel` & `SlippageModel` (basis points slippage & `NEXT_BAR_OPEN` timing).
- `cost_model.py`: `TransactionCostModel` calculating brokerage commissions.
- `fill.py`: `Fill` dataclass storing canonical trade fill details.
- `portfolio.py`: `PositionTracker` and `PortfolioManager` for P&L and cash constraints.
- `trade_log.py`: `TradeLogger` converting fills to canonical DataFrame format.
- `validation.py`: `DataValidator` enforcing point-in-time data integrity checks.
- `result.py`: `BacktestResult` output container with AI tool compatibility accessors.
- `engine.py`: `BacktestEngine` bar-by-bar orchestrator and `run_backtest` wrapper.
- `demo.py`: Reproducible demo script comparing realistic vs zero-cost backtests.
