# Overfitting (Curve Fitting) in Strategy Backtesting

## Definition
Overfitting occurs when a trading strategy is excessively tuned to past historical market noise rather than capturing a genuine underlying market anomaly or momentum edge.

## Signs of Overfitting
- Strategy performs exceptionally well in-sample (e.g. Sharpe > 3.0) but collapses completely out-of-sample or in live trading.
- Minor parameter tweaks (e.g. SMA short window from 20 to 21) cause drastic performance drops.

## Prevention Techniques
- Out-of-Sample (OOS) testing and Walk-Forward Validation.
- Parameter sensitivity analysis (Robustness testing across adjacent parameter grids).
- Keeping strategy rules simple with minimal free parameters.
