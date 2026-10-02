"""Configuration loader and settings for QuantLab Backtesting Engine."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional
import os
import yaml

INITIAL_CAPITAL = 10000000.0
CAPITAL_ALLOCATION_PER_TRADE = 0.02
RISK_FREE_RATE = 0.06
STRATEGY_MODE = "long_only"
POSITION_SIZING = "fixed"
TOLERANCE = 0.0001


@dataclass
class BacktestConfig:
    """Backtesting configuration model."""
    initial_capital: float = INITIAL_CAPITAL
    capital_allocation_per_trade: float = CAPITAL_ALLOCATION_PER_TRADE
    risk_free_rate: float = RISK_FREE_RATE
    strategy_mode: str = STRATEGY_MODE
    position_sizing: str = POSITION_SIZING
    transaction_cost: float = 0.0
    slippage: float = 0.0
    tolerance: float = TOLERANCE
    output_dir: str = "output"
    openai_api_key: Optional[str] = field(default=None)
    openai_model: str = "gpt-4o-mini"

    def __post_init__(self):
        valid_modes = {"long_only", "short_only", "long_short"}
        if self.strategy_mode not in valid_modes:
            raise ValueError(f"Invalid strategy_mode: {self.strategy_mode}. Must be one of {valid_modes}")

        valid_sizings = {"fixed", "compounding"}
        if self.position_sizing not in valid_sizings:
            raise ValueError(f"Invalid position_sizing: {self.position_sizing}. Must be one of {valid_sizings}")

        if not self.openai_api_key:
            self.openai_api_key = os.environ.get("OPENAI_API_KEY", None)


def load_config(config_path: Optional[str] = None, **overrides: Any) -> BacktestConfig:
    """Load configuration from YAML file and apply CLI/runtime overrides."""
    config_dict: Dict[str, Any] = {}

    # Check potential default paths if not provided
    paths_to_check = []
    if config_path:
        paths_to_check.append(Path(config_path))
    else:
        paths_to_check.extend([
            Path("config.yaml"),
            Path(__file__).resolve().parent.parent / "config.yaml"
        ])

    for p in paths_to_check:
        if p.exists() and p.is_file():
            with open(p, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                if isinstance(data, dict):
                    config_dict.update(data)
            break

    # Apply overrides (ignoring None values)
    for k, v in overrides.items():
        if v is not None:
            config_dict[k] = v

    # Convert numeric fields
    if "initial_capital" in config_dict:
        config_dict["initial_capital"] = float(config_dict["initial_capital"])
    if "capital_allocation_per_trade" in config_dict:
        config_dict["capital_allocation_per_trade"] = float(config_dict["capital_allocation_per_trade"])
    if "risk_free_rate" in config_dict:
        config_dict["risk_free_rate"] = float(config_dict["risk_free_rate"])
    if "tolerance" in config_dict:
        config_dict["tolerance"] = float(config_dict["tolerance"])

    return BacktestConfig(**config_dict)
