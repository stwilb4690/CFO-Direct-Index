"""
Configuration loading and management for the Direct Indexing Shadow System.

This module handles loading portfolio configurations from YAML files and
provides validation of configuration parameters.
"""

from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import yaml

from di_pilot.models import PortfolioConfig


class ConfigurationError(Exception):
    """Raised when configuration is invalid or cannot be loaded."""
    pass


def load_portfolio_config(config_path: str | Path) -> PortfolioConfig:
    """
    Load portfolio configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file

    Returns:
        PortfolioConfig object with validated settings

    Raises:
        ConfigurationError: If the file cannot be loaded or is invalid
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise ConfigurationError(f"Configuration file not found: {config_path}")

    try:
        with open(config_path, "r") as f:
            raw_config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigurationError(f"Invalid YAML in configuration file: {e}")

    return _parse_portfolio_config(raw_config)


def _parse_portfolio_config(raw: dict[str, Any]) -> PortfolioConfig:
    """
    Parse and validate raw configuration dictionary into PortfolioConfig.

    Args:
        raw: Dictionary loaded from YAML

    Returns:
        Validated PortfolioConfig

    Raises:
        ConfigurationError: If required fields are missing or invalid
    """
    required_fields = ["portfolio_id", "cash", "start_date"]
    for field in required_fields:
        if field not in raw:
            raise ConfigurationError(f"Missing required configuration field: {field}")

    # Parse portfolio_id
    portfolio_id = str(raw["portfolio_id"])
    if not portfolio_id:
        raise ConfigurationError("portfolio_id cannot be empty")

    # Parse cash amount
    try:
        cash = Decimal(str(raw["cash"]))
        if cash <= 0:
            raise ConfigurationError("cash must be positive")
    except Exception as e:
        raise ConfigurationError(f"Invalid cash value: {e}")

    # Parse start_date
    start_date = _parse_date(raw["start_date"], "start_date")

    # Parse optional parameters with defaults
    tlh_threshold = _parse_decimal(
        raw.get("tlh_threshold", "0.03"),
        "tlh_threshold",
        min_val=Decimal("0"),
        max_val=Decimal("1"),
    )

    drift_threshold = _parse_decimal(
        raw.get("drift_threshold", "0.005"),
        "drift_threshold",
        min_val=Decimal("0"),
        max_val=Decimal("1"),
    )

    min_trade_value = _parse_decimal(
        raw.get("min_trade_value", "100"),
        "min_trade_value",
        min_val=Decimal("0"),
    )

    output_dir = str(raw.get("output_dir", "output"))

    return PortfolioConfig(
        portfolio_id=portfolio_id,
        cash=cash,
        start_date=start_date,
        tlh_threshold=tlh_threshold,
        drift_threshold=drift_threshold,
        min_trade_value=min_trade_value,
        output_dir=output_dir,
    )


def _parse_date(value: Any, field_name: str) -> date:
    """
    Parse a date value from various formats.

    Args:
        value: The value to parse (string or date object)
        field_name: Name of the field for error messages

    Returns:
        Parsed date object

    Raises:
        ConfigurationError: If the date cannot be parsed
    """
    if isinstance(value, date):
        return value

    if isinstance(value, datetime):
        return value.date()

    if isinstance(value, str):
        try:
            return datetime.strptime(value, "%Y-%m-%d").date()
        except ValueError:
            pass

        try:
            return datetime.fromisoformat(value).date()
        except ValueError:
            pass

    raise ConfigurationError(
        f"Invalid date format for {field_name}: {value}. Expected YYYY-MM-DD"
    )


def _parse_decimal(
    value: Any,
    field_name: str,
    min_val: Decimal | None = None,
    max_val: Decimal | None = None,
) -> Decimal:
    """
    Parse a decimal value with optional range validation.

    Args:
        value: The value to parse
        field_name: Name of the field for error messages
        min_val: Minimum allowed value (inclusive)
        max_val: Maximum allowed value (inclusive)

    Returns:
        Parsed Decimal

    Raises:
        ConfigurationError: If the value is invalid or out of range
    """
    try:
        decimal_value = Decimal(str(value))
    except Exception:
        raise ConfigurationError(f"Invalid decimal value for {field_name}: {value}")

    if min_val is not None and decimal_value < min_val:
        raise ConfigurationError(
            f"{field_name} must be >= {min_val}, got {decimal_value}"
        )

    if max_val is not None and decimal_value > max_val:
        raise ConfigurationError(
            f"{field_name} must be <= {max_val}, got {decimal_value}"
        )

    return decimal_value


def create_default_config(
    portfolio_id: str,
    cash: Decimal,
    start_date: date,
    output_path: str | Path | None = None,
) -> PortfolioConfig:
    """
    Create a portfolio config with default parameters.

    Useful for programmatic configuration without a YAML file.

    Args:
        portfolio_id: Unique portfolio identifier
        cash: Initial cash amount
        start_date: Portfolio start date
        output_path: Optional path to write config YAML

    Returns:
        PortfolioConfig with default thresholds
    """
    config = PortfolioConfig(
        portfolio_id=portfolio_id,
        cash=cash,
        start_date=start_date,
    )

    if output_path:
        write_config(config, output_path)

    return config


def write_config(config: PortfolioConfig, output_path: str | Path) -> None:
    """
    Write a PortfolioConfig to a YAML file.

    Args:
        config: The configuration to write
        output_path: Path to write the YAML file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    config_dict = {
        "portfolio_id": config.portfolio_id,
        "cash": str(config.cash),
        "start_date": config.start_date.isoformat(),
        "tlh_threshold": str(config.tlh_threshold),
        "drift_threshold": str(config.drift_threshold),
        "min_trade_value": str(config.min_trade_value),
        "output_dir": config.output_dir,
    }

    with open(output_path, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
