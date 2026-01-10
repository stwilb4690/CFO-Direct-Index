"""
Configuration loading and management for the Direct Indexing Shadow System.

This module handles loading portfolio configurations from YAML files,
API key management, and validation of configuration parameters.
"""

import os
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import yaml

from di_pilot.models import PortfolioConfig


# Default paths for configuration files
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
DEFAULT_ENV_FILE = PROJECT_ROOT / ".env"
DEFAULT_API_KEYS_FILE = PROJECT_ROOT / "config" / "api_keys.yaml"


class ConfigurationError(Exception):
    """Raised when configuration is invalid or cannot be loaded."""
    pass


def load_api_keys(
    env_file: str | Path | None = None,
    api_keys_file: str | Path | None = None,
) -> dict[str, str]:
    """
    Load API keys from multiple sources with priority.

    Sources are checked in this order (later sources override earlier):
    1. config/api_keys.yaml file
    2. .env file in project root
    3. Environment variables

    Args:
        env_file: Path to .env file (defaults to project root .env)
        api_keys_file: Path to api_keys.yaml (defaults to config/api_keys.yaml)

    Returns:
        Dictionary with API keys:
        - eodhd_api_key: EODHD API key (if available)

    Example:
        >>> keys = load_api_keys()
        >>> eodhd_key = keys.get("eodhd_api_key")
    """
    api_keys: dict[str, str] = {}

    # 1. Load from config/api_keys.yaml
    yaml_path = Path(api_keys_file) if api_keys_file else DEFAULT_API_KEYS_FILE
    if yaml_path.exists():
        try:
            with open(yaml_path, "r") as f:
                yaml_config = yaml.safe_load(f) or {}
            if isinstance(yaml_config, dict):
                if "eodhd_api_key" in yaml_config:
                    api_keys["eodhd_api_key"] = str(yaml_config["eodhd_api_key"])
        except (yaml.YAMLError, OSError):
            pass  # Silently ignore yaml errors

    # 2. Load from .env file
    env_path = Path(env_file) if env_file else DEFAULT_ENV_FILE
    if env_path.exists():
        try:
            from dotenv import dotenv_values
            env_values = dotenv_values(env_path)
            if "EODHD_API_KEY" in env_values:
                api_keys["eodhd_api_key"] = str(env_values["EODHD_API_KEY"])
        except ImportError:
            # python-dotenv not installed, try manual parsing
            api_keys.update(_parse_env_file(env_path))

    # 3. Override with environment variables (highest priority)
    if os.environ.get("EODHD_API_KEY"):
        api_keys["eodhd_api_key"] = os.environ["EODHD_API_KEY"]

    return api_keys


def _parse_env_file(env_path: Path) -> dict[str, str]:
    """
    Simple .env file parser as fallback when python-dotenv is not installed.

    Args:
        env_path: Path to .env file

    Returns:
        Dictionary with parsed environment variables
    """
    result: dict[str, str] = {}
    try:
        with open(env_path, "r") as f:
            for line in f:
                line = line.strip()
                # Skip comments and empty lines
                if not line or line.startswith("#"):
                    continue
                # Parse KEY=VALUE format
                if "=" in line:
                    key, _, value = line.partition("=")
                    key = key.strip()
                    value = value.strip()
                    # Remove quotes if present
                    if (value.startswith('"') and value.endswith('"')) or \
                       (value.startswith("'") and value.endswith("'")):
                        value = value[1:-1]
                    if key == "EODHD_API_KEY":
                        result["eodhd_api_key"] = value
    except OSError:
        pass
    return result


def get_eodhd_api_key() -> str:
    """
    Get the EODHD API key from available configuration sources.

    Returns:
        The EODHD API key

    Raises:
        ConfigurationError: If EODHD_API_KEY is not configured
    """
    api_keys = load_api_keys()
    if "eodhd_api_key" not in api_keys or not api_keys["eodhd_api_key"]:
        raise ConfigurationError(
            "EODHD API key is not configured. Please set it using one of:\n"
            "  1. Environment variable: export EODHD_API_KEY=your-key\n"
            "  2. .env file: EODHD_API_KEY=your-key\n"
            "  3. config/api_keys.yaml: eodhd_api_key: your-key\n"
            "\n"
            "Get your API key at: https://eodhd.com/"
        )
    return api_keys["eodhd_api_key"]


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
