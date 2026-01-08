"""
Pytest fixtures for the Direct Indexing Shadow System tests.

Provides common test data and utilities used across test modules.
"""

import tempfile
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from di_pilot.models import (
    BenchmarkConstituent,
    PortfolioConfig,
    PriceData,
    TaxLot,
)


@pytest.fixture
def sample_portfolio_config() -> PortfolioConfig:
    """Create a sample portfolio configuration for testing."""
    return PortfolioConfig(
        portfolio_id="TEST001",
        cash=Decimal("1000000"),  # $1M
        start_date=date(2024, 1, 2),
        tlh_threshold=Decimal("0.03"),
        drift_threshold=Decimal("0.005"),
        min_trade_value=Decimal("100"),
        output_dir="output",
    )


@pytest.fixture
def sample_constituents() -> list[BenchmarkConstituent]:
    """Create sample S&P 500 constituents for testing."""
    return [
        BenchmarkConstituent(symbol="AAPL", weight=Decimal("0.07"), as_of_date=date(2024, 1, 2)),
        BenchmarkConstituent(symbol="MSFT", weight=Decimal("0.065"), as_of_date=date(2024, 1, 2)),
        BenchmarkConstituent(symbol="AMZN", weight=Decimal("0.035"), as_of_date=date(2024, 1, 2)),
        BenchmarkConstituent(symbol="NVDA", weight=Decimal("0.03"), as_of_date=date(2024, 1, 2)),
        BenchmarkConstituent(symbol="GOOGL", weight=Decimal("0.025"), as_of_date=date(2024, 1, 2)),
        BenchmarkConstituent(symbol="META", weight=Decimal("0.02"), as_of_date=date(2024, 1, 2)),
        BenchmarkConstituent(symbol="BRK.B", weight=Decimal("0.018"), as_of_date=date(2024, 1, 2)),
        BenchmarkConstituent(symbol="TSLA", weight=Decimal("0.015"), as_of_date=date(2024, 1, 2)),
        BenchmarkConstituent(symbol="UNH", weight=Decimal("0.013"), as_of_date=date(2024, 1, 2)),
        BenchmarkConstituent(symbol="JNJ", weight=Decimal("0.012"), as_of_date=date(2024, 1, 2)),
        # Normalize weights to sum to ~0.283 for this subset
        # Remaining weight distributed among other symbols
        BenchmarkConstituent(symbol="OTHER", weight=Decimal("0.717"), as_of_date=date(2024, 1, 2)),
    ]


@pytest.fixture
def sample_prices_jan() -> dict[str, PriceData]:
    """Create sample price data for January 2024."""
    price_date = date(2024, 1, 2)
    return {
        "AAPL": PriceData(symbol="AAPL", date=price_date, close=Decimal("185.64")),
        "MSFT": PriceData(symbol="MSFT", date=price_date, close=Decimal("374.58")),
        "AMZN": PriceData(symbol="AMZN", date=price_date, close=Decimal("151.94")),
        "NVDA": PriceData(symbol="NVDA", date=price_date, close=Decimal("481.68")),
        "GOOGL": PriceData(symbol="GOOGL", date=price_date, close=Decimal("140.25")),
        "META": PriceData(symbol="META", date=price_date, close=Decimal("346.29")),
        "BRK.B": PriceData(symbol="BRK.B", date=price_date, close=Decimal("355.98")),
        "TSLA": PriceData(symbol="TSLA", date=price_date, close=Decimal("248.48")),
        "UNH": PriceData(symbol="UNH", date=price_date, close=Decimal("524.51")),
        "JNJ": PriceData(symbol="JNJ", date=price_date, close=Decimal("156.74")),
        "OTHER": PriceData(symbol="OTHER", date=price_date, close=Decimal("100.00")),
    }


@pytest.fixture
def sample_prices_jun() -> dict[str, PriceData]:
    """Create sample price data for June 2024 (with some gains and losses)."""
    price_date = date(2024, 6, 15)
    return {
        "AAPL": PriceData(symbol="AAPL", date=price_date, close=Decimal("214.29")),  # +15.4%
        "MSFT": PriceData(symbol="MSFT", date=price_date, close=Decimal("442.57")),  # +18.2%
        "AMZN": PriceData(symbol="AMZN", date=price_date, close=Decimal("186.00")),  # +22.4%
        "NVDA": PriceData(symbol="NVDA", date=price_date, close=Decimal("129.61")),  # -73.1% (post-split adjusted)
        "GOOGL": PriceData(symbol="GOOGL", date=price_date, close=Decimal("177.29")),  # +26.4%
        "META": PriceData(symbol="META", date=price_date, close=Decimal("505.95")),  # +46.1%
        "BRK.B": PriceData(symbol="BRK.B", date=price_date, close=Decimal("411.92")),  # +15.7%
        "TSLA": PriceData(symbol="TSLA", date=price_date, close=Decimal("182.47")),  # -26.6%
        "UNH": PriceData(symbol="UNH", date=price_date, close=Decimal("495.62")),  # -5.5%
        "JNJ": PriceData(symbol="JNJ", date=price_date, close=Decimal("146.07")),  # -6.8%
        "OTHER": PriceData(symbol="OTHER", date=price_date, close=Decimal("95.00")),  # -5%
    }


@pytest.fixture
def sample_lots(sample_portfolio_config: PortfolioConfig) -> list[TaxLot]:
    """Create sample tax lots for testing."""
    return [
        TaxLot(
            lot_id="lot-001",
            symbol="AAPL",
            shares=Decimal("377.062"),
            cost_basis=Decimal("185.64"),
            acquisition_date=date(2024, 1, 2),
            portfolio_id="TEST001",
        ),
        TaxLot(
            lot_id="lot-002",
            symbol="MSFT",
            shares=Decimal("173.524"),
            cost_basis=Decimal("374.58"),
            acquisition_date=date(2024, 1, 2),
            portfolio_id="TEST001",
        ),
        TaxLot(
            lot_id="lot-003",
            symbol="AMZN",
            shares=Decimal("230.359"),
            cost_basis=Decimal("151.94"),
            acquisition_date=date(2024, 1, 2),
            portfolio_id="TEST001",
        ),
        TaxLot(
            lot_id="lot-004",
            symbol="NVDA",
            shares=Decimal("62.287"),
            cost_basis=Decimal("481.68"),
            acquisition_date=date(2024, 1, 2),
            portfolio_id="TEST001",
        ),
        TaxLot(
            lot_id="lot-005",
            symbol="TSLA",
            shares=Decimal("60.367"),
            cost_basis=Decimal("248.48"),
            acquisition_date=date(2024, 1, 2),
            portfolio_id="TEST001",
        ),
    ]


@pytest.fixture
def temp_output_dir() -> Path:
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_csv_file(temp_output_dir: Path) -> Path:
    """Create a temporary CSV file path."""
    return temp_output_dir / "test_data.csv"
