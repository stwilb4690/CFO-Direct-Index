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


# =============================================================================
# EODHD Provider Fixtures
# =============================================================================


@pytest.fixture
def sample_eodhd_price_data() -> dict[str, list[dict]]:
    """
    Sample EODHD API price response data.

    Format matches the EODHD EOD endpoint:
    https://eodhd.com/api/eod/{symbol}.US
    """
    return {
        "AAPL": [
            {
                "date": "2024-01-02",
                "open": 184.22,
                "high": 185.88,
                "low": 183.43,
                "close": 185.64,
                "adjusted_close": 185.64,
                "volume": 82488700,
            },
            {
                "date": "2024-01-03",
                "open": 184.22,
                "high": 185.15,
                "low": 183.20,
                "close": 184.25,
                "adjusted_close": 184.25,
                "volume": 58414460,
            },
            {
                "date": "2024-01-04",
                "open": 182.15,
                "high": 183.09,
                "low": 180.88,
                "close": 181.91,
                "adjusted_close": 181.91,
                "volume": 71983640,
            },
            {
                "date": "2024-01-05",
                "open": 181.99,
                "high": 182.76,
                "low": 180.17,
                "close": 181.18,
                "adjusted_close": 181.18,
                "volume": 62303340,
            },
        ],
        "MSFT": [
            {
                "date": "2024-01-02",
                "open": 373.86,
                "high": 376.04,
                "low": 371.32,
                "close": 374.58,
                "adjusted_close": 374.58,
                "volume": 18157000,
            },
            {
                "date": "2024-01-03",
                "open": 372.45,
                "high": 373.30,
                "low": 368.68,
                "close": 370.87,
                "adjusted_close": 370.87,
                "volume": 20264000,
            },
            {
                "date": "2024-01-04",
                "open": 369.15,
                "high": 370.73,
                "low": 366.00,
                "close": 366.53,
                "adjusted_close": 366.53,
                "volume": 20547400,
            },
            {
                "date": "2024-01-05",
                "open": 367.36,
                "high": 368.50,
                "low": 365.25,
                "close": 367.94,
                "adjusted_close": 367.94,
                "volume": 18932500,
            },
        ],
        "SPY": [
            {
                "date": "2024-01-02",
                "open": 472.50,
                "high": 474.13,
                "low": 469.95,
                "close": 472.65,
                "adjusted_close": 472.65,
                "volume": 65852600,
            },
            {
                "date": "2024-01-03",
                "open": 470.50,
                "high": 471.41,
                "low": 468.00,
                "close": 468.79,
                "adjusted_close": 468.79,
                "volume": 73152200,
            },
            {
                "date": "2024-01-04",
                "open": 468.25,
                "high": 469.40,
                "low": 465.18,
                "close": 465.92,
                "adjusted_close": 465.92,
                "volume": 68136600,
            },
            {
                "date": "2024-01-05",
                "open": 467.22,
                "high": 468.86,
                "low": 465.44,
                "close": 467.92,
                "adjusted_close": 467.92,
                "volume": 63765700,
            },
            {
                "date": "2024-01-08",
                "open": 469.70,
                "high": 474.04,
                "low": 469.60,
                "close": 473.62,
                "adjusted_close": 473.62,
                "volume": 52019300,
            },
        ],
    }


@pytest.fixture
def sample_eodhd_constituents_data() -> dict:
    """
    Sample EODHD API constituents response data.

    Format matches the EODHD fundamentals endpoint for GSPC.INDX
    """
    return {
        "General": {
            "Code": "GSPC",
            "Type": "INDEX",
            "Name": "S&P 500",
        },
        "Components": {
            "AAPL.US": {"Code": "AAPL", "Exchange": "US", "Name": "Apple Inc", "Weight": 7.12},
            "MSFT.US": {"Code": "MSFT", "Exchange": "US", "Name": "Microsoft Corp", "Weight": 6.85},
            "AMZN.US": {"Code": "AMZN", "Exchange": "US", "Name": "Amazon.com Inc", "Weight": 3.42},
            "NVDA.US": {"Code": "NVDA", "Exchange": "US", "Name": "NVIDIA Corp", "Weight": 3.15},
            "GOOGL.US": {"Code": "GOOGL", "Exchange": "US", "Name": "Alphabet Inc Class A", "Weight": 2.08},
            "GOOG.US": {"Code": "GOOG", "Exchange": "US", "Name": "Alphabet Inc Class C", "Weight": 1.79},
            "META.US": {"Code": "META", "Exchange": "US", "Name": "Meta Platforms Inc", "Weight": 2.45},
            "BRK.B.US": {"Code": "BRK.B", "Exchange": "US", "Name": "Berkshire Hathaway Inc", "Weight": 1.72},
            "TSLA.US": {"Code": "TSLA", "Exchange": "US", "Name": "Tesla Inc", "Weight": 1.58},
            "UNH.US": {"Code": "UNH", "Exchange": "US", "Name": "UnitedHealth Group Inc", "Weight": 1.25},
            "JNJ.US": {"Code": "JNJ", "Exchange": "US", "Name": "Johnson & Johnson", "Weight": 1.12},
            "JPM.US": {"Code": "JPM", "Exchange": "US", "Name": "JPMorgan Chase & Co", "Weight": 1.18},
            "V.US": {"Code": "V", "Exchange": "US", "Name": "Visa Inc", "Weight": 1.05},
            "XOM.US": {"Code": "XOM", "Exchange": "US", "Name": "Exxon Mobil Corp", "Weight": 1.08},
            "PG.US": {"Code": "PG", "Exchange": "US", "Name": "Procter & Gamble Co", "Weight": 0.95},
            # Remaining weight distributed among ~490 other stocks
            # Using a simplified subset for testing
        },
    }


@pytest.fixture
def mock_eodhd_response():
    """
    Factory fixture for creating mock EODHD API responses.

    Usage:
        def test_something(mock_eodhd_response):
            response = mock_eodhd_response(status_code=200, json_data={...})
    """
    from unittest.mock import MagicMock

    def _create_response(status_code: int = 200, json_data=None, raise_for_status=False):
        mock_response = MagicMock()
        mock_response.status_code = status_code
        mock_response.json.return_value = json_data or []

        if raise_for_status:
            mock_response.raise_for_status.side_effect = Exception("HTTP Error")
        else:
            mock_response.raise_for_status.return_value = None

        return mock_response

    return _create_response
