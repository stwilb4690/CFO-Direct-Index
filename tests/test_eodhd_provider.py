"""
Tests for the EODHD data provider.

Unit tests use mocked API responses.
Integration tests (marked with @pytest.mark.integration) require a valid API key.
"""

import os
from datetime import date, timedelta
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import requests

from di_pilot.data.providers.base import DataProviderError
from di_pilot.data.providers.eodhd_provider import EODHDProvider, get_eodhd_provider
from di_pilot.models import BenchmarkConstituent, PriceData


# =============================================================================
# Unit Tests (mocked API responses)
# =============================================================================


class TestEODHDProviderInit:
    """Tests for EODHDProvider initialization."""

    def test_init_with_api_key(self):
        """Test initialization with explicit API key."""
        provider = EODHDProvider(api_key="test-api-key")
        assert provider._api_key == "test-api-key"
        assert provider.name == "EODHD"
        assert provider.supports_historical_constituents is True

    def test_init_from_environment(self, monkeypatch):
        """Test initialization from environment variable."""
        monkeypatch.setenv("EODHD_API_KEY", "env-api-key")
        provider = EODHDProvider()
        assert provider._api_key == "env-api-key"

    def test_missing_api_key_error(self, monkeypatch):
        """Test that missing API key raises DataProviderError."""
        # Clear any existing env var
        monkeypatch.delenv("EODHD_API_KEY", raising=False)

        # Mock load_api_keys to return empty dict
        with patch("di_pilot.data.providers.eodhd_provider.load_api_keys") as mock_load:
            mock_load.return_value = {}

            with pytest.raises(DataProviderError) as exc_info:
                EODHDProvider()

            assert "EODHD API key is not configured" in str(exc_info.value)
            assert "EODHD_API_KEY" in str(exc_info.value)


class TestGetPrices:
    """Tests for get_prices method."""

    @pytest.fixture
    def provider(self):
        """Create provider with test API key."""
        return EODHDProvider(api_key="test-api-key")

    def test_get_prices_single_symbol(
        self, provider, mock_eodhd_response, sample_eodhd_price_data
    ):
        """Test fetching prices for a single symbol."""
        with patch("requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = sample_eodhd_price_data["AAPL"]
            mock_get.return_value = mock_response

            df = provider.get_prices(
                ["AAPL"],
                date(2024, 1, 2),
                date(2024, 1, 5),
            )

            assert len(df) > 0
            assert "date" in df.columns
            assert "symbol" in df.columns
            assert "close" in df.columns
            assert df["symbol"].iloc[0] == "AAPL"

    def test_get_prices_multiple_symbols(
        self, provider, sample_eodhd_price_data
    ):
        """Test fetching prices for multiple symbols."""
        with patch("requests.get") as mock_get:
            def mock_response_factory(*args, **kwargs):
                url = args[0]
                mock_response = MagicMock()
                mock_response.status_code = 200

                if "AAPL" in url:
                    mock_response.json.return_value = sample_eodhd_price_data["AAPL"]
                elif "MSFT" in url:
                    mock_response.json.return_value = sample_eodhd_price_data["MSFT"]
                else:
                    mock_response.json.return_value = []

                return mock_response

            mock_get.side_effect = mock_response_factory

            df = provider.get_prices(
                ["AAPL", "MSFT"],
                date(2024, 1, 2),
                date(2024, 1, 5),
            )

            assert len(df) > 0
            symbols = df["symbol"].unique()
            assert "AAPL" in symbols
            assert "MSFT" in symbols

    def test_get_prices_empty_symbols(self, provider):
        """Test fetching prices with empty symbol list."""
        df = provider.get_prices([], date(2024, 1, 2), date(2024, 1, 5))
        assert df.empty
        assert list(df.columns) == ["date", "symbol", "close"]

    def test_get_prices_handles_missing_data(self, provider):
        """Test handling of symbols with no data."""
        with patch("requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = []  # No data
            mock_get.return_value = mock_response

            df = provider.get_prices(
                ["NONEXISTENT"],
                date(2024, 1, 2),
                date(2024, 1, 5),
            )

            assert df.empty

    def test_get_prices_normalizes_symbols(self, provider, sample_eodhd_price_data):
        """Test that symbols are normalized to uppercase."""
        with patch("requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = sample_eodhd_price_data["AAPL"]
            mock_get.return_value = mock_response

            df = provider.get_prices(
                ["aapl", " AAPL "],  # Lowercase and with spaces
                date(2024, 1, 2),
                date(2024, 1, 5),
            )

            # Should deduplicate and normalize
            assert len(df["symbol"].unique()) == 1
            assert df["symbol"].iloc[0] == "AAPL"


class TestGetPriceForDate:
    """Tests for get_price_for_date method."""

    @pytest.fixture
    def provider(self):
        return EODHDProvider(api_key="test-api-key")

    def test_get_price_for_date_exact(self, provider, sample_eodhd_price_data):
        """Test fetching price for exact date."""
        with patch("requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = sample_eodhd_price_data["AAPL"]
            mock_get.return_value = mock_response

            result = provider.get_price_for_date(["AAPL"], date(2024, 1, 3))

            assert "AAPL" in result
            assert isinstance(result["AAPL"], PriceData)
            assert result["AAPL"].symbol == "AAPL"

    def test_get_price_for_date_weekend(self, provider, sample_eodhd_price_data):
        """Test fetching price for weekend returns most recent."""
        with patch("requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            # Return Friday's data when asking for Saturday
            mock_response.json.return_value = sample_eodhd_price_data["AAPL"]
            mock_get.return_value = mock_response

            # Saturday - should get Friday's price
            result = provider.get_price_for_date(["AAPL"], date(2024, 1, 6))

            assert "AAPL" in result
            # Should return most recent available date
            assert result["AAPL"].date <= date(2024, 1, 6)


class TestGetConstituents:
    """Tests for get_constituents method."""

    @pytest.fixture
    def provider(self):
        return EODHDProvider(api_key="test-api-key")

    def test_get_constituents_current(
        self, provider, sample_eodhd_constituents_data
    ):
        """Test fetching current constituents."""
        with patch("requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = sample_eodhd_constituents_data
            mock_get.return_value = mock_response

            constituents = provider.get_constituents()

            assert len(constituents) > 0
            assert all(isinstance(c, BenchmarkConstituent) for c in constituents)

            # Check first constituent
            assert constituents[0].symbol is not None
            assert constituents[0].weight > Decimal("0")

    def test_get_constituents_historical(
        self, provider, sample_eodhd_constituents_data
    ):
        """Test fetching historical constituents."""
        with patch("requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = sample_eodhd_constituents_data
            mock_get.return_value = mock_response

            historical_date = date(2023, 6, 15)
            constituents = provider.get_constituents(as_of_date=historical_date)

            assert len(constituents) > 0

            # Verify API was called with historical parameters
            call_args = mock_get.call_args
            assert "historical" in str(call_args)

    def test_get_constituents_weights_normalized(
        self, provider, sample_eodhd_constituents_data
    ):
        """Test that constituent weights are normalized to sum to ~1."""
        with patch("requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = sample_eodhd_constituents_data
            mock_get.return_value = mock_response

            constituents = provider.get_constituents()

            total_weight = sum(c.weight for c in constituents)
            # Should be close to 1 (allowing for rounding)
            assert abs(total_weight - Decimal("1")) < Decimal("0.01")


class TestGetTradingDays:
    """Tests for get_trading_days method."""

    @pytest.fixture
    def provider(self):
        return EODHDProvider(api_key="test-api-key")

    def test_get_trading_days(self, provider, sample_eodhd_price_data):
        """Test fetching trading days."""
        with patch("requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = sample_eodhd_price_data["SPY"]
            mock_get.return_value = mock_response

            trading_days = provider.get_trading_days(
                date(2024, 1, 2),
                date(2024, 1, 10),
            )

            assert len(trading_days) > 0
            assert all(isinstance(d, date) for d in trading_days)
            # Should be sorted
            assert trading_days == sorted(trading_days)

    def test_get_trading_days_excludes_weekends(self, provider):
        """Test that weekends are excluded from trading days."""
        # Return data only for weekdays
        weekday_data = [
            {"date": "2024-01-02", "adjusted_close": 100},  # Tuesday
            {"date": "2024-01-03", "adjusted_close": 101},  # Wednesday
            {"date": "2024-01-04", "adjusted_close": 102},  # Thursday
            {"date": "2024-01-05", "adjusted_close": 103},  # Friday
            {"date": "2024-01-08", "adjusted_close": 104},  # Monday
        ]

        with patch("requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = weekday_data
            mock_get.return_value = mock_response

            trading_days = provider.get_trading_days(
                date(2024, 1, 2),
                date(2024, 1, 8),
            )

            # Should have 5 days (no weekend)
            assert len(trading_days) == 5

            # None should be weekend
            for d in trading_days:
                assert d.weekday() < 5  # 0-4 = Mon-Fri


class TestRateLimiting:
    """Tests for rate limiting and retry behavior."""

    @pytest.fixture
    def provider(self):
        return EODHDProvider(api_key="test-api-key", max_retries=3, retry_delay=0.01)

    def test_rate_limit_retry(self, provider):
        """Test that 429 responses trigger retry."""
        with patch("requests.get") as mock_get:
            # First call returns 429, second succeeds
            rate_limit_response = MagicMock()
            rate_limit_response.status_code = 429

            success_response = MagicMock()
            success_response.status_code = 200
            success_response.json.return_value = [
                {"date": "2024-01-02", "adjusted_close": 100}
            ]

            mock_get.side_effect = [rate_limit_response, success_response]

            with patch("time.sleep"):  # Speed up test
                df = provider.get_prices(["AAPL"], date(2024, 1, 2), date(2024, 1, 5))

            # Should have retried and succeeded
            assert mock_get.call_count == 2
            assert len(df) > 0

    def test_rate_limit_max_retries(self, provider):
        """Test that max retries are respected."""
        with patch("requests.get") as mock_get:
            # All calls return 429
            rate_limit_response = MagicMock()
            rate_limit_response.status_code = 429
            mock_get.return_value = rate_limit_response

            with patch("time.sleep"):  # Speed up test
                with pytest.raises(DataProviderError) as exc_info:
                    provider.get_prices(["AAPL"], date(2024, 1, 2), date(2024, 1, 5))

            assert "rate limit" in str(exc_info.value).lower()

    def test_request_timeout_retry(self, provider):
        """Test that timeouts trigger retry."""
        with patch("requests.get") as mock_get:
            # First call times out, second succeeds
            mock_get.side_effect = [
                requests.exceptions.Timeout("Connection timed out"),
                MagicMock(
                    status_code=200,
                    json=lambda: [{"date": "2024-01-02", "adjusted_close": 100}]
                ),
            ]

            with patch("time.sleep"):
                df = provider.get_prices(["AAPL"], date(2024, 1, 2), date(2024, 1, 5))

            assert mock_get.call_count == 2
            assert len(df) > 0


class TestErrorHandling:
    """Tests for error handling."""

    @pytest.fixture
    def provider(self):
        return EODHDProvider(api_key="test-api-key", max_retries=1, retry_delay=0.01)

    def test_invalid_json_response(self, provider):
        """Test handling of invalid JSON response."""
        with patch("requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.side_effect = ValueError("Invalid JSON")
            mock_get.return_value = mock_response

            with pytest.raises(DataProviderError) as exc_info:
                provider.get_prices(["AAPL"], date(2024, 1, 2), date(2024, 1, 5))

            assert "Invalid JSON" in str(exc_info.value)

    def test_api_error_response(self, provider):
        """Test handling of API error response."""
        with patch("requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"error": "Invalid API key"}
            mock_get.return_value = mock_response

            # Should return empty df for single symbol error
            df = provider.get_prices(["AAPL"], date(2024, 1, 2), date(2024, 1, 5))
            assert df.empty


class TestHelperFunction:
    """Tests for get_eodhd_provider helper function."""

    def test_get_eodhd_provider_with_cache(self):
        """Test helper function with caching enabled."""
        with patch.dict(os.environ, {"EODHD_API_KEY": "test-key"}):
            provider = get_eodhd_provider(use_cache=True, api_key="test-key")

            # Should be wrapped in CachedDataProvider
            assert "Cached" in provider.name

    def test_get_eodhd_provider_without_cache(self):
        """Test helper function with caching disabled."""
        with patch.dict(os.environ, {"EODHD_API_KEY": "test-key"}):
            provider = get_eodhd_provider(use_cache=False, api_key="test-key")

            # Should be bare EODHDProvider
            assert provider.name == "EODHD"


# =============================================================================
# Integration Tests (require valid API key)
# =============================================================================


def has_eodhd_api_key() -> bool:
    """Check if EODHD API key is available."""
    try:
        from di_pilot.config import load_api_keys
        keys = load_api_keys()
        return bool(keys.get("eodhd_api_key"))
    except Exception:
        return False


@pytest.mark.integration
@pytest.mark.skipif(
    not has_eodhd_api_key(),
    reason="EODHD_API_KEY not set"
)
class TestEODHDIntegration:
    """Integration tests that require a valid EODHD API key."""

    @pytest.fixture
    def provider(self):
        """Create provider using real API key."""
        return EODHDProvider()

    def test_fetch_real_prices(self, provider):
        """Test fetching real price data from EODHD."""
        # Use a recent date range
        end_date = date.today() - timedelta(days=1)
        start_date = end_date - timedelta(days=7)

        df = provider.get_prices(["AAPL", "MSFT"], start_date, end_date)

        assert len(df) > 0
        assert "AAPL" in df["symbol"].values
        assert "MSFT" in df["symbol"].values
        assert all(df["close"] > 0)

    def test_fetch_real_constituents(self, provider):
        """Test fetching real S&P 500 constituents."""
        constituents = provider.get_constituents()

        assert len(constituents) > 400  # S&P 500 should have ~500
        assert len(constituents) <= 510  # Allow some buffer

        # Check some known large caps are present
        symbols = [c.symbol for c in constituents]
        assert "AAPL" in symbols
        assert "MSFT" in symbols

        # Weights should sum to approximately 1
        total_weight = sum(c.weight for c in constituents)
        assert Decimal("0.99") < total_weight < Decimal("1.01")

    def test_historical_constituents_available(self, provider):
        """Test that historical constituents are available."""
        # Use a date from 6 months ago
        historical_date = date.today() - timedelta(days=180)

        constituents = provider.get_constituents(as_of_date=historical_date)

        # Should have data (may be same as current if historical not available)
        assert len(constituents) > 0

        # All should have the historical date
        for c in constituents:
            assert c.as_of_date == historical_date

    def test_fetch_real_trading_days(self, provider):
        """Test fetching real trading days."""
        end_date = date.today() - timedelta(days=1)
        start_date = end_date - timedelta(days=30)

        trading_days = provider.get_trading_days(start_date, end_date)

        # Should have approximately 20-22 trading days in a month
        assert len(trading_days) >= 15
        assert len(trading_days) <= 25

        # All should be weekdays
        for d in trading_days:
            assert d.weekday() < 5

    def test_validate_symbols(self, provider):
        """Test symbol validation with real data."""
        valid, invalid = provider.validate_symbols(
            ["AAPL", "MSFT", "INVALID_SYMBOL_12345"]
        )

        assert "AAPL" in valid
        assert "MSFT" in valid
        assert "INVALID_SYMBOL_12345" in invalid
