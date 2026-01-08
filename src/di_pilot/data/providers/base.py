"""
Abstract base class for data providers.

Defines the interface that all data providers must implement,
enabling pluggable data sources for backtesting and forward testing.
"""

from abc import ABC, abstractmethod
from datetime import date
from decimal import Decimal
from typing import Optional

import pandas as pd

from di_pilot.models import BenchmarkConstituent, PriceData


class DataProviderError(Exception):
    """Raised when a data provider encounters an error."""
    pass


class DataProvider(ABC):
    """
    Abstract base class for market data providers.

    Implementations must provide methods to fetch:
    - Historical price data for symbols
    - S&P 500 constituent information
    """

    @abstractmethod
    def get_prices(
        self,
        symbols: list[str],
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """
        Fetch historical adjusted close prices for symbols.

        Args:
            symbols: List of ticker symbols
            start_date: Start date (inclusive)
            end_date: End date (inclusive)

        Returns:
            DataFrame with columns: date, symbol, close
            - date: Trading date
            - symbol: Ticker symbol
            - close: Adjusted close price

        Raises:
            DataProviderError: If data cannot be fetched
        """
        pass

    @abstractmethod
    def get_price_for_date(
        self,
        symbols: list[str],
        as_of_date: date,
    ) -> dict[str, PriceData]:
        """
        Get prices for a specific date.

        Args:
            symbols: List of ticker symbols
            as_of_date: Date to get prices for

        Returns:
            Dictionary mapping symbol to PriceData

        Raises:
            DataProviderError: If data cannot be fetched
        """
        pass

    @abstractmethod
    def get_constituents(
        self,
        as_of_date: Optional[date] = None,
    ) -> list[BenchmarkConstituent]:
        """
        Get S&P 500 constituents with weights.

        Args:
            as_of_date: Date for constituents (default: current)

        Returns:
            List of BenchmarkConstituent objects

        Raises:
            DataProviderError: If data cannot be fetched

        Note:
            Historical constituent data requires Mode 2 (v0.3).
            Default implementation returns current constituents.
        """
        pass

    @abstractmethod
    def get_trading_days(
        self,
        start_date: date,
        end_date: date,
    ) -> list[date]:
        """
        Get list of trading days in date range.

        Args:
            start_date: Start date (inclusive)
            end_date: End date (inclusive)

        Returns:
            Sorted list of trading dates
        """
        pass

    def validate_symbols(self, symbols: list[str]) -> tuple[list[str], list[str]]:
        """
        Validate that symbols have available data.

        Args:
            symbols: List of ticker symbols to validate

        Returns:
            Tuple of (valid_symbols, invalid_symbols)
        """
        # Default implementation - subclasses may override
        return symbols, []

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this data provider."""
        pass

    @property
    def supports_historical_constituents(self) -> bool:
        """Whether this provider supports historical constituent data."""
        return False
