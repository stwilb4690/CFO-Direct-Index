"""
Data providers for market data and S&P 500 constituents.

Provides a pluggable interface for fetching historical prices and
benchmark constituent data with built-in caching for reproducibility.
"""

from di_pilot.data.providers.base import DataProvider, DataProviderError
from di_pilot.data.providers.cache import CachedDataProvider, FileCache
from di_pilot.data.providers.eodhd_provider import EODHDProvider, get_eodhd_provider

__all__ = [
    "DataProvider",
    "DataProviderError",
    "CachedDataProvider",
    "FileCache",
    "EODHDProvider",
    "get_eodhd_provider",
]
