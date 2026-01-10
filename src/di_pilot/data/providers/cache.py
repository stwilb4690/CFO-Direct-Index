"""
Caching layer for data providers.

Provides file-based caching for market data to ensure:
- Reproducibility across simulation runs
- Reduced API calls to data providers
- Faster subsequent runs
"""

import hashlib
import json
import os
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from di_pilot.models import BenchmarkConstituent, PriceData
from di_pilot.data.providers.base import DataProvider, DataProviderError


class FileCache:
    """
    File-based cache for market data.

    Stores data in CSV/JSON files organized by data type and date range.
    """

    def __init__(self, cache_dir: str | Path = "data/cache"):
        """
        Initialize the file cache.

        Args:
            cache_dir: Directory to store cached data
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Subdirectories for different data types
        self.prices_dir = self.cache_dir / "prices"
        self.constituents_dir = self.cache_dir / "constituents"
        self.prices_dir.mkdir(exist_ok=True)
        self.constituents_dir.mkdir(exist_ok=True)

    def _get_cache_key(self, symbols: list[str], start_date: date, end_date: date) -> str:
        """Generate a cache key for a price request."""
        # Sort symbols for consistent hashing
        sorted_symbols = sorted(set(s.upper() for s in symbols))
        key_str = f"{','.join(sorted_symbols)}_{start_date}_{end_date}"
        return hashlib.md5(key_str.encode()).hexdigest()[:12]

    def get_prices(
        self,
        symbols: list[str],
        start_date: date,
        end_date: date,
    ) -> Optional[pd.DataFrame]:
        """
        Get cached price data if available.

        Args:
            symbols: List of ticker symbols
            start_date: Start date
            end_date: End date

        Returns:
            Cached DataFrame or None if not cached
        """
        cache_key = self._get_cache_key(symbols, start_date, end_date)
        cache_file = self.prices_dir / f"prices_{cache_key}.parquet"

        if cache_file.exists():
            try:
                df = pd.read_parquet(cache_file)
                # Verify all symbols are present
                cached_symbols = set(df["symbol"].unique())
                requested_symbols = set(s.upper() for s in symbols)
                if requested_symbols.issubset(cached_symbols):
                    # Validate date coverage - reject cache if it doesn't cover requested range
                    if not df.empty and "date" in df.columns:
                        cached_dates = df["date"].unique()
                        cached_min = min(cached_dates)
                        cached_max = max(cached_dates)

                        # Convert to date objects if needed for comparison
                        if hasattr(cached_min, 'date'):
                            cached_min = cached_min.date()
                        if hasattr(cached_max, 'date'):
                            cached_max = cached_max.date()

                        # Allow 7-day tolerance for weekends/holidays at boundaries
                        start_gap = (cached_min - start_date).days if isinstance(cached_min, date) else 0
                        end_gap = (end_date - cached_max).days if isinstance(cached_max, date) else 0

                        if start_gap > 7 or end_gap > 7:
                            # Cache doesn't cover requested range - treat as cache miss
                            return None

                    return df[df["symbol"].isin(requested_symbols)]
            except Exception:
                # Cache corrupted, will re-fetch
                pass

        return None

    def save_prices(
        self,
        df: pd.DataFrame,
        symbols: list[str],
        start_date: date,
        end_date: date,
    ) -> None:
        """
        Save price data to cache.

        Args:
            df: DataFrame with price data
            symbols: List of symbols (for cache key)
            start_date: Start date
            end_date: End date
        """
        cache_key = self._get_cache_key(symbols, start_date, end_date)
        cache_file = self.prices_dir / f"prices_{cache_key}.parquet"

        try:
            df.to_parquet(cache_file, index=False)
        except Exception as e:
            # Don't fail on cache write errors
            pass

    def get_constituents(self, as_of_date: Optional[date] = None) -> Optional[list[dict]]:
        """
        Get cached constituent data if available.

        Args:
            as_of_date: Date for constituents (None = current)

        Returns:
            List of constituent dictionaries or None if not cached
        """
        date_str = as_of_date.isoformat() if as_of_date else "current"
        cache_file = self.constituents_dir / f"constituents_{date_str}.json"

        if cache_file.exists():
            try:
                # Check if cache is recent (< 24 hours for current)
                if as_of_date is None:
                    mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
                    if (datetime.now() - mtime).total_seconds() > 86400:
                        return None

                with open(cache_file, "r") as f:
                    return json.load(f)
            except Exception:
                pass

        return None

    def save_constituents(
        self,
        constituents: list[dict],
        as_of_date: Optional[date] = None,
    ) -> None:
        """
        Save constituent data to cache.

        Args:
            constituents: List of constituent dictionaries
            as_of_date: Date for constituents (None = current)
        """
        date_str = as_of_date.isoformat() if as_of_date else "current"
        cache_file = self.constituents_dir / f"constituents_{date_str}.json"

        try:
            with open(cache_file, "w") as f:
                json.dump(constituents, f, indent=2, default=str)
        except Exception:
            pass

    def clear(self) -> None:
        """Clear all cached data."""
        import shutil

        for subdir in [self.prices_dir, self.constituents_dir]:
            if subdir.exists():
                shutil.rmtree(subdir)
                subdir.mkdir(exist_ok=True)


class CachedDataProvider(DataProvider):
    """
    Wrapper that adds caching to any DataProvider.

    Checks cache before calling underlying provider,
    and saves results to cache after fetching.
    """

    def __init__(
        self,
        provider: DataProvider,
        cache: Optional[FileCache] = None,
    ):
        """
        Initialize cached provider.

        Args:
            provider: Underlying data provider
            cache: File cache instance (creates default if None)
        """
        self._provider = provider
        self._cache = cache or FileCache()

    @property
    def name(self) -> str:
        return f"Cached({self._provider.name})"

    @property
    def supports_historical_constituents(self) -> bool:
        return self._provider.supports_historical_constituents

    def get_prices(
        self,
        symbols: list[str],
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """
        Get prices with caching.

        Checks cache first, falls back to provider if not cached.
        """
        # Try cache first
        cached = self._cache.get_prices(symbols, start_date, end_date)
        if cached is not None:
            return cached

        # Fetch from provider
        df = self._provider.get_prices(symbols, start_date, end_date)

        # Save to cache
        self._cache.save_prices(df, symbols, start_date, end_date)

        return df

    def get_price_for_date(
        self,
        symbols: list[str],
        as_of_date: date,
    ) -> dict[str, PriceData]:
        """Get prices for a specific date."""
        # Use the prices cache with single-day range
        df = self.get_prices(symbols, as_of_date, as_of_date)

        result = {}
        for _, row in df.iterrows():
            symbol = str(row["symbol"])
            result[symbol] = PriceData(
                symbol=symbol,
                date=as_of_date,
                close=Decimal(str(row["close"])),
            )

        return result

    def get_constituents(
        self,
        as_of_date: Optional[date] = None,
    ) -> list[BenchmarkConstituent]:
        """Get constituents with caching."""
        # Try cache first
        cached = self._cache.get_constituents(as_of_date)
        if cached is not None:
            return [
                BenchmarkConstituent(
                    symbol=c["symbol"],
                    weight=Decimal(str(c["weight"])),
                    as_of_date=date.fromisoformat(c["as_of_date"]) if c.get("as_of_date") else None,
                )
                for c in cached
            ]

        # Fetch from provider
        constituents = self._provider.get_constituents(as_of_date)

        # Save to cache
        cache_data = [
            {
                "symbol": c.symbol,
                "weight": str(c.weight),
                "as_of_date": c.as_of_date.isoformat() if c.as_of_date else None,
            }
            for c in constituents
        ]
        self._cache.save_constituents(cache_data, as_of_date)

        return constituents

    def get_trading_days(
        self,
        start_date: date,
        end_date: date,
    ) -> list[date]:
        """Get trading days (delegated to provider)."""
        return self._provider.get_trading_days(start_date, end_date)

    def validate_symbols(self, symbols: list[str]) -> tuple[list[str], list[str]]:
        """Validate symbols (delegated to provider)."""
        return self._provider.validate_symbols(symbols)
