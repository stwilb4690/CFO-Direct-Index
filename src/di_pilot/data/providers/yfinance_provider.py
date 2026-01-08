"""
Yahoo Finance data provider implementation.

Uses the yfinance library to fetch historical price data and
Wikipedia to fetch current S&P 500 constituents.
"""

import time
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup

from di_pilot.models import BenchmarkConstituent, PriceData
from di_pilot.data.providers.base import DataProvider, DataProviderError


class YFinanceProvider(DataProvider):
    """
    Data provider using Yahoo Finance for prices and Wikipedia for constituents.

    Features:
    - Fetches adjusted close prices (handles splits/dividends)
    - Batches symbol requests to avoid rate limiting
    - Scrapes S&P 500 constituents from Wikipedia
    - Supports current constituents only (Mode 1)
    """

    # Wikipedia URL for S&P 500 constituents
    WIKIPEDIA_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

    # Batch size for price requests (to avoid rate limiting)
    BATCH_SIZE = 50

    # Delay between batches (seconds)
    BATCH_DELAY = 1.0

    def __init__(self, max_retries: int = 3, retry_delay: float = 2.0):
        """
        Initialize Yahoo Finance provider.

        Args:
            max_retries: Maximum retries for failed requests
            retry_delay: Delay between retries (seconds)
        """
        self._max_retries = max_retries
        self._retry_delay = retry_delay

        # Import yfinance here to allow graceful failure if not installed
        try:
            import yfinance as yf
            self._yf = yf
        except ImportError:
            raise DataProviderError(
                "yfinance is required for YFinanceProvider. "
                "Install with: pip install yfinance"
            )

    @property
    def name(self) -> str:
        return "YahooFinance"

    @property
    def supports_historical_constituents(self) -> bool:
        return False  # Wikipedia only provides current constituents

    def get_prices(
        self,
        symbols: list[str],
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """
        Fetch historical adjusted close prices.

        Args:
            symbols: List of ticker symbols
            start_date: Start date (inclusive)
            end_date: End date (inclusive)

        Returns:
            DataFrame with columns: date, symbol, close
        """
        if not symbols:
            return pd.DataFrame(columns=["date", "symbol", "close"])

        # Normalize symbols
        symbols = [s.upper().strip() for s in symbols]
        symbols = list(set(symbols))  # Remove duplicates

        # Fetch in batches
        all_data = []
        for i in range(0, len(symbols), self.BATCH_SIZE):
            batch = symbols[i:i + self.BATCH_SIZE]
            batch_data = self._fetch_batch_prices(batch, start_date, end_date)
            all_data.append(batch_data)

            # Delay between batches to avoid rate limiting
            if i + self.BATCH_SIZE < len(symbols):
                time.sleep(self.BATCH_DELAY)

        if not all_data:
            return pd.DataFrame(columns=["date", "symbol", "close"])

        # Combine all batches
        df = pd.concat(all_data, ignore_index=True)

        # Sort by date and symbol
        df = df.sort_values(["date", "symbol"]).reset_index(drop=True)

        return df

    def _fetch_batch_prices(
        self,
        symbols: list[str],
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """Fetch prices for a batch of symbols."""
        for attempt in range(self._max_retries):
            try:
                # yfinance expects end_date to be exclusive, so add 1 day
                end_date_adj = end_date + timedelta(days=1)

                # Download data for all symbols at once
                tickers = self._yf.Tickers(" ".join(symbols))

                # Get historical data
                df = self._yf.download(
                    tickers=" ".join(symbols),
                    start=start_date.isoformat(),
                    end=end_date_adj.isoformat(),
                    progress=False,
                    auto_adjust=True,  # Use adjusted prices
                    threads=True,
                )

                if df.empty:
                    return pd.DataFrame(columns=["date", "symbol", "close"])

                # Handle single vs multiple symbols
                if len(symbols) == 1:
                    # Single symbol - df has simple columns
                    if "Close" not in df.columns:
                        return pd.DataFrame(columns=["date", "symbol", "close"])

                    result = pd.DataFrame({
                        "date": df.index.date,
                        "symbol": symbols[0],
                        "close": df["Close"].values,
                    })
                else:
                    # Multiple symbols - df has MultiIndex columns
                    records = []
                    for symbol in symbols:
                        try:
                            if ("Close", symbol) in df.columns:
                                closes = df[("Close", symbol)]
                            elif "Close" in df.columns and symbol in df["Close"].columns:
                                closes = df["Close"][symbol]
                            else:
                                continue

                            for idx, close in closes.items():
                                if pd.notna(close):
                                    records.append({
                                        "date": idx.date() if hasattr(idx, "date") else idx,
                                        "symbol": symbol,
                                        "close": float(close),
                                    })
                        except (KeyError, TypeError):
                            continue

                    result = pd.DataFrame(records)

                # Filter out NaN values
                result = result.dropna(subset=["close"])

                return result

            except Exception as e:
                if attempt < self._max_retries - 1:
                    time.sleep(self._retry_delay * (attempt + 1))
                else:
                    raise DataProviderError(
                        f"Failed to fetch prices after {self._max_retries} attempts: {e}"
                    )

        return pd.DataFrame(columns=["date", "symbol", "close"])

    def get_price_for_date(
        self,
        symbols: list[str],
        as_of_date: date,
    ) -> dict[str, PriceData]:
        """
        Get prices for a specific date.

        If the date is not a trading day, returns the most recent prior price.
        """
        # Fetch a few days before to handle weekends/holidays
        start_date = as_of_date - timedelta(days=7)
        df = self.get_prices(symbols, start_date, as_of_date)

        if df.empty:
            return {}

        # Get the most recent price for each symbol
        result = {}
        for symbol in symbols:
            symbol_df = df[df["symbol"] == symbol.upper()]
            if not symbol_df.empty:
                # Get the most recent row
                latest = symbol_df.sort_values("date").iloc[-1]
                result[symbol.upper()] = PriceData(
                    symbol=symbol.upper(),
                    date=latest["date"],
                    close=Decimal(str(latest["close"])),
                )

        return result

    def get_constituents(
        self,
        as_of_date: Optional[date] = None,
    ) -> list[BenchmarkConstituent]:
        """
        Get S&P 500 constituents from Wikipedia.

        Note: This only provides CURRENT constituents regardless of as_of_date.
        Historical constituent data is not available without a paid data source.

        Args:
            as_of_date: Ignored in Mode 1 (current constituents only)

        Returns:
            List of BenchmarkConstituent with estimated market-cap weights
        """
        for attempt in range(self._max_retries):
            try:
                # Fetch Wikipedia page
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                }
                response = requests.get(self.WIKIPEDIA_SP500_URL, headers=headers, timeout=30)
                response.raise_for_status()

                # Parse HTML
                soup = BeautifulSoup(response.content, "lxml")

                # Find the constituents table (first table with 'Symbol' column)
                tables = soup.find_all("table", {"class": "wikitable"})

                constituents = []
                for table in tables:
                    headers = [th.get_text(strip=True) for th in table.find_all("th")]
                    if "Symbol" in headers or "Ticker" in headers:
                        # Found the right table
                        symbol_idx = headers.index("Symbol") if "Symbol" in headers else headers.index("Ticker")

                        rows = table.find_all("tr")[1:]  # Skip header row
                        for row in rows:
                            cells = row.find_all(["td", "th"])
                            if len(cells) > symbol_idx:
                                # Get symbol - may be in <a> tag
                                symbol_cell = cells[symbol_idx]
                                symbol = symbol_cell.get_text(strip=True)

                                # Clean up symbol (remove any non-alphanumeric except . and -)
                                symbol = "".join(
                                    c for c in symbol if c.isalnum() or c in ".-"
                                ).upper()

                                if symbol and len(symbol) <= 5:
                                    constituents.append(symbol)

                        break

                if not constituents:
                    raise DataProviderError("Could not parse S&P 500 constituents from Wikipedia")

                # Estimate weights based on equal weighting for now
                # In practice, we'd want market cap data for accurate weights
                # For simplicity, use tiered weights based on typical S&P 500 distribution
                return self._estimate_weights(constituents, as_of_date or date.today())

            except requests.RequestException as e:
                if attempt < self._max_retries - 1:
                    time.sleep(self._retry_delay * (attempt + 1))
                else:
                    raise DataProviderError(f"Failed to fetch constituents: {e}")

        return []

    def _estimate_weights(
        self,
        symbols: list[str],
        as_of_date: date,
    ) -> list[BenchmarkConstituent]:
        """
        Estimate market-cap weights for constituents.

        Uses a simplified approach based on typical S&P 500 concentration:
        - Top 10: ~30% of index
        - Top 50: ~50% of index
        - Remaining: ~50% of index

        For more accurate weights, would need market cap data.
        """
        # Known large-cap tickers (approximate top holdings)
        mega_caps = {"AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "GOOG", "META", "BRK.B", "TSLA", "UNH"}
        large_caps = {
            "XOM", "JNJ", "JPM", "V", "PG", "MA", "HD", "CVX", "MRK", "ABBV",
            "LLY", "AVGO", "PEP", "KO", "COST", "TMO", "MCD", "CSCO", "WMT", "ACN",
            "ABT", "DHR", "NEE", "LIN", "TXN", "PM", "VZ", "CMCSA", "RTX", "ADBE",
            "NKE", "NFLX", "CRM", "AMD", "INTC", "QCOM", "HON", "UPS", "IBM", "CAT",
        }

        # Calculate weights
        total_symbols = len(symbols)
        constituents = []

        for symbol in symbols:
            if symbol in mega_caps:
                # Mega cap: ~3% each (top 10 = ~30%)
                weight = Decimal("0.03")
            elif symbol in large_caps:
                # Large cap: ~0.5% each
                weight = Decimal("0.005")
            else:
                # Remaining: distribute equally among rest (~50% / 450 = ~0.11%)
                remaining_count = total_symbols - len(mega_caps) - len(large_caps)
                if remaining_count > 0:
                    weight = Decimal("0.50") / Decimal(str(remaining_count))
                else:
                    weight = Decimal("0.001")

            constituents.append(
                BenchmarkConstituent(
                    symbol=symbol,
                    weight=weight,
                    as_of_date=as_of_date,
                )
            )

        # Normalize weights to sum to 1
        total_weight = sum(c.weight for c in constituents)
        if total_weight > 0:
            constituents = [
                BenchmarkConstituent(
                    symbol=c.symbol,
                    weight=c.weight / total_weight,
                    as_of_date=c.as_of_date,
                )
                for c in constituents
            ]

        # Sort by weight descending
        constituents.sort(key=lambda c: c.weight, reverse=True)

        return constituents

    def get_trading_days(
        self,
        start_date: date,
        end_date: date,
    ) -> list[date]:
        """
        Get list of trading days by checking SPY data.

        Args:
            start_date: Start date (inclusive)
            end_date: End date (inclusive)

        Returns:
            Sorted list of trading dates
        """
        # Use SPY as proxy for trading days
        df = self.get_prices(["SPY"], start_date, end_date)

        if df.empty:
            # Fallback: generate weekdays
            trading_days = []
            current = start_date
            while current <= end_date:
                if current.weekday() < 5:  # Monday = 0, Friday = 4
                    trading_days.append(current)
                current += timedelta(days=1)
            return trading_days

        return sorted(df["date"].unique().tolist())

    def validate_symbols(self, symbols: list[str]) -> tuple[list[str], list[str]]:
        """
        Validate symbols by checking if they have recent price data.

        Returns:
            Tuple of (valid_symbols, invalid_symbols)
        """
        end_date = date.today()
        start_date = end_date - timedelta(days=7)

        df = self.get_prices(symbols, start_date, end_date)

        valid = set(df["symbol"].unique())
        all_symbols = set(s.upper() for s in symbols)
        invalid = all_symbols - valid

        return list(valid), list(invalid)


def get_default_provider(use_cache: bool = True, cache_dir: str = "data/cache") -> DataProvider:
    """
    Get the default data provider instance.

    Args:
        use_cache: Whether to wrap with caching layer
        cache_dir: Directory for cache files

    Returns:
        DataProvider instance (YFinance with optional caching)
    """
    from di_pilot.data.providers.cache import CachedDataProvider, FileCache

    provider = YFinanceProvider()

    if use_cache:
        cache = FileCache(cache_dir)
        return CachedDataProvider(provider, cache)

    return provider
