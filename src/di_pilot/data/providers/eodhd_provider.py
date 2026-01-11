"""
EODHD data provider implementation.

Uses the EODHD API (https://eodhd.com/api/) to fetch historical price data
and S&P 500 constituent information with historical support.
"""

import decimal
import time
from datetime import date, timedelta
from decimal import Decimal
from typing import Optional

import pandas as pd
import requests

from di_pilot.config import load_api_keys
from di_pilot.models import BenchmarkConstituent, PriceData
from di_pilot.data.providers.base import DataProvider, DataProviderError


class EODHDProvider(DataProvider):
    """
    Data provider using EODHD API for prices and constituents.

    Features:
    - Fetches adjusted close prices via EODHD end-of-day API
    - Batches symbol requests to avoid rate limiting
    - Supports historical S&P 500 constituents (Mode 2)
    - Requires EODHD_API_KEY environment variable
    """

    # EODHD API base URLs
    BASE_URL = "https://eodhd.com/api"
    EOD_ENDPOINT = "{base}/eod/{symbol}.US"
    FUNDAMENTALS_ENDPOINT = "{base}/fundamentals/{symbol}"

    # Batch size for price requests (to avoid rate limiting)
    BATCH_SIZE = 50

    # Delay between batches (seconds)
    BATCH_DELAY = 1.0

    # Rate limit retry settings
    RATE_LIMIT_RETRIES = 3
    RATE_LIMIT_DELAY = 5.0  # seconds

    def __init__(
        self,
        api_key: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ):
        """
        Initialize EODHD provider.

        Args:
            api_key: EODHD API key (defaults to loading from config sources)
            max_retries: Maximum retries for failed requests
            retry_delay: Delay between retries (seconds)

        Raises:
            DataProviderError: If API key is not provided or found in config

        Note:
            API key is loaded from (in priority order):
            1. api_key parameter
            2. EODHD_API_KEY environment variable
            3. .env file in project root
            4. config/api_keys.yaml
        """
        if api_key:
            self._api_key = api_key
        else:
            # Load from configuration sources
            api_keys = load_api_keys()
            self._api_key = api_keys.get("eodhd_api_key")

        if not self._api_key:
            raise DataProviderError(
                "EODHD API key is not configured. Please set it using one of:\n"
                "  1. Pass api_key parameter to EODHDProvider\n"
                "  2. Environment variable: export EODHD_API_KEY=your-key\n"
                "  3. .env file: EODHD_API_KEY=your-key\n"
                "  4. config/api_keys.yaml: eodhd_api_key: your-key\n"
                "\n"
                "Get your API key at: https://eodhd.com/"
            )

        self._max_retries = max_retries
        self._retry_delay = retry_delay

    @property
    def name(self) -> str:
        return "EODHD"

    @property
    def supports_historical_constituents(self) -> bool:
        return True  # EODHD provides historical constituent data

    def get_prices(
        self,
        symbols: list[str],
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """
        Fetch historical adjusted close prices.

        Uses endpoint: https://eodhd.com/api/eod/{symbol}.US?from={start}&to={end}&api_token={key}&fmt=json

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
            batch = symbols[i : i + self.BATCH_SIZE]
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
        """
        Fetch prices for a batch of symbols.

        EODHD requires individual requests per symbol, so we iterate.
        """
        records = []

        for symbol in symbols:
            try:
                symbol_data = self._fetch_symbol_prices(symbol, start_date, end_date)
                records.extend(symbol_data)
            except DataProviderError:
                # Log and continue with other symbols
                continue

        return pd.DataFrame(records, columns=["date", "symbol", "close"])

    def _fetch_symbol_prices(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
    ) -> list[tuple]:
        """Fetch prices for a single symbol from EODHD API."""
        url = f"{self.BASE_URL}/eod/{symbol}.US"
        params = {
            "from": start_date.isoformat(),
            "to": end_date.isoformat(),
            "api_token": self._api_key,
            "fmt": "json",
        }

        response = self._make_request(url, params)
        records = []

        for item in response:
            try:
                # Use adjusted_close if available, otherwise close
                close_price = item.get("adjusted_close") or item.get("close")
                if close_price is not None:
                    records.append((
                        date.fromisoformat(item["date"]),
                        symbol,
                        float(close_price),
                    ))
            except (KeyError, ValueError, TypeError):
                continue

        return records

    def _make_request(
        self,
        url: str,
        params: dict,
        expect_list: bool = True,
    ) -> list | dict:
        """
        Make an HTTP request to EODHD API with retry logic.

        Args:
            url: API endpoint URL
            params: Query parameters
            expect_list: Whether to expect a list response

        Returns:
            Parsed JSON response

        Raises:
            DataProviderError: On request failure
        """
        last_error = None

        for attempt in range(self._max_retries):
            try:
                response = requests.get(url, params=params, timeout=30)

                # Handle rate limiting
                if response.status_code == 429:
                    if attempt < self.RATE_LIMIT_RETRIES - 1:
                        time.sleep(self.RATE_LIMIT_DELAY * (attempt + 1))
                        continue
                    raise DataProviderError(
                        f"EODHD API rate limit exceeded after {self.RATE_LIMIT_RETRIES} retries"
                    )

                response.raise_for_status()

                # Parse JSON response
                data = response.json()

                # Validate response format
                if expect_list and not isinstance(data, list):
                    # Some endpoints return error objects
                    if isinstance(data, dict) and "error" in data:
                        raise DataProviderError(f"EODHD API error: {data['error']}")
                    # Empty response is okay
                    return []

                return data

            except requests.exceptions.Timeout:
                last_error = "Request timeout"
            except requests.exceptions.RequestException as e:
                last_error = str(e)
            except ValueError as e:
                raise DataProviderError(f"Invalid JSON response from EODHD API: {e}")

            if attempt < self._max_retries - 1:
                time.sleep(self._retry_delay * (attempt + 1))

        raise DataProviderError(
            f"Failed to fetch data from EODHD after {self._max_retries} attempts: {last_error}"
        )

    def get_price_for_date(
        self,
        symbols: list[str],
        as_of_date: date,
    ) -> dict[str, PriceData]:
        """
        Get prices for a specific date.

        Fetches prices with 7-day lookback to handle weekends/holidays.
        Returns most recent price if exact date not available.

        Args:
            symbols: List of ticker symbols
            as_of_date: Date to get prices for

        Returns:
            Dictionary mapping symbol to PriceData
        """
        # Fetch 7 days before to handle weekends/holidays
        start_date = as_of_date - timedelta(days=7)
        df = self.get_prices(symbols, start_date, as_of_date)

        if df.empty:
            return {}

        # Get the most recent price for each symbol
        result = {}
        for symbol in symbols:
            symbol_upper = symbol.upper()
            symbol_df = df[df["symbol"] == symbol_upper]
            if not symbol_df.empty:
                # Get the most recent row
                latest = symbol_df.sort_values("date").iloc[-1]
                result[symbol_upper] = PriceData(
                    symbol=symbol_upper,
                    date=latest["date"],
                    close=Decimal(str(latest["close"])),
                )

        return result

    def get_constituents(
        self,
        as_of_date: Optional[date] = None,
    ) -> list[BenchmarkConstituent]:
        """
        Get S&P 500 constituents with weights from EODHD S&P Global Marketplace.

        Uses the premium S&P Global indices dataset for accurate constituent weights:
        https://eodhd.com/api/mp/unicornbay/spglobal/comp/GSPC.INDX

        Args:
            as_of_date: Date for constituents (default: current)

        Returns:
            List of BenchmarkConstituent objects with symbol, weight, as_of_date
        """
        effective_date = as_of_date or date.today()
        
        # Use S&P Global Marketplace API for accurate constituent weights
        url = f"{self.BASE_URL}/mp/unicornbay/spglobal/comp/GSPC.INDX"

        params = {
            "api_token": self._api_key,
            "fmt": "json",
        }

        response = self._make_request(url, params, expect_list=False)

        if not isinstance(response, dict):
            raise DataProviderError("Invalid response format from S&P Global API")

        # Extract components from S&P Global response
        components = response.get("Components", {})
        historical = response.get("HistoricalTickerComponents", {})

        if not components:
            raise DataProviderError(
                f"Could not parse S&P 500 constituents from S&P Global for date {effective_date}"
            )

        # Build set of symbols that were constituents on the target date
        # using HistoricalTickerComponents
        active_on_date = set()
        if as_of_date and historical:
            for key, comp in historical.items():
                symbol = comp.get("Code", "").upper()
                start_str = comp.get("StartDate")
                end_str = comp.get("EndDate")
                
                try:
                    start = date.fromisoformat(start_str) if start_str else date.min
                    end = date.fromisoformat(end_str) if end_str else date.max
                    
                    if start <= as_of_date <= end:
                        active_on_date.add(symbol)
                except:
                    continue
        
        # Convert to BenchmarkConstituent objects
        constituents = []
        total_weight = Decimal("0")

        for key, component_info in components.items():
            if isinstance(component_info, dict):
                symbol = component_info.get("Code") or component_info.get("code")
                weight = component_info.get("Weight") or component_info.get("weight", 0)
            else:
                symbol = key
                weight = component_info

            if not symbol:
                continue
            
            clean_symbol = str(symbol).split(".")[0].upper()
            
            # For historical dates, check if this symbol was active
            if as_of_date and historical and active_on_date:
                if clean_symbol not in active_on_date:
                    continue

            try:
                weight_decimal = Decimal(str(weight)) if weight else Decimal("0")
                # Weights may be percentages or decimals
                if weight_decimal > 1:
                    weight_decimal = weight_decimal / Decimal("100")

                constituents.append(
                    BenchmarkConstituent(
                        symbol=clean_symbol,
                        weight=weight_decimal,
                        as_of_date=effective_date,
                    )
                )
                total_weight += weight_decimal
            except (ValueError, TypeError, decimal.InvalidOperation):
                continue

        # Normalize weights if they don't sum to 1
        if total_weight > 0 and abs(total_weight - Decimal("1")) > Decimal("0.01"):
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

    def _extract_components(
        self,
        response: dict,
        as_of_date: Optional[date],
    ) -> dict:
        """
        Extract components from EODHD fundamentals response.

        Tries multiple possible field names/structures.
        """
        # Try HistoricalComponents first for historical dates
        if as_of_date and "HistoricalComponents" in response:
            historical = response["HistoricalComponents"]
            if isinstance(historical, dict):
                # Find the closest date
                date_str = as_of_date.isoformat()
                if date_str in historical:
                    return historical[date_str]
                # Return most recent historical data
                if historical:
                    latest_date = max(historical.keys())
                    return historical[latest_date]

        # Try Components for current data
        if "Components" in response:
            return response["Components"]

        # Try General -> Components
        if "General" in response and "Components" in response["General"]:
            return response["General"]["Components"]

        # Try lowercase variants
        for key in ["components", "historicalComponents", "historicalcomponents"]:
            if key in response:
                return response[key]

        return {}

    def get_trading_days(
        self,
        start_date: date,
        end_date: date,
    ) -> list[date]:
        """
        Get list of trading days by checking SPY.US data.

        Args:
            start_date: Start date (inclusive)
            end_date: End date (inclusive)

        Returns:
            Sorted list of trading dates
        """
        # Use SPY as proxy for trading days
        df = self.get_prices(["SPY"], start_date, end_date)

        if df.empty:
            # Fallback: generate weekdays (excludes holidays)
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


def get_eodhd_provider(
    use_cache: bool = True,
    cache_dir: str = "data/cache",
    api_key: Optional[str] = None,
) -> DataProvider:
    """
    Get an EODHD data provider instance.

    Args:
        use_cache: Whether to wrap with caching layer
        cache_dir: Directory for cache files
        api_key: EODHD API key (defaults to EODHD_API_KEY env var)

    Returns:
        DataProvider instance (EODHD with optional caching)

    Raises:
        DataProviderError: If API key is not available
    """
    from di_pilot.data.providers.cache import CachedDataProvider, FileCache

    provider = EODHDProvider(api_key=api_key)

    if use_cache:
        cache = FileCache(cache_dir)
        return CachedDataProvider(provider, cache)

    return provider
