"""
S&P Global Indices data provider via EODHD Marketplace.

Uses the UnicornBay S&P Global data package for accurate constituent weights.
https://eodhd.com/marketplace/unicornbay/spglobal/docs

NOTE: This endpoint costs 10 API calls per request. Caching is critical.
"""

import json
import time
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Optional

import requests

from di_pilot.config import load_api_keys
from di_pilot.models import BenchmarkConstituent
from di_pilot.data.providers.base import DataProvider, DataProviderError


class SPGlobalProvider:
    """
    S&P Global constituent data provider via EODHD Marketplace.
    
    Provides accurate S&P 500 constituent weights from the S&P Global dataset.
    Caches responses aggressively to minimize API costs (10 calls per request).
    """
    
    BASE_URL = "https://eodhd.com/api/mp/unicornbay/spglobal"
    CACHE_FILE = "spglobal_constituents_cache.json"
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        cache_dir: str = "data/cache",
        cache_ttl_hours: int = 24,
    ):
        """
        Initialize S&P Global provider.
        
        Args:
            api_key: EODHD API key (defaults to loading from config)
            cache_dir: Directory for cache files
            cache_ttl_hours: Cache time-to-live in hours (default 24)
        """
        if api_key:
            self._api_key = api_key
        else:
            api_keys = load_api_keys()
            self._api_key = api_keys.get("eodhd_api_key")
        
        if not self._api_key:
            raise DataProviderError(
                "EODHD API key is not configured. Required for S&P Global data."
            )
        
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_ttl_hours = cache_ttl_hours
        self._cached_data = None
        self._cache_timestamp = None
    
    @property
    def name(self) -> str:
        return "SPGlobal"
    
    def _get_cache_path(self) -> Path:
        """Get path to cache file."""
        return self._cache_dir / self.CACHE_FILE
    
    def _load_cache(self) -> Optional[dict]:
        """Load cached data if valid."""
        cache_path = self._get_cache_path()
        if not cache_path.exists():
            return None
        
        try:
            with open(cache_path, "r") as f:
                cached = json.load(f)
            
            # Check TTL
            cached_time = cached.get("timestamp", 0)
            age_hours = (time.time() - cached_time) / 3600
            
            if age_hours > self._cache_ttl_hours:
                return None
            
            return cached.get("data")
        except (json.JSONDecodeError, IOError):
            return None
    
    def _save_cache(self, data: dict) -> None:
        """Save data to cache."""
        cache_path = self._get_cache_path()
        try:
            with open(cache_path, "w") as f:
                json.dump({
                    "timestamp": time.time(),
                    "data": data,
                }, f)
        except IOError:
            pass  # Cache write failure is not critical
    
    def _fetch_index_data(self, symbol: str = "GSPC.INDX") -> dict:
        """
        Fetch index data from S&P Global API.
        
        Args:
            symbol: Index symbol (default: GSPC.INDX for S&P 500)
            
        Returns:
            API response dict with Components and HistoricalTickerComponents
        """
        # Try cache first
        cached = self._load_cache()
        if cached:
            return cached
        
        # Fetch from API
        url = f"{self.BASE_URL}/comp/{symbol}"
        params = {
            "api_token": self._api_key,
            "fmt": "json",
        }
        
        try:
            response = requests.get(url, params=params, timeout=60)
            response.raise_for_status()
            data = response.json()
            
            # Cache the response
            self._save_cache(data)
            
            return data
        except requests.exceptions.RequestException as e:
            # Try to use stale cache as fallback
            cache_path = self._get_cache_path()
            if cache_path.exists():
                try:
                    with open(cache_path, "r") as f:
                        cached = json.load(f)
                    return cached.get("data", {})
                except:
                    pass
            raise DataProviderError(f"Failed to fetch S&P Global data: {e}")
    
    def get_constituents(
        self,
        as_of_date: Optional[date] = None,
    ) -> list[BenchmarkConstituent]:
        """
        Get S&P 500 constituents with weights.
        
        For current constituents, uses the 'Components' field.
        For historical dates, filters 'HistoricalTickerComponents' by date.
        
        Args:
            as_of_date: Date for constituents (default: current)
            
        Returns:
            List of BenchmarkConstituent objects
        """
        data = self._fetch_index_data()
        effective_date = as_of_date or date.today()
        
        # Use Components for current/recent data (has weights)
        components = data.get("Components", {})
        
        if not components:
            raise DataProviderError("No component data returned from S&P Global API")
        
        # Build constituent list from Components
        constituents = []
        total_weight = Decimal("0")
        
        for key, comp in components.items():
            symbol = comp.get("Code")
            weight = comp.get("Weight", 0)
            
            if not symbol:
                continue
            
            # Check if this component was active on as_of_date
            # by cross-referencing with HistoricalTickerComponents
            if as_of_date:
                if not self._was_constituent_on_date(data, symbol, as_of_date):
                    continue
            
            try:
                weight_decimal = Decimal(str(weight)) if weight else Decimal("0")
                # Weights may be percentages or decimals
                if weight_decimal > 1:
                    weight_decimal = weight_decimal / Decimal("100")
                
                constituents.append(
                    BenchmarkConstituent(
                        symbol=symbol.upper(),
                        weight=weight_decimal,
                        as_of_date=effective_date,
                    )
                )
                total_weight += weight_decimal
            except:
                continue
        
        # Normalize weights if needed
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
    
    def _was_constituent_on_date(
        self,
        data: dict,
        symbol: str,
        target_date: date,
    ) -> bool:
        """
        Check if a symbol was a constituent on a specific date.
        
        Uses HistoricalTickerComponents to determine membership.
        """
        historical = data.get("HistoricalTickerComponents", {})
        
        for key, comp in historical.items():
            if comp.get("Code", "").upper() != symbol.upper():
                continue
            
            start_str = comp.get("StartDate")
            end_str = comp.get("EndDate")
            
            try:
                start_date = date.fromisoformat(start_str) if start_str else date.min
                end_date = date.fromisoformat(end_str) if end_str else date.max
                
                if start_date <= target_date <= end_date:
                    return True
            except:
                continue
        
        # If no historical record found but it's in Components, 
        # assume it's currently active
        return True


def get_spglobal_provider(
    cache_dir: str = "data/cache",
    api_key: Optional[str] = None,
) -> SPGlobalProvider:
    """
    Get an S&P Global data provider instance.
    
    Args:
        cache_dir: Directory for cache files
        api_key: EODHD API key (optional, loads from config)
        
    Returns:
        SPGlobalProvider instance
    """
    return SPGlobalProvider(api_key=api_key, cache_dir=cache_dir)
