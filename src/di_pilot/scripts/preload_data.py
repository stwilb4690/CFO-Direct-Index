"""
Preload and cache S&P 500 price data for faster backtesting.

This script downloads and caches all S&P 500 constituent price data
for a specified date range. After running this once, future backtests
will use the cached data and run much faster.
"""

import sys
from datetime import date, timedelta
from pathlib import Path


def preload_sp500_data(
    start_date: date,
    end_date: date,
    progress_callback=None,
) -> dict:
    """
    Preload S&P 500 data into cache.
    
    Args:
        start_date: Start of date range
        end_date: End of date range
        progress_callback: Optional callback for progress updates
        
    Returns:
        Dictionary with download statistics
    """
    from di_pilot.data.providers import get_eodhd_provider
    
    provider = get_eodhd_provider(use_cache=True)
    
    # Get constituents
    if progress_callback:
        progress_callback("Fetching S&P 500 constituents...")
    
    constituents = provider.get_constituents(as_of_date=end_date)
    symbols = [c.symbol for c in constituents]
    
    if progress_callback:
        progress_callback(f"Found {len(symbols)} constituents")
    
    # Fetch price data (will be cached)
    if progress_callback:
        progress_callback(f"Downloading price data for {len(symbols)} symbols...")
        progress_callback(f"Date range: {start_date} to {end_date}")
        progress_callback("This may take 10-20 minutes on first run...")
    
    price_df = provider.get_prices(symbols, start_date, end_date)
    
    # Get trading days
    trading_days = provider.get_trading_days(start_date, end_date)
    
    stats = {
        "symbols": len(symbols),
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "trading_days": len(trading_days),
        "price_records": len(price_df) if hasattr(price_df, '__len__') else 0,
    }
    
    if progress_callback:
        progress_callback(f"✅ Downloaded {stats['price_records']:,} price records")
        progress_callback(f"Data is now cached for future use")
    
    return stats


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Preload S&P 500 price data for faster backtesting"
    )
    parser.add_argument(
        "--years",
        type=int,
        default=3,
        help="Number of years of data to preload (default: 3)"
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="End date (YYYY-MM-DD), defaults to yesterday"
    )
    
    args = parser.parse_args()
    
    # Calculate date range
    if args.end_date:
        end_date = date.fromisoformat(args.end_date)
    else:
        end_date = date.today() - timedelta(days=1)
    
    start_date = date(end_date.year - args.years, end_date.month, end_date.day)
    
    print("=" * 60)
    print("  S&P 500 Data Preloader")
    print("=" * 60)
    print(f"  Start Date: {start_date}")
    print(f"  End Date:   {end_date}")
    print(f"  Years:      {args.years}")
    print("=" * 60)
    print()
    
    def progress(msg):
        print(f"  {msg}")
    
    try:
        stats = preload_sp500_data(start_date, end_date, progress)
        
        print()
        print("=" * 60)
        print("  PRELOAD COMPLETE")
        print("=" * 60)
        print(f"  Symbols:       {stats['symbols']}")
        print(f"  Trading Days:  {stats['trading_days']}")
        print(f"  Price Records: {stats['price_records']:,}")
        print()
        print("  Data is now cached. Future backtests will be much faster!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
