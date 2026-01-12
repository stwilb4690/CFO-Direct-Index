"""
Initialize the $10MM forward test portfolio with market-cap weights.

Uses SPGlobalProvider to get proper S&P 500 market-cap weighted constituents.

Usage:
    python scripts/initialize_forward_portfolio.py
"""

import sys
from pathlib import Path
from datetime import date
from decimal import Decimal

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from di_pilot.simulation.forward import ForwardTestRunner
from di_pilot.simulation.engine import SimulationEngine, SimulationConfig
from di_pilot.data.providers.eodhd_provider import EODHDProvider
from di_pilot.data.providers.spglobal_provider import SPGlobalProvider
from di_pilot.config import load_api_keys


def main():
    """Initialize the forward test portfolio with market-cap weights."""
    print("=" * 60)
    print("CFO Direct Index - Forward Test Portfolio Initialization")
    print("S&P 500 Market-Cap Weighted")
    print("=" * 60)
    
    # Configuration
    portfolio_id = "forward_10mm"
    initial_cash = Decimal("10000000")  # $10MM
    start_date = date(2026, 1, 9)  # Friday, January 9, 2026
    
    print(f"\nPortfolio ID: {portfolio_id}")
    print(f"Initial Cash: ${initial_cash:,.2f}")
    print(f"Start Date: {start_date}")
    
    # Check for existing portfolio
    runner = ForwardTestRunner()
    if runner.portfolio_exists(portfolio_id):
        print(f"\n⚠️  Portfolio '{portfolio_id}' already exists!")
        response = input("Delete and reinitialize? (yes/no): ").strip().lower()
        if response == "yes":
            runner.delete_portfolio(portfolio_id)
            print("Existing portfolio deleted.")
        else:
            print("Aborted.")
            return
    
    # Load API keys and initialize providers
    print("\nLoading API credentials...")
    load_api_keys()
    
    # Use SPGlobalProvider for market-cap weighted constituents
    print("Using SPGlobalProvider for market-cap weighted constituents...")
    spglobal = SPGlobalProvider()
    eodhd = EODHDProvider()
    
    # Create simulation config
    config = SimulationConfig(
        initial_cash=initial_cash,
        rebalance_freq="monthly",  # Monthly rebalancing
        cash_buffer_pct=Decimal("0.01"),  # 1% cash buffer
        tlh_loss_threshold=Decimal("0.03"),  # 3% loss threshold for TLH
    )
    
    # Get constituents with market-cap weights
    # This will fetch historical market caps and calculate proper weights
    print(f"\nFetching S&P 500 constituents with market-cap weights for {start_date}...")
    print("(This may take several minutes to fetch market cap data for all ~500 symbols)")
    
    try:
        constituents = spglobal.get_constituents(as_of_date=start_date)
    except Exception as e:
        print(f"SPGlobalProvider failed: {e}")
        print("Falling back to EODHD with current market-cap approximation...")
        # Alternative: use EODHD to get current constituents and market caps
        constituents = eodhd.get_constituents()
    
    print(f"Found {len(constituents)} constituents")
    
    # Check weights
    total_weight = sum(c.weight for c in constituents)
    print(f"Total weight: {float(total_weight):.6f}")
    
    if total_weight == 0:
        print("❌ Total weight is 0 - weight data not available")
        print("Please check EODHD API data or use alternative weight source")
        return
    
    # Get prices for start_date  
    symbols = [c.symbol for c in constituents]
    print(f"\nFetching prices for {len(symbols)} symbols on {start_date}...")
    prices = eodhd.get_price_for_date(symbols, start_date)
    
    if not prices:
        print(f"❌ No prices available for {start_date}")
        return
    
    print(f"Got prices for {len(prices)} symbols")
    
    # Initialize using the engine
    engine = SimulationEngine(config)
    
    try:
        state = engine.initialize_portfolio(
            portfolio_id=portfolio_id,
            start_date=start_date,
            constituents=constituents,
            prices=prices,
        )
        
        # Record initial snapshot
        state = engine.record_snapshot(state, prices, start_date)
        
        # Save using runner
        runner = ForwardTestRunner(config=config)
        runner._save_state(portfolio_id, state)
        
        # Append snapshot to CSV
        if state.snapshots:
            runner._append_snapshot(portfolio_id, state.snapshots[-1])
        
        print("\n✅ Portfolio initialized successfully!")
        print("-" * 50)
        print(f"Portfolio ID: {state.portfolio_id}")
        print(f"Date: {state.current_date}")
        print(f"Cash: ${float(state.cash):,.2f}")
        print(f"Number of positions: {len(set(lot.symbol for lot in state.lots))}")
        print(f"Number of tax lots: {len(state.lots)}")
        
        # Calculate total holdings value
        total_cost = sum(lot.shares * lot.cost_basis for lot in state.lots)
        print(f"Holdings Cost: ${float(total_cost):,.2f}")
        print(f"Total Value: ${float(state.cash + total_cost):,.2f}")
        
        # Show top 10 holdings by value
        holdings = {}
        for lot in state.lots:
            if lot.symbol not in holdings:
                holdings[lot.symbol] = Decimal("0")
            holdings[lot.symbol] += lot.shares * lot.cost_basis
        
        sorted_holdings = sorted(holdings.items(), key=lambda x: x[1], reverse=True)
        
        print("\nTop 10 Holdings (market-cap weighted):")
        print("-" * 55)
        for symbol, value in sorted_holdings[:10]:
            price = float(prices[symbol].close)
            weight = float(value) / float(initial_cash) * 100
            print(f"  {symbol:6s}  ${float(value):>12,.2f}  ({weight:>5.2f}%)  @ ${price:.2f}")
        
        # Save location
        portfolio_dir = runner._get_portfolio_dir(portfolio_id)
        print(f"\nData saved to: {portfolio_dir.absolute()}")
        
    except Exception as e:
        print(f"\n❌ Error initializing portfolio: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
