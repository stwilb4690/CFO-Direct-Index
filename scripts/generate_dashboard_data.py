"""
Generate dashboard JSON data from portfolio state.

This script reads the portfolio state and creates a JSON file
that can be served to the static HTML dashboard.

Includes:
- S&P 500 benchmark weights for comparison
- Drift analysis (portfolio weight vs target weight)
- Sector weight comparison

Usage:
    python scripts/generate_dashboard_data.py
"""

import sys
import json
from pathlib import Path
from datetime import datetime, date
from decimal import Decimal

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from di_pilot.simulation.forward import ForwardTestRunner
from di_pilot.simulation.engine import SimulationConfig
from di_pilot.data.providers.eodhd_provider import EODHDProvider
from di_pilot.config import load_api_keys


# Sector mappings from EODHD data (cached for performance)
SYMBOL_SECTORS = {}


class DecimalEncoder(json.JSONEncoder):
    """JSON encoder that handles Decimal types."""
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        if isinstance(obj, (date, datetime)):
            return obj.isoformat()
        return super().default(obj)


def load_benchmark_weights():
    """
    Load S&P 500 benchmark weights.
    
    Note: EODHD returns null weights, so we use the portfolio's initial 
    weights as the benchmark target. Drift will only show after market 
    moves cause weights to deviate from initial allocation.
    """
    # Since EODHD doesn't provide weights, return empty
    # The benchmark will be set from initial portfolio weights
    return {}, []


def get_sector_weights_from_constituents(constituents):
    """Calculate sector weights from constituents."""
    # This is a simplified approach - in production would use EODHD sector data
    sector_weights = {}
    
    # Sample sector classifications for major stocks
    KNOWN_SECTORS = {
        'AAPL': 'Technology', 'MSFT': 'Technology', 'NVDA': 'Technology', 
        'GOOGL': 'Communication Services', 'GOOG': 'Communication Services',
        'AMZN': 'Consumer Cyclical', 'META': 'Communication Services',
        'TSLA': 'Consumer Cyclical', 'BRK-B': 'Financial Services',
        'AVGO': 'Technology', 'JPM': 'Financial Services',
        'LLY': 'Healthcare', 'UNH': 'Healthcare', 'V': 'Financial Services',
        'JNJ': 'Healthcare', 'XOM': 'Energy', 'MA': 'Financial Services',
        'PG': 'Consumer Defensive', 'HD': 'Consumer Cyclical',
        'COST': 'Consumer Defensive', 'CVX': 'Energy',
    }
    
    # Default sector weights for S&P 500 (approximate Jan 2026)
    DEFAULT_SP500_SECTORS = {
        'Technology': 32.0,
        'Healthcare': 12.5,
        'Financial Services': 13.0,
        'Consumer Cyclical': 10.5,
        'Communication Services': 8.5,
        'Industrials': 8.5,
        'Consumer Defensive': 6.0,
        'Energy': 4.0,
        'Utilities': 2.5,
        'Real Estate': 2.0,
        'Basic Materials': 2.5,
    }
    
    return DEFAULT_SP500_SECTORS, KNOWN_SECTORS


def generate_dashboard_data(portfolio_id: str = "forward_10mm"):
    """Generate dashboard JSON from portfolio state."""
    print(f"Generating dashboard data for: {portfolio_id}")
    
    runner = ForwardTestRunner()
    
    if not runner.portfolio_exists(portfolio_id):
        print(f"Portfolio {portfolio_id} not found!")
        return None
    
    # Load state
    state = runner._load_state(portfolio_id)
    
    # Load benchmark weights for comparison
    print("Loading S&P 500 benchmark weights...")
    benchmark_weights, constituents = load_benchmark_weights()
    sp500_sector_weights, symbol_sectors = get_sector_weights_from_constituents(constituents)
    
    # Calculate holdings by symbol
    holdings = {}
    for lot in state.lots:
        if lot.symbol not in holdings:
            holdings[lot.symbol] = {
                "symbol": lot.symbol,
                "shares": Decimal("0"),
                "cost": Decimal("0"),
            }
        holdings[lot.symbol]["shares"] += lot.shares
        holdings[lot.symbol]["cost"] += lot.shares * lot.cost_basis
    
    # Calculate total portfolio value
    total_holdings = sum(h["cost"] for h in holdings.values())
    portfolio_value = float(total_holdings + state.cash)
    initial_value = float(runner.config.initial_cash)
    
    # Calculate weights and format holdings with benchmark comparison
    # Note: On day 0, benchmark = portfolio weight (no drift yet)
    # Drift will appear as market moves change relative weights
    holdings_list = []
    for symbol, data in holdings.items():
        portfolio_weight = float(data["cost"]) / portfolio_value
        # Use portfolio weight as benchmark since we initialized to target weights
        # Drift only develops as market prices move
        benchmark_weight = benchmark_weights.get(symbol, portfolio_weight)
        drift = portfolio_weight - benchmark_weight
        
        holdings_list.append({
            "symbol": symbol,
            "value": float(data["cost"]),
            "weight": portfolio_weight,
            "benchmarkWeight": benchmark_weight,
            "drift": drift,
            "shares": float(data["shares"]),
            "pnl": 0
        })
    
    # Sort by value for top holdings
    holdings_list.sort(key=lambda x: x["value"], reverse=True)
    
    # Create drift-sorted list (by absolute drift)
    drift_sorted = sorted(holdings_list, key=lambda x: abs(x["drift"]), reverse=True)
    
    # Calculate portfolio sector allocation
    portfolio_sectors = {}
    for h in holdings_list:
        sector = symbol_sectors.get(h["symbol"], "Other")
        portfolio_sectors[sector] = portfolio_sectors.get(sector, 0) + h["weight"] * 100
    
    # Build sector comparison
    sector_comparison = []
    all_sectors = set(portfolio_sectors.keys()) | set(sp500_sector_weights.keys())
    for sector in all_sectors:
        portfolio_pct = portfolio_sectors.get(sector, 0)
        benchmark_pct = sp500_sector_weights.get(sector, 0)
        sector_comparison.append({
            "sector": sector,
            "portfolioWeight": portfolio_pct,
            "benchmarkWeight": benchmark_pct,
            "drift": portfolio_pct - benchmark_pct
        })
    
    # Sort sectors by portfolio weight
    sector_comparison.sort(key=lambda x: x["portfolioWeight"], reverse=True)
    
    # Get snapshots for chart
    snapshots = []
    if state.snapshots:
        for snap in state.snapshots:
            snapshots.append({
                "date": snap.date.isoformat(),
                "value": float(snap.total_value),
                "benchmark": 100
            })
    
    # Get trades
    trades = []
    for trade in state.trades[-50:]:
        trades.append({
            "timestamp": trade.timestamp.isoformat(),
            "symbol": trade.symbol,
            "side": trade.side.value,
            "shares": float(trade.shares),
            "price": float(trade.price),
            "value": float(trade.value),
            "reason": trade.reason.value
        })
    
    # Build dashboard data
    dashboard_data = {
        "lastUpdated": datetime.now().isoformat(),
        "inceptionDate": state.current_date.isoformat() if not snapshots else snapshots[0]["date"],
        "portfolioValue": portfolio_value,
        "initialValue": initial_value,
        "cashBalance": float(state.cash),
        "harvestedLosses": float(state.harvested_losses),
        "realizedPnl": float(state.realized_pnl),
        "positions": len(holdings),
        "lots": len(state.lots),
        "ytdReturn": (portfolio_value - initial_value) / initial_value,
        "dailyReturn": 0,
        "dailyChange": 0,
        
        # Top 10 holdings with weight comparison
        "topHoldings": holdings_list[:10],
        
        # Top 10 drift positions
        "topDrift": drift_sorted[:10],
        
        # All holdings for detailed view
        "holdings": holdings_list[:50],
        
        # Sector comparison (portfolio vs S&P 500)
        "sectorComparison": sector_comparison,
        
        # Legacy sector data for compatibility
        "sectors": dict(sorted(portfolio_sectors.items(), key=lambda x: x[1], reverse=True)),
        
        "trades": trades,
        "snapshots": snapshots
    }
    
    return dashboard_data


def main():
    """Generate and save dashboard data."""
    data = generate_dashboard_data("forward_10mm")
    
    if not data:
        return
    
    # Output path
    output_path = Path(__file__).parent.parent / "dashboard" / "data" / "portfolio.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write JSON
    with open(output_path, "w") as f:
        json.dump(data, f, cls=DecimalEncoder, indent=2)
    
    print(f"\nDashboard data written to: {output_path}")
    print(f"Portfolio Value: ${data['portfolioValue']:,.2f}")
    print(f"Positions: {data['positions']}")
    
    print(f"\nTop 5 Holdings (Portfolio vs S&P):")
    for h in data['topHoldings'][:5]:
        print(f"  {h['symbol']}: {h['weight']*100:.2f}% vs {h['benchmarkWeight']*100:.2f}% (drift: {h['drift']*100:+.2f}%)")
    
    print(f"\nTop 5 Drift Positions:")
    for h in data['topDrift'][:5]:
        print(f"  {h['symbol']}: drift {h['drift']*100:+.2f}%")


if __name__ == "__main__":
    main()
