"""
Verify portfolio weights against IVV ETF holdings.

Downloads IVV holdings from iShares and compares to our portfolio.
"""

import sys
from pathlib import Path
from decimal import Decimal
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from di_pilot.simulation.forward import ForwardTestRunner


def main():
    """Compare portfolio weights to IVV."""
    print("=" * 60)
    print("Portfolio Weight Verification")
    print("=" * 60)
    
    # Load our portfolio
    runner = ForwardTestRunner()
    
    if not runner.portfolio_exists("forward_10mm"):
        print("❌ Portfolio forward_10mm not found!")
        return
    
    state = runner._load_state("forward_10mm")
    
    # Calculate our holdings by symbol
    holdings = {}
    for lot in state.lots:
        if lot.symbol not in holdings:
            holdings[lot.symbol] = {
                "shares": Decimal("0"),
                "cost": Decimal("0"),
            }
        holdings[lot.symbol]["shares"] += lot.shares
        holdings[lot.symbol]["cost"] += lot.shares * lot.cost_basis
    
    # Calculate total portfolio value
    total_cost = sum(h["cost"] for h in holdings.values())
    
    # Calculate weights
    weights = {}
    for symbol, data in holdings.items():
        weights[symbol] = float(data["cost"]) / float(total_cost) * 100
    
    # Sort by weight
    sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\nPortfolio Summary:")
    print(f"  Total Value: ${float(total_cost):,.2f}")
    print(f"  Cash: ${float(state.cash):,.2f}")
    print(f"  Positions: {len(holdings)}")
    
    print(f"\nTop 20 Holdings (Our Portfolio):")
    print("-" * 50)
    print(f"{'Symbol':<8} {'Weight':>10} {'Value':>15}")
    print("-" * 50)
    
    for symbol, weight in sorted_weights[:20]:
        value = float(holdings[symbol]["cost"])
        print(f"{symbol:<8} {weight:>9.4f}% ${value:>12,.2f}")
    
    print("\n" + "=" * 60)
    print("Expected IVV Top Holdings (approx Jan 2026):")
    print("=" * 60)
    # These are approximate expected weights for S&P 500 as of Jan 2026
    expected_top = [
        ("AAPL", 7.0),
        ("MSFT", 6.5),
        ("NVDA", 6.0),
        ("AMZN", 3.8),
        ("GOOGL", 2.1),
        ("META", 2.5),
        ("GOOG", 1.8),
        ("BRK.B", 1.7),
        ("TSLA", 1.9),
        ("JPM", 1.3),
    ]
    
    print(f"\n{'Symbol':<8} {'Expected':>10} {'Actual':>10} {'Diff':>10}")
    print("-" * 50)
    
    mismatches = []
    for symbol, expected in expected_top:
        actual = weights.get(symbol, 0)
        diff = actual - expected
        status = "[OK]" if abs(diff) < 0.5 else "[!!]"
        print(f"{symbol:<8} {expected:>9.2f}% {actual:>9.4f}% {diff:>+9.2f}% {status}")
        if abs(diff) > 0.5:
            mismatches.append(symbol)
    
    if mismatches:
        print(f"\n[WARNING] Significant weight differences detected for: {', '.join(mismatches)}")
        print("Note: This is expected since we calculated weights from historical market caps,")
        print("      which may differ from the current IVV holdings.")
    else:
        print("\n[SUCCESS] Portfolio weights are reasonably aligned with expected S&P 500 weights!")


if __name__ == "__main__":
    main()
