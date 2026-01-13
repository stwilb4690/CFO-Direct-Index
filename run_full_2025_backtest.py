"""
Full 2025 Backtest with Total Return Benchmark

This script runs a complete 500-symbol backtest and compares against
total return (price + dividends) rather than price-only return.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from datetime import date
from decimal import Decimal
from di_pilot.simulation.backtest import run_backtest
from di_pilot.data.providers.eodhd_provider import EODHDProvider
from di_pilot.simulation.engine import SimulationConfig

print("=" * 70)
print("2025 FULL YEAR BACKTEST - ALL S&P 500 SYMBOLS")
print("=" * 70)

provider = EODHDProvider()

# Full S&P 500 configuration
config = SimulationConfig(
    initial_cash=Decimal("1000000"),
    rebalance_freq="weekly",
    target_positions=0,  # 0 = ALL constituents
    enable_dividend_simulation=True,
)

def progress(msg):
    print(f"  {msg}")

print("\nRunning backtest...")
result = run_backtest(
    provider=provider,
    start_date=date(2025, 1, 2),
    end_date=date(2025, 12, 31),
    config=config,
    top_n_symbols=None,  # Use all symbols
    progress_callback=progress
)

# Calculate metrics
symbols = set(lot.symbol for lot in result.final_state.lots)
tlh_sells = [t for t in result.trades if "TLH_SELL" in str(t.reason)]
div_trades = [t for t in result.trades if "DIVIDEND" in str(t.reason)]
has_brk = "BRK.B" in symbols

# Get SPY total return (price + dividends)
# SPY 2025: Price return ~16.6%, Dividend yield ~1.3%, Total return ~17.9%
spy_price_return = float(result.benchmark_return) * 100
spy_dividend_yield = 1.3  # Approximate SPY dividend yield
spy_total_return = spy_price_return + spy_dividend_yield

portfolio_return = float(result.total_return) * 100

print()
print("=" * 70)
print("RESULTS")
print("=" * 70)
print(f"Initial Value:           $1,000,000.00")
print(f"Final Value:             ${float(result.final_value):,.2f}")
print()
print(f"Portfolio Return:        {portfolio_return:.2f}%")
print(f"SPY Price Return:        {spy_price_return:.2f}%")
print(f"SPY Total Return (est):  {spy_total_return:.2f}%")
print(f"Active Return:           {portfolio_return - spy_total_return:.2f}%")
print()
print(f"Total Trades:            {result.total_trades}")
print(f"Dividend Events:         {len(div_trades)}")
print(f"TLH Sell Trades:         {len(tlh_sells)}")
print(f"Harvested Losses:        ${float(result.harvested_losses):,.2f}")
print()
print(f"Final Positions:         {len(symbols)}")
print(f"Trading Days:            {len(result.snapshots)}")
print(f"Has BRK.B:               {has_brk}")
print("=" * 70)

# Save detailed results
output_path = Path("outputs/backtest_2025_full.txt")
output_path.parent.mkdir(exist_ok=True)

with open(output_path, "w") as f:
    f.write("2025 FULL YEAR BACKTEST - ALL S&P 500 SYMBOLS\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Initial Value:           $1,000,000.00\n")
    f.write(f"Final Value:             ${float(result.final_value):,.2f}\n\n")
    f.write(f"Portfolio Return:        {portfolio_return:.2f}%\n")
    f.write(f"SPY Price Return:        {spy_price_return:.2f}%\n")
    f.write(f"SPY Total Return (est):  {spy_total_return:.2f}%\n")
    f.write(f"Active Return:           {portfolio_return - spy_total_return:.2f}%\n\n")
    f.write(f"Total Trades:            {result.total_trades}\n")
    f.write(f"Dividend Events:         {len(div_trades)}\n")
    f.write(f"TLH Sell Trades:         {len(tlh_sells)}\n")
    f.write(f"Harvested Losses:        ${float(result.harvested_losses):,.2f}\n\n")
    f.write(f"Final Positions:         {len(symbols)}\n")
    f.write(f"Trading Days:            {len(result.snapshots)}\n")
    f.write(f"Has BRK.B:               {has_brk}\n")

print(f"\nResults saved to: {output_path}")

# Check for potential issues
print("\n" + "=" * 70)
print("DIAGNOSTIC CHECKS")
print("=" * 70)

# 1. Check if return seems reasonable
if portfolio_return > 25:
    print(f"⚠️  WARNING: Portfolio return ({portfolio_return:.2f}%) seems high")
    print("   Possible causes:")
    print("   - Top-heavy weighting in outperforming mega-caps")
    print("   - Dividend double-counting")
    print("   - Price data issue")

# 2. Check cash balance
cash_pct = float(result.final_state.cash) / float(result.final_value) * 100
print(f"\n📊 Final Cash Balance: ${float(result.final_state.cash):,.2f} ({cash_pct:.2f}% of portfolio)")

# 3. Check for large positions
position_values = {}
from di_pilot.models import PriceData
for lot in result.final_state.lots:
    sym = lot.symbol
    if sym not in position_values:
        position_values[sym] = Decimal("0")
    # Use last known price from snapshots
    position_values[sym] += lot.shares * lot.cost_basis  # Approximate

top_5 = sorted(position_values.items(), key=lambda x: x[1], reverse=True)[:5]
print("\n📊 Top 5 Positions by Value:")
for sym, val in top_5:
    print(f"   {sym}: ${float(val):,.2f}")
