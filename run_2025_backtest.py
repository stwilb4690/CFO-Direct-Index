"""Run 2025 Full Year Backtest and save results."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from datetime import date
from di_pilot.simulation.backtest import run_backtest
from di_pilot.data.providers.eodhd_provider import EODHDProvider
from di_pilot.simulation.engine import SimulationConfig
from decimal import Decimal

print("=" * 60)
print("2025 FULL YEAR BACKTEST - Top 100 Symbols")
print("=" * 60)

provider = EODHDProvider()
config = SimulationConfig(
    initial_cash=Decimal("1000000"),
    rebalance_freq="weekly",
    target_positions=100,
)

def progress(msg):
    print(f"  {msg}")

result = run_backtest(
    provider=provider,
    start_date=date(2025, 1, 2),
    end_date=date(2025, 12, 31),
    config=config,
    top_n_symbols=100,
    progress_callback=progress
)

# Calculate metrics
symbols = set(lot.symbol for lot in result.final_state.lots)
tlh_sells = [t for t in result.trades if "TLH_SELL" in str(t.reason)]
has_brk = "BRK.B" in symbols

print()
print("=" * 60)
print("RESULTS")
print("=" * 60)
print(f"Initial Value:        $1,000,000.00")
print(f"Final Value:          ${float(result.final_value):,.2f}")
print(f"Portfolio Return:     {float(result.total_return)*100:.2f}%")
print(f"Benchmark (SPY):      {float(result.benchmark_return)*100:.2f}%")
print(f"Tracking Difference:  {(float(result.total_return) - float(result.benchmark_return))*100:.2f}%")
print(f"Total Trades:         {result.total_trades}")
print(f"Harvested Losses:     ${float(result.harvested_losses):,.2f}")
print(f"Final Positions:      {len(symbols)}")
print(f"Trading Days:         {len(result.snapshots)}")
print(f"TLH Sell Trades:      {len(tlh_sells)}")
print(f"Has BRK.B:            {has_brk}")
print("=" * 60)

# Save to file as well
with open("outputs/backtest_2025_results.txt", "w") as f:
    f.write("2025 FULL YEAR BACKTEST RESULTS\n")
    f.write("=" * 50 + "\n")
    f.write(f"Initial Value:        $1,000,000.00\n")
    f.write(f"Final Value:          ${float(result.final_value):,.2f}\n")
    f.write(f"Portfolio Return:     {float(result.total_return)*100:.2f}%\n")
    f.write(f"Benchmark (SPY):      {float(result.benchmark_return)*100:.2f}%\n")
    f.write(f"Tracking Difference:  {(float(result.total_return) - float(result.benchmark_return))*100:.2f}%\n")
    f.write(f"Total Trades:         {result.total_trades}\n")
    f.write(f"Harvested Losses:     ${float(result.harvested_losses):,.2f}\n")
    f.write(f"Final Positions:      {len(symbols)}\n")
    f.write(f"Trading Days:         {len(result.snapshots)}\n")
    f.write(f"TLH Sell Trades:      {len(tlh_sells)}\n")
    f.write(f"Has BRK.B:            {has_brk}\n")

print("\nResults also saved to: outputs/backtest_2025_results.txt")
