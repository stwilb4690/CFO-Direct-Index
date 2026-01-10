"""
Backtest orchestration for direct indexing simulation.

Simulates historical portfolio performance from a start date to an end date,
including initial deployment, periodic rebalancing, and tax-loss harvesting.
"""

import uuid
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Optional

import pandas as pd

from di_pilot.data.providers.base import DataProvider
from di_pilot.simulation.engine import (
    SimulationEngine,
    SimulationState,
    SimulationConfig,
    SimulationTrade,
    DailySnapshot,
)


class BacktestResult:
    """Container for backtest results."""

    def __init__(
        self,
        run_id: str,
        config: SimulationConfig,
        start_date: date,
        end_date: date,
        final_state: SimulationState,
        benchmark_prices: dict[date, Decimal] = None,
    ):
        self.run_id = run_id
        self.config = config
        self.start_date = start_date
        self.end_date = end_date
        self.final_state = final_state
        self.benchmark_prices = benchmark_prices or {}

    @property
    def trades(self) -> list[SimulationTrade]:
        return self.final_state.trades

    @property
    def snapshots(self) -> list[DailySnapshot]:
        return self.final_state.snapshots

    @property
    def initial_value(self) -> Decimal:
        return self.config.initial_cash

    @property
    def final_value(self) -> Decimal:
        if self.snapshots:
            return self.snapshots[-1].total_value
        return self.config.initial_cash

    @property
    def total_return(self) -> Decimal:
        return (self.final_value - self.initial_value) / self.initial_value

    @property
    def benchmark_return(self) -> Decimal:
        """Calculate total return of the benchmark."""
        if not self.benchmark_prices:
            return Decimal("0")
        
        # Get start and end prices
        dates = sorted(self.benchmark_prices.keys())
        if not dates:
            return Decimal("0")
            
        start_price = self.benchmark_prices[dates[0]]
        end_price = self.benchmark_prices[dates[-1]]
        
        if start_price == 0:
            return Decimal("0")
            
        return (end_price - start_price) / start_price

    @property
    def total_trades(self) -> int:
        return len(self.trades)

    @property
    def harvested_losses(self) -> Decimal:
        return self.final_state.harvested_losses

    def trades_to_dataframe(self) -> pd.DataFrame:
        """Convert trades to DataFrame."""
        records = []
        for trade in self.trades:
            records.append({
                "trade_id": trade.trade_id,
                "timestamp": trade.timestamp,
                "symbol": trade.symbol,
                "side": trade.side.value,
                "shares": float(trade.shares),
                "price": float(trade.price),
                "value": float(trade.value),
                "reason": trade.reason.value,
                "lot_id": trade.lot_id or "",
                "notes": trade.notes,
            })
        return pd.DataFrame(records)

    def snapshots_to_dataframe(self) -> pd.DataFrame:
        """Convert snapshots to DataFrame."""
        records = []
        # Create map of date to benchmark price
        bench_map = {d: float(p) for d, p in self.benchmark_prices.items()}
        
        # Get initial benchmark price for normalization
        initial_bench = None
        
        for snap in self.snapshots:
            bench_price = bench_map.get(snap.date)
            
            # Initialize benchmark tracking
            if initial_bench is None and bench_price is not None:
                initial_bench = bench_price
                
            # Calculate daily benchmark return
            # (Approximated since we only strictly have prices)
            bench_return = 0.0
            if initial_bench and bench_price:
                # This is actually cumulative return relative to start
                bench_cum_return = (bench_price - initial_bench) / initial_bench
            else:
                bench_cum_return = 0.0

            records.append({
                "date": snap.date,
                "total_value": float(snap.total_value),
                "cash": float(snap.cash),
                "holdings_value": float(snap.holdings_value),
                "num_positions": snap.num_positions,
                "num_lots": snap.num_lots,
                "unrealized_pnl": float(snap.unrealized_pnl),
                "realized_pnl": float(snap.realized_pnl),
                "daily_return": float(snap.daily_return),
                "cumulative_return": float(snap.cumulative_return),
                "benchmark_price": bench_price,
                "benchmark_return": bench_cum_return,
            })
        return pd.DataFrame(records)

    def save_outputs(self, output_dir: str | Path) -> dict[str, Path]:
        """
        Save backtest outputs to files.

        Args:
            output_dir: Directory to save outputs

        Returns:
            Dictionary mapping output type to file path
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        paths = {}

        # Save trades
        trades_path = output_dir / "trades.csv"
        self.trades_to_dataframe().to_csv(trades_path, index=False)
        paths["trades"] = trades_path

        # Save daily snapshots
        portfolio_path = output_dir / "portfolio_daily.csv"
        self.snapshots_to_dataframe().to_csv(portfolio_path, index=False)
        paths["portfolio_daily"] = portfolio_path

        return paths


def run_backtest(
    provider: DataProvider,
    start_date: date,
    end_date: date,
    config: Optional[SimulationConfig] = None,
    top_n_symbols: Optional[int] = None,
    progress_callback: Optional[callable] = None,
) -> BacktestResult:
    """
    Run a complete backtest simulation.

    Args:
        provider: Data provider for prices and constituents
        start_date: Start date for the backtest
        end_date: End date for the backtest
        config: Simulation configuration (uses defaults if None)
        top_n_symbols: Limit to top N symbols by weight (for faster testing)
        progress_callback: Optional callback for progress updates

    Returns:
        BacktestResult containing all simulation data
    """
    if config is None:
        config = SimulationConfig()

    run_id = f"backtest_{start_date}_{end_date}_{uuid.uuid4().hex[:8]}"

    # Get constituents
    if progress_callback:
        progress_callback("Fetching constituents...")

    constituents = provider.get_constituents(as_of_date=start_date)

    # Warn about historical constituent support
    if not provider.supports_historical_constituents:
        if progress_callback:
            progress_callback(
                "WARNING: Using current S&P 500 members (survivorship bias). "
                "Use EODHD provider for historical constituent accuracy."
            )

    # Optionally limit to top N
    if top_n_symbols:
        constituents = sorted(constituents, key=lambda c: c.weight, reverse=True)[:top_n_symbols]
        # Renormalize weights
        total_weight = sum(c.weight for c in constituents)
        from di_pilot.models import BenchmarkConstituent
        constituents = [
            BenchmarkConstituent(
                symbol=c.symbol,
                weight=c.weight / total_weight,
                as_of_date=c.as_of_date,
            )
            for c in constituents
        ]

    symbols = [c.symbol for c in constituents]

    # Get trading days
    if progress_callback:
        progress_callback("Fetching trading days...")

    trading_days = provider.get_trading_days(start_date, end_date)
    if not trading_days:
        raise ValueError(f"No trading days between {start_date} and {end_date}")

    # Validate date range coverage
    actual_first = trading_days[0]
    actual_last = trading_days[-1]
    expected_days = (end_date - start_date).days * 5 // 7  # Rough estimate excluding weekends

    if (actual_first - start_date).days > 7:
        raise ValueError(
            f"Data provider returned data starting {actual_first}, but requested {start_date}. "
            f"Try running with --no-cache flag to fetch fresh data."
        )

    if (end_date - actual_last).days > 7:
        raise ValueError(
            f"Data provider returned data only through {actual_last}, but requested {end_date}. "
            f"Received {len(trading_days)} trading days instead of expected ~{expected_days}. "
            f"Try running with --no-cache flag to fetch fresh data, or check your data provider subscription."
        )

    if progress_callback:
        progress_callback(f"Found {len(trading_days)} trading days ({actual_first} to {actual_last})")

    # Fetch BENCHMARK prices (SPY)
    if progress_callback:
        progress_callback("Fetching benchmark (SPY) data...")
    
    benchmark_df = provider.get_prices(["SPY"], start_date, end_date)
    benchmark_prices = {}
    if not benchmark_df.empty:
        for _, row in benchmark_df.iterrows():
            benchmark_prices[row["date"]] = Decimal(str(row["close"]))

    # Get all price data upfront
    if progress_callback:
        progress_callback(f"Fetching price data for {len(symbols)} symbols...")

    price_df = provider.get_prices(symbols, start_date, end_date)
    if price_df.empty:
        raise ValueError("No price data available")

    # Build price lookup by date
    price_by_date = {}
    for dt in trading_days:
        day_prices = price_df[price_df["date"] == dt]
        from di_pilot.models import PriceData
        price_by_date[dt] = {
            row["symbol"]: PriceData(
                symbol=row["symbol"],
                date=dt,
                close=Decimal(str(row["close"])),
            )
            for _, row in day_prices.iterrows()
        }

    # Initialize engine
    engine = SimulationEngine(config)

    # Initialize portfolio on first trading day
    first_day = trading_days[0]
    if first_day not in price_by_date or not price_by_date[first_day]:
        raise ValueError(f"No prices available for start date {first_day}")

    if progress_callback:
        progress_callback("Initializing portfolio...")

    state = engine.initialize_portfolio(
        portfolio_id=run_id,
        start_date=first_day,
        constituents=constituents,
        prices=price_by_date[first_day],
    )

    # Record initial snapshot
    state = engine.record_snapshot(state, price_by_date[first_day], first_day)

    # Track last rebalance date
    last_rebalance = first_day

    # Simulate each trading day
    total_days = len(trading_days)
    for i, current_date in enumerate(trading_days[1:], 1):
        # Report progress every 5 days or at 25%, 50%, 75%, 100%
        if progress_callback:
            pct = int((i / total_days) * 100)
            if i % 5 == 0 or pct in [25, 50, 75] or i == total_days - 1:
                progress_callback(f"Progress: {pct}% (Day {i}/{total_days})")

        if current_date not in price_by_date:
            continue

        prices = price_by_date[current_date]
        if not prices:
            continue

        state.current_date = current_date

        # Check if rebalance day
        if engine.should_rebalance(current_date, last_rebalance):
            # Execute TLH first (on rebalance days)
            state = engine.execute_tlh(state, prices, current_date)

            # Then rebalance
            state = engine.execute_rebalance(state, constituents, prices, current_date)
            last_rebalance = current_date

        # Record daily snapshot
        state = engine.record_snapshot(state, prices, current_date)

    if progress_callback:
        progress_callback("Progress: 100% - Backtest complete!")

    return BacktestResult(
        run_id=run_id,
        config=config,
        start_date=start_date,
        end_date=end_date,
        final_state=state,
        benchmark_prices=benchmark_prices,
    )


def run_quick_backtest(
    provider: DataProvider,
    start_date: date,
    end_date: date,
    initial_cash: Decimal = Decimal("1000000"),
    top_n: int = 20,
) -> BacktestResult:
    """
    Run a quick backtest with limited symbols for sanity checking.

    Args:
        provider: Data provider
        start_date: Start date
        end_date: End date
        initial_cash: Initial cash amount
        top_n: Number of top symbols to use

    Returns:
        BacktestResult
    """
    config = SimulationConfig(
        initial_cash=initial_cash,
        rebalance_freq="weekly",
    )

    return run_backtest(
        provider=provider,
        start_date=start_date,
        end_date=end_date,
        config=config,
        top_n_symbols=top_n,
    )
