"""
Forward test orchestration for direct indexing simulation.

Initializes a portfolio from cash on a specified start date and
tracks portfolio value and drift going forward. Used for paper
trading simulation to evaluate strategy performance in real-time.
"""

import json
import uuid
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Optional

import pandas as pd

from di_pilot.data.providers.base import DataProvider
from di_pilot.models import BenchmarkConstituent, TaxLot
from di_pilot.simulation.engine import (
    SimulationEngine,
    SimulationState,
    SimulationConfig,
    SimulationTrade,
    DailySnapshot,
)
from di_pilot.simulation.backtest import BacktestResult


class ForwardTestResult:
    """Container for forward test results."""

    def __init__(
        self,
        run_id: str,
        config: SimulationConfig,
        start_date: date,
        current_date: date,
        state: SimulationState,
    ):
        self.run_id = run_id
        self.config = config
        self.start_date = start_date
        self.current_date = current_date
        self.state = state

    @property
    def trades(self) -> list[SimulationTrade]:
        return self.state.trades

    @property
    def snapshots(self) -> list[DailySnapshot]:
        return self.state.snapshots

    @property
    def initial_value(self) -> Decimal:
        return self.config.initial_cash

    @property
    def current_value(self) -> Decimal:
        if self.snapshots:
            return self.snapshots[-1].total_value
        return self.config.initial_cash

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
        for snap in self.snapshots:
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
            })
        return pd.DataFrame(records)

    def save_outputs(self, output_dir: str | Path) -> dict[str, Path]:
        """Save forward test outputs to files."""
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

        # Save state for resumption
        state_path = output_dir / "state.json"
        self._save_state(state_path)
        paths["state"] = state_path

        return paths

    def _save_state(self, path: Path) -> None:
        """Save simulation state for later resumption."""
        state_dict = {
            "run_id": self.run_id,
            "start_date": self.start_date.isoformat(),
            "current_date": self.current_date.isoformat(),
            "config": {
                "initial_cash": str(self.config.initial_cash),
                "cash_buffer_pct": str(self.config.cash_buffer_pct),
                "min_trade_value": str(self.config.min_trade_value),
                "max_turnover_pct": str(self.config.max_turnover_pct),
                "rebalance_band_pct": str(self.config.rebalance_band_pct),
                "tlh_loss_threshold": str(self.config.tlh_loss_threshold),
                "tlh_wash_sale_days": self.config.tlh_wash_sale_days,
                "rebalance_freq": self.config.rebalance_freq,
            },
            "state": {
                "portfolio_id": self.state.portfolio_id,
                "cash": str(self.state.cash),
                "realized_pnl": str(self.state.realized_pnl),
                "harvested_losses": str(self.state.harvested_losses),
                "lots": [
                    {
                        "lot_id": lot.lot_id,
                        "symbol": lot.symbol,
                        "shares": str(lot.shares),
                        "cost_basis": str(lot.cost_basis),
                        "acquisition_date": lot.acquisition_date.isoformat(),
                        "portfolio_id": lot.portfolio_id,
                    }
                    for lot in self.state.lots
                ],
                "wash_sale_symbols": {
                    s: d.isoformat() for s, d in self.state.wash_sale_symbols.items()
                },
            },
        }

        with open(path, "w") as f:
            json.dump(state_dict, f, indent=2)

    @classmethod
    def load_state(cls, path: Path) -> "ForwardTestResult":
        """Load simulation state from file."""
        with open(path, "r") as f:
            state_dict = json.load(f)

        config = SimulationConfig(
            initial_cash=Decimal(state_dict["config"]["initial_cash"]),
            cash_buffer_pct=Decimal(state_dict["config"]["cash_buffer_pct"]),
            min_trade_value=Decimal(state_dict["config"]["min_trade_value"]),
            max_turnover_pct=Decimal(state_dict["config"]["max_turnover_pct"]),
            rebalance_band_pct=Decimal(state_dict["config"]["rebalance_band_pct"]),
            tlh_loss_threshold=Decimal(state_dict["config"]["tlh_loss_threshold"]),
            tlh_wash_sale_days=state_dict["config"]["tlh_wash_sale_days"],
            rebalance_freq=state_dict["config"]["rebalance_freq"],
        )

        state_data = state_dict["state"]
        lots = [
            TaxLot(
                lot_id=lot["lot_id"],
                symbol=lot["symbol"],
                shares=Decimal(lot["shares"]),
                cost_basis=Decimal(lot["cost_basis"]),
                acquisition_date=date.fromisoformat(lot["acquisition_date"]),
                portfolio_id=lot["portfolio_id"],
            )
            for lot in state_data["lots"]
        ]

        state = SimulationState(
            portfolio_id=state_data["portfolio_id"],
            current_date=date.fromisoformat(state_dict["current_date"]),
            cash=Decimal(state_data["cash"]),
            lots=lots,
            realized_pnl=Decimal(state_data["realized_pnl"]),
            harvested_losses=Decimal(state_data["harvested_losses"]),
            wash_sale_symbols={
                s: date.fromisoformat(d)
                for s, d in state_data.get("wash_sale_symbols", {}).items()
            },
        )

        return cls(
            run_id=state_dict["run_id"],
            config=config,
            start_date=date.fromisoformat(state_dict["start_date"]),
            current_date=date.fromisoformat(state_dict["current_date"]),
            state=state,
        )


def run_forward_test(
    provider: DataProvider,
    start_date: date,
    config: Optional[SimulationConfig] = None,
    top_n_symbols: Optional[int] = None,
    simulate_days: int = 0,
    progress_callback: Optional[callable] = None,
) -> ForwardTestResult:
    """
    Initialize and run a forward test simulation.

    This creates the initial portfolio from cash and optionally simulates
    forward for a number of days using available market data.

    Args:
        provider: Data provider for prices and constituents
        start_date: Start date for the forward test
        config: Simulation configuration (uses defaults if None)
        top_n_symbols: Limit to top N symbols by weight (for faster testing)
        simulate_days: Number of days to simulate forward (0 = just initialize)
        progress_callback: Optional callback for progress updates

    Returns:
        ForwardTestResult containing simulation state
    """
    if config is None:
        config = SimulationConfig()

    run_id = f"forward_{start_date}_{uuid.uuid4().hex[:8]}"

    # Get constituents
    if progress_callback:
        progress_callback("Fetching constituents...")

    constituents = provider.get_constituents(as_of_date=start_date)

    # Optionally limit to top N
    if top_n_symbols:
        constituents = sorted(constituents, key=lambda c: c.weight, reverse=True)[:top_n_symbols]
        # Renormalize weights
        total_weight = sum(c.weight for c in constituents)
        constituents = [
            BenchmarkConstituent(
                symbol=c.symbol,
                weight=c.weight / total_weight,
                as_of_date=c.as_of_date,
            )
            for c in constituents
        ]

    symbols = [c.symbol for c in constituents]

    # Get prices for start date
    if progress_callback:
        progress_callback("Fetching initial prices...")

    prices = provider.get_price_for_date(symbols, start_date)
    if not prices:
        raise ValueError(f"No prices available for start date {start_date}")

    # Initialize engine and portfolio
    if progress_callback:
        progress_callback("Initializing portfolio...")

    engine = SimulationEngine(config)
    state = engine.initialize_portfolio(
        portfolio_id=run_id,
        start_date=start_date,
        constituents=constituents,
        prices=prices,
    )

    # Record initial snapshot
    state = engine.record_snapshot(state, prices, start_date)
    current_date = start_date

    # Optionally simulate forward
    if simulate_days > 0:
        if progress_callback:
            progress_callback(f"Simulating {simulate_days} days forward...")

        from datetime import timedelta
        end_date = start_date + timedelta(days=simulate_days + 10)  # Buffer for non-trading days

        trading_days = provider.get_trading_days(start_date, end_date)
        trading_days = [d for d in trading_days if d > start_date][:simulate_days]

        if trading_days:
            # Get price data for simulation period
            price_df = provider.get_prices(symbols, start_date, trading_days[-1])

            # Build price lookup
            from di_pilot.models import PriceData
            price_by_date = {}
            for dt in trading_days:
                day_prices = price_df[price_df["date"] == dt]
                price_by_date[dt] = {
                    row["symbol"]: PriceData(
                        symbol=row["symbol"],
                        date=dt,
                        close=Decimal(str(row["close"])),
                    )
                    for _, row in day_prices.iterrows()
                }

            last_rebalance = start_date

            for i, sim_date in enumerate(trading_days):
                if sim_date not in price_by_date:
                    continue

                day_prices = price_by_date[sim_date]
                if not day_prices:
                    continue

                state.current_date = sim_date
                current_date = sim_date

                # Check if rebalance day
                if engine.should_rebalance(sim_date, last_rebalance):
                    state = engine.execute_tlh(state, day_prices, sim_date)
                    state = engine.execute_rebalance(state, constituents, day_prices, sim_date)
                    last_rebalance = sim_date

                # Record snapshot
                state = engine.record_snapshot(state, day_prices, sim_date)

    if progress_callback:
        progress_callback("Forward test complete!")

    return ForwardTestResult(
        run_id=run_id,
        config=config,
        start_date=start_date,
        current_date=current_date,
        state=state,
    )


def resume_forward_test(
    state_path: Path,
    provider: DataProvider,
    simulate_days: int = 1,
    progress_callback: Optional[callable] = None,
) -> ForwardTestResult:
    """
    Resume a forward test from saved state.

    Args:
        state_path: Path to saved state JSON file
        provider: Data provider
        simulate_days: Number of additional days to simulate
        progress_callback: Optional callback for progress updates

    Returns:
        Updated ForwardTestResult
    """
    if progress_callback:
        progress_callback("Loading saved state...")

    result = ForwardTestResult.load_state(state_path)

    if simulate_days <= 0:
        return result

    # Get constituents
    constituents = provider.get_constituents(as_of_date=result.current_date)
    symbols = [lot.symbol for lot in result.state.lots]

    # Get trading days from current date forward
    from datetime import timedelta
    end_date = result.current_date + timedelta(days=simulate_days + 10)
    trading_days = provider.get_trading_days(result.current_date, end_date)
    trading_days = [d for d in trading_days if d > result.current_date][:simulate_days]

    if not trading_days:
        return result

    # Get price data
    if progress_callback:
        progress_callback("Fetching price data...")

    price_df = provider.get_prices(symbols, result.current_date, trading_days[-1])

    from di_pilot.models import PriceData
    price_by_date = {}
    for dt in trading_days:
        day_prices = price_df[price_df["date"] == dt]
        price_by_date[dt] = {
            row["symbol"]: PriceData(
                symbol=row["symbol"],
                date=dt,
                close=Decimal(str(row["close"])),
            )
            for _, row in day_prices.iterrows()
        }

    # Continue simulation
    engine = SimulationEngine(result.config)
    state = result.state

    # Determine last rebalance from trades
    rebalance_trades = [
        t for t in state.trades
        if t.reason.value in ("REBALANCE_BUY", "REBALANCE_SELL", "TLH_SELL")
    ]
    last_rebalance = max(
        (t.timestamp.date() for t in rebalance_trades),
        default=result.start_date
    )

    if progress_callback:
        progress_callback(f"Simulating {len(trading_days)} days...")

    for sim_date in trading_days:
        if sim_date not in price_by_date:
            continue

        day_prices = price_by_date[sim_date]
        if not day_prices:
            continue

        state.current_date = sim_date

        if engine.should_rebalance(sim_date, last_rebalance):
            state = engine.execute_tlh(state, day_prices, sim_date)
            state = engine.execute_rebalance(state, constituents, day_prices, sim_date)
            last_rebalance = sim_date

        state = engine.record_snapshot(state, day_prices, sim_date)

    result.current_date = state.current_date
    result.state = state

    if progress_callback:
        progress_callback("Simulation updated!")

    return result
