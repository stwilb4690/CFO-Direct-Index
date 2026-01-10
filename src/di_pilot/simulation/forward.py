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


# =============================================================================
# ForwardTestRunner - Daily cron-style runner for live forward testing
# =============================================================================


class ForwardTestRunner:
    """
    Runner for daily forward test execution.

    Designed to be called daily (e.g., via cron) to:
    - Load existing portfolio state
    - Execute one day of simulation
    - Save updated state

    State is persisted to JSON files in data/portfolios/{portfolio_id}/
    """

    DEFAULT_DATA_DIR = Path("data/portfolios")

    def __init__(
        self,
        data_dir: str | Path | None = None,
        config: SimulationConfig | None = None,
    ):
        """
        Initialize the forward test runner.

        Args:
            data_dir: Base directory for portfolio data (default: data/portfolios)
            config: Simulation configuration (uses defaults if None)
        """
        self.data_dir = Path(data_dir) if data_dir else self.DEFAULT_DATA_DIR
        self.config = config or SimulationConfig()

    def _get_portfolio_dir(self, portfolio_id: str) -> Path:
        """Get directory for a portfolio's data."""
        return self.data_dir / portfolio_id

    def _get_state_path(self, portfolio_id: str) -> Path:
        """Get path to portfolio state file."""
        return self._get_portfolio_dir(portfolio_id) / "state.json"

    def _get_snapshots_path(self, portfolio_id: str) -> Path:
        """Get path to portfolio snapshots CSV."""
        return self._get_portfolio_dir(portfolio_id) / "snapshots.csv"

    def _save_state(self, portfolio_id: str, state: SimulationState) -> None:
        """
        Save simulation state to JSON file.

        Args:
            portfolio_id: Portfolio identifier
            state: Current simulation state
        """
        portfolio_dir = self._get_portfolio_dir(portfolio_id)
        portfolio_dir.mkdir(parents=True, exist_ok=True)

        state_dict = {
            "portfolio_id": state.portfolio_id,
            "current_date": state.current_date.isoformat(),
            "cash": str(state.cash),
            "realized_pnl": str(state.realized_pnl),
            "harvested_losses": str(state.harvested_losses),
            "lots": [
                {
                    "lot_id": lot.lot_id,
                    "symbol": lot.symbol,
                    "shares": str(lot.shares),
                    "cost_basis": str(lot.cost_basis),
                    "acquisition_date": lot.acquisition_date.isoformat(),
                    "portfolio_id": lot.portfolio_id,
                }
                for lot in state.lots
            ],
            "wash_sale_symbols": {
                s: d.isoformat() for s, d in state.wash_sale_symbols.items()
            },
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
            "last_updated": datetime.now().isoformat(),
        }

        state_path = self._get_state_path(portfolio_id)
        with open(state_path, "w") as f:
            json.dump(state_dict, f, indent=2)

    def _load_state(self, portfolio_id: str) -> SimulationState:
        """
        Load simulation state from JSON file.

        Args:
            portfolio_id: Portfolio identifier

        Returns:
            SimulationState loaded from file

        Raises:
            FileNotFoundError: If state file doesn't exist
        """
        state_path = self._get_state_path(portfolio_id)

        if not state_path.exists():
            raise FileNotFoundError(
                f"No state file found for portfolio {portfolio_id}. "
                f"Initialize first with initialize_portfolio()."
            )

        with open(state_path, "r") as f:
            state_dict = json.load(f)

        # Restore config from saved state
        if "config" in state_dict:
            self.config = SimulationConfig(
                initial_cash=Decimal(state_dict["config"]["initial_cash"]),
                cash_buffer_pct=Decimal(state_dict["config"]["cash_buffer_pct"]),
                min_trade_value=Decimal(state_dict["config"]["min_trade_value"]),
                max_turnover_pct=Decimal(state_dict["config"]["max_turnover_pct"]),
                rebalance_band_pct=Decimal(state_dict["config"]["rebalance_band_pct"]),
                tlh_loss_threshold=Decimal(state_dict["config"]["tlh_loss_threshold"]),
                tlh_wash_sale_days=state_dict["config"]["tlh_wash_sale_days"],
                rebalance_freq=state_dict["config"]["rebalance_freq"],
            )

        # Reconstruct lots
        lots = [
            TaxLot(
                lot_id=lot["lot_id"],
                symbol=lot["symbol"],
                shares=Decimal(lot["shares"]),
                cost_basis=Decimal(lot["cost_basis"]),
                acquisition_date=date.fromisoformat(lot["acquisition_date"]),
                portfolio_id=lot["portfolio_id"],
            )
            for lot in state_dict["lots"]
        ]

        # Reconstruct wash sale symbols
        wash_sale_symbols = {
            s: date.fromisoformat(d)
            for s, d in state_dict.get("wash_sale_symbols", {}).items()
        }

        return SimulationState(
            portfolio_id=state_dict["portfolio_id"],
            current_date=date.fromisoformat(state_dict["current_date"]),
            cash=Decimal(state_dict["cash"]),
            lots=lots,
            realized_pnl=Decimal(state_dict.get("realized_pnl", "0")),
            harvested_losses=Decimal(state_dict.get("harvested_losses", "0")),
            wash_sale_symbols=wash_sale_symbols,
        )

    def _append_snapshot(self, portfolio_id: str, snapshot: DailySnapshot) -> None:
        """
        Append a daily snapshot to the CSV file.

        Args:
            portfolio_id: Portfolio identifier
            snapshot: Daily snapshot to append
        """
        snapshots_path = self._get_snapshots_path(portfolio_id)
        file_exists = snapshots_path.exists()

        record = {
            "date": snapshot.date.isoformat(),
            "total_value": float(snapshot.total_value),
            "cash": float(snapshot.cash),
            "holdings_value": float(snapshot.holdings_value),
            "num_positions": snapshot.num_positions,
            "num_lots": snapshot.num_lots,
            "unrealized_pnl": float(snapshot.unrealized_pnl),
            "realized_pnl": float(snapshot.realized_pnl),
            "daily_return": float(snapshot.daily_return),
            "cumulative_return": float(snapshot.cumulative_return),
        }

        df = pd.DataFrame([record])

        if file_exists:
            df.to_csv(snapshots_path, mode="a", header=False, index=False)
        else:
            df.to_csv(snapshots_path, index=False)

    def portfolio_exists(self, portfolio_id: str) -> bool:
        """Check if a portfolio exists."""
        return self._get_state_path(portfolio_id).exists()

    def initialize_portfolio(
        self,
        portfolio_id: str,
        initial_cash: Decimal,
        provider: DataProvider,
        start_date: date | None = None,
        top_n_symbols: int | None = None,
    ) -> SimulationState:
        """
        Initialize a new portfolio from cash.

        Args:
            portfolio_id: Unique portfolio identifier
            initial_cash: Initial cash amount
            provider: Data provider for prices and constituents
            start_date: Start date (default: today)
            top_n_symbols: Limit to top N symbols by weight

        Returns:
            Initialized SimulationState

        Raises:
            ValueError: If portfolio already exists
        """
        if self.portfolio_exists(portfolio_id):
            raise ValueError(
                f"Portfolio {portfolio_id} already exists. "
                f"Delete the state file to reinitialize."
            )

        effective_date = start_date or date.today()

        # Update config with initial cash
        self.config = SimulationConfig(
            initial_cash=initial_cash,
            cash_buffer_pct=self.config.cash_buffer_pct,
            min_trade_value=self.config.min_trade_value,
            max_turnover_pct=self.config.max_turnover_pct,
            rebalance_band_pct=self.config.rebalance_band_pct,
            tlh_loss_threshold=self.config.tlh_loss_threshold,
            tlh_wash_sale_days=self.config.tlh_wash_sale_days,
            rebalance_freq=self.config.rebalance_freq,
        )

        # Get constituents
        constituents = provider.get_constituents(as_of_date=effective_date)

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

        # Get prices
        prices = provider.get_price_for_date(symbols, effective_date)
        if not prices:
            raise ValueError(f"No prices available for date {effective_date}")

        # Initialize engine and portfolio
        engine = SimulationEngine(self.config)
        state = engine.initialize_portfolio(
            portfolio_id=portfolio_id,
            start_date=effective_date,
            constituents=constituents,
            prices=prices,
        )

        # Record initial snapshot
        state = engine.record_snapshot(state, prices, effective_date)

        # Save state
        self._save_state(portfolio_id, state)

        # Append snapshot to CSV
        if state.snapshots:
            self._append_snapshot(portfolio_id, state.snapshots[-1])

        return state

    def run_daily(
        self,
        portfolio_id: str,
        provider: DataProvider,
        as_of_date: date | None = None,
        force_rebalance: bool = False,
    ) -> DailySnapshot:
        """
        Run one day of simulation.

        Args:
            portfolio_id: Portfolio identifier
            provider: Data provider for prices and constituents
            as_of_date: Date to simulate (default: today)
            force_rebalance: Force rebalance regardless of schedule

        Returns:
            Today's DailySnapshot

        Raises:
            FileNotFoundError: If portfolio doesn't exist
            ValueError: If no prices available
        """
        run_date = as_of_date or date.today()

        # Load existing state
        state = self._load_state(portfolio_id)

        # Check if we already processed this date
        if state.snapshots and state.snapshots[-1].date >= run_date:
            # Return existing snapshot for this date
            for snap in reversed(state.snapshots):
                if snap.date == run_date:
                    return snap
            # If exact date not found, return most recent
            return state.snapshots[-1]

        # Get symbols from current holdings
        symbols = list(set(lot.symbol for lot in state.lots))
        if not symbols:
            raise ValueError(f"No holdings in portfolio {portfolio_id}")

        # Get today's prices
        prices = provider.get_price_for_date(symbols, run_date)
        if not prices:
            raise ValueError(f"No prices available for date {run_date}")

        # Get constituents for rebalancing
        constituents = provider.get_constituents(as_of_date=run_date)

        # Initialize engine
        engine = SimulationEngine(self.config)
        state.current_date = run_date

        # Determine last rebalance date
        rebalance_trades = [
            t for t in state.trades
            if t.reason.value in ("REBALANCE_BUY", "REBALANCE_SELL", "TLH_SELL")
        ]
        last_rebalance = max(
            (t.timestamp.date() for t in rebalance_trades),
            default=state.current_date
        )

        # Check if rebalance is needed
        if force_rebalance or engine.should_rebalance(run_date, last_rebalance):
            # Execute TLH first
            state = engine.execute_tlh(state, prices, run_date)

            # Then rebalance
            state = engine.execute_rebalance(state, constituents, prices, run_date)

        # Record snapshot
        state = engine.record_snapshot(state, prices, run_date)
        today_snapshot = state.snapshots[-1]

        # Save updated state
        self._save_state(portfolio_id, state)

        # Append snapshot to CSV
        self._append_snapshot(portfolio_id, today_snapshot)

        return today_snapshot

    def get_portfolio_status(
        self,
        portfolio_id: str,
        provider: DataProvider | None = None,
    ) -> dict:
        """
        Get current portfolio status.

        Args:
            portfolio_id: Portfolio identifier
            provider: Optional data provider for live prices

        Returns:
            Dictionary with:
            - portfolio_id: Portfolio identifier
            - current_date: Date of last update
            - total_value: Current portfolio value
            - cash: Cash balance
            - holdings_value: Value of holdings
            - num_positions: Number of positions
            - num_lots: Number of tax lots
            - realized_pnl: Realized P&L to date
            - harvested_losses: Total harvested losses
            - unrealized_pnl: Unrealized P&L (if provider given)
            - cumulative_return: Total return percentage
            - positions: List of position summaries
            - tlh_candidates: List of TLH candidates (if provider given)
            - drift: Drift analysis (if provider given)
        """
        state = self._load_state(portfolio_id)

        # Basic status from saved state
        status = {
            "portfolio_id": portfolio_id,
            "current_date": state.current_date.isoformat(),
            "cash": float(state.cash),
            "realized_pnl": float(state.realized_pnl),
            "harvested_losses": float(state.harvested_losses),
            "num_positions": len(set(lot.symbol for lot in state.lots)),
            "num_lots": len(state.lots),
            "initial_cash": float(self.config.initial_cash),
        }

        # Calculate position summaries
        positions = {}
        for lot in state.lots:
            if lot.symbol not in positions:
                positions[lot.symbol] = {
                    "symbol": lot.symbol,
                    "shares": Decimal("0"),
                    "cost_basis": Decimal("0"),
                    "num_lots": 0,
                }
            positions[lot.symbol]["shares"] += lot.shares
            positions[lot.symbol]["cost_basis"] += lot.shares * lot.cost_basis
            positions[lot.symbol]["num_lots"] += 1

        # Convert to list with averages
        position_list = []
        for symbol, pos in positions.items():
            avg_cost = pos["cost_basis"] / pos["shares"] if pos["shares"] > 0 else Decimal("0")
            position_list.append({
                "symbol": symbol,
                "shares": float(pos["shares"]),
                "avg_cost_basis": float(avg_cost),
                "total_cost": float(pos["cost_basis"]),
                "num_lots": pos["num_lots"],
            })

        status["positions"] = sorted(position_list, key=lambda p: p["total_cost"], reverse=True)

        # If we have last snapshot, use its values
        if state.snapshots:
            last_snap = state.snapshots[-1]
            status["total_value"] = float(last_snap.total_value)
            status["holdings_value"] = float(last_snap.holdings_value)
            status["unrealized_pnl"] = float(last_snap.unrealized_pnl)
            status["cumulative_return"] = float(last_snap.cumulative_return)
            status["daily_return"] = float(last_snap.daily_return)

        # If provider given, get live data
        if provider:
            try:
                symbols = list(set(lot.symbol for lot in state.lots))
                prices = provider.get_price_for_date(symbols, date.today())

                if prices:
                    engine = SimulationEngine(self.config)

                    # Calculate live values
                    total_value = engine.value_portfolio(state, prices)
                    status["live_total_value"] = float(total_value)
                    status["live_return"] = float(
                        (total_value - self.config.initial_cash) / self.config.initial_cash
                    )

                    # Add current prices to positions
                    for pos in status["positions"]:
                        if pos["symbol"] in prices:
                            current_price = float(prices[pos["symbol"]].close)
                            pos["current_price"] = current_price
                            pos["market_value"] = pos["shares"] * current_price
                            pos["unrealized_pnl"] = pos["market_value"] - pos["total_cost"]

                    # TLH candidates
                    candidates = engine.identify_tlh_candidates(state, prices, date.today())
                    status["tlh_candidates"] = [
                        {
                            "symbol": lot.symbol,
                            "shares": float(lot.shares),
                            "cost_basis": float(lot.cost_basis),
                            "current_price": float(prices[lot.symbol].close),
                            "loss_amount": float(
                                lot.shares * (prices[lot.symbol].close - lot.cost_basis)
                            ),
                            "loss_pct": float(
                                (prices[lot.symbol].close - lot.cost_basis) / lot.cost_basis
                            ),
                        }
                        for lot in candidates
                        if lot.symbol in prices
                    ]

                    # Drift analysis
                    constituents = provider.get_constituents()
                    drift = engine.calculate_drift(state, constituents, prices)
                    significant_drift = {
                        s: float(d) for s, d in drift.items()
                        if abs(d) > self.config.rebalance_band_pct
                    }
                    status["significant_drift"] = dict(
                        sorted(significant_drift.items(), key=lambda x: abs(x[1]), reverse=True)
                    )

            except Exception as e:
                status["live_data_error"] = str(e)

        return status

    def get_trade_history(self, portfolio_id: str) -> list[dict]:
        """
        Get trade history for a portfolio.

        Args:
            portfolio_id: Portfolio identifier

        Returns:
            List of trade dictionaries
        """
        state = self._load_state(portfolio_id)

        return [
            {
                "trade_id": trade.trade_id,
                "timestamp": trade.timestamp.isoformat(),
                "symbol": trade.symbol,
                "side": trade.side.value,
                "shares": float(trade.shares),
                "price": float(trade.price),
                "value": float(trade.value),
                "reason": trade.reason.value,
                "lot_id": trade.lot_id or "",
                "notes": trade.notes,
            }
            for trade in state.trades
        ]

    def delete_portfolio(self, portfolio_id: str) -> bool:
        """
        Delete a portfolio and all its data.

        Args:
            portfolio_id: Portfolio identifier

        Returns:
            True if deleted, False if not found
        """
        import shutil

        portfolio_dir = self._get_portfolio_dir(portfolio_id)
        if portfolio_dir.exists():
            shutil.rmtree(portfolio_dir)
            return True
        return False
