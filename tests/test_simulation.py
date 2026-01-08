"""
Tests for simulation functionality.
"""

from datetime import date, datetime
from decimal import Decimal

import pytest

from di_pilot.models import BenchmarkConstituent, PriceData, TaxLot
from di_pilot.simulation.engine import (
    SimulationEngine,
    SimulationState,
    SimulationConfig,
    SimulationTrade,
    TradeReason,
)
from di_pilot.simulation.metrics import calculate_metrics, SimulationMetrics


@pytest.fixture
def simulation_config() -> SimulationConfig:
    """Create a test simulation configuration."""
    return SimulationConfig(
        initial_cash=Decimal("100000"),
        cash_buffer_pct=Decimal("0.01"),
        min_trade_value=Decimal("100"),
        max_turnover_pct=Decimal("0.10"),
        rebalance_band_pct=Decimal("0.02"),
        tlh_loss_threshold=Decimal("0.03"),
        rebalance_freq="weekly",
    )


@pytest.fixture
def simple_constituents() -> list[BenchmarkConstituent]:
    """Create simple test constituents."""
    return [
        BenchmarkConstituent(symbol="AAPL", weight=Decimal("0.40"), as_of_date=date(2024, 1, 2)),
        BenchmarkConstituent(symbol="MSFT", weight=Decimal("0.35"), as_of_date=date(2024, 1, 2)),
        BenchmarkConstituent(symbol="GOOG", weight=Decimal("0.25"), as_of_date=date(2024, 1, 2)),
    ]


@pytest.fixture
def simple_prices() -> dict[str, PriceData]:
    """Create simple test prices."""
    price_date = date(2024, 1, 2)
    return {
        "AAPL": PriceData(symbol="AAPL", date=price_date, close=Decimal("200")),
        "MSFT": PriceData(symbol="MSFT", date=price_date, close=Decimal("400")),
        "GOOG": PriceData(symbol="GOOG", date=price_date, close=Decimal("150")),
    }


class TestSimulationEngine:
    """Tests for the SimulationEngine class."""

    def test_initialize_portfolio(
        self,
        simulation_config: SimulationConfig,
        simple_constituents: list[BenchmarkConstituent],
        simple_prices: dict[str, PriceData],
    ):
        """Test portfolio initialization."""
        engine = SimulationEngine(simulation_config)
        state = engine.initialize_portfolio(
            portfolio_id="TEST001",
            start_date=date(2024, 1, 2),
            constituents=simple_constituents,
            prices=simple_prices,
        )

        # Should have lots for each constituent
        assert len(state.lots) == 3

        # Should have initial purchase trades
        initial_trades = [
            t for t in state.trades
            if t.reason == TradeReason.INITIAL_PURCHASE
        ]
        assert len(initial_trades) == 3

        # Cash should be mostly invested
        total_invested = sum(lot.shares * lot.cost_basis for lot in state.lots)
        assert total_invested <= simulation_config.initial_cash
        assert state.cash >= Decimal("0")
        assert state.cash + total_invested <= simulation_config.initial_cash

    def test_value_portfolio(
        self,
        simulation_config: SimulationConfig,
        simple_constituents: list[BenchmarkConstituent],
        simple_prices: dict[str, PriceData],
    ):
        """Test portfolio valuation."""
        engine = SimulationEngine(simulation_config)
        state = engine.initialize_portfolio(
            portfolio_id="TEST001",
            start_date=date(2024, 1, 2),
            constituents=simple_constituents,
            prices=simple_prices,
        )

        value = engine.value_portfolio(state, simple_prices)

        # Value should approximately equal initial cash
        assert value > Decimal("0")
        assert value <= simulation_config.initial_cash

    def test_calculate_weights(
        self,
        simulation_config: SimulationConfig,
        simple_constituents: list[BenchmarkConstituent],
        simple_prices: dict[str, PriceData],
    ):
        """Test weight calculation."""
        engine = SimulationEngine(simulation_config)
        state = engine.initialize_portfolio(
            portfolio_id="TEST001",
            start_date=date(2024, 1, 2),
            constituents=simple_constituents,
            prices=simple_prices,
        )

        weights = engine.calculate_weights(state, simple_prices)

        # Should have weights for all symbols
        assert "AAPL" in weights
        assert "MSFT" in weights
        assert "GOOG" in weights

        # Weights should sum to approximately 1 (excluding cash)
        total_weight = sum(weights.values())
        assert Decimal("0.95") < total_weight <= Decimal("1.00")

    def test_calculate_drift(
        self,
        simulation_config: SimulationConfig,
        simple_constituents: list[BenchmarkConstituent],
        simple_prices: dict[str, PriceData],
    ):
        """Test drift calculation."""
        engine = SimulationEngine(simulation_config)
        state = engine.initialize_portfolio(
            portfolio_id="TEST001",
            start_date=date(2024, 1, 2),
            constituents=simple_constituents,
            prices=simple_prices,
        )

        drift = engine.calculate_drift(state, simple_constituents, simple_prices)

        # Should have drift for all symbols
        assert "AAPL" in drift
        assert "MSFT" in drift
        assert "GOOG" in drift

        # Initial drift should be small (just from cash buffer)
        for symbol, d in drift.items():
            assert abs(d) < Decimal("0.05")  # Within 5%

    def test_identify_tlh_candidates(
        self,
        simulation_config: SimulationConfig,
    ):
        """Test TLH candidate identification."""
        engine = SimulationEngine(simulation_config)

        # Create state with a losing position
        state = SimulationState(
            portfolio_id="TEST001",
            current_date=date(2024, 6, 15),
            cash=Decimal("1000"),
            lots=[
                TaxLot(
                    lot_id="lot-001",
                    symbol="LOSER",
                    shares=Decimal("100"),
                    cost_basis=Decimal("100"),  # Bought at $100
                    acquisition_date=date(2024, 1, 2),
                    portfolio_id="TEST001",
                ),
                TaxLot(
                    lot_id="lot-002",
                    symbol="WINNER",
                    shares=Decimal("100"),
                    cost_basis=Decimal("50"),  # Bought at $50
                    acquisition_date=date(2024, 1, 2),
                    portfolio_id="TEST001",
                ),
            ],
        )

        # Current prices: LOSER down 10%, WINNER up 20%
        prices = {
            "LOSER": PriceData(symbol="LOSER", date=date(2024, 6, 15), close=Decimal("90")),
            "WINNER": PriceData(symbol="WINNER", date=date(2024, 6, 15), close=Decimal("60")),
        }

        candidates = engine.identify_tlh_candidates(state, prices, date(2024, 6, 15))

        # LOSER should be a TLH candidate (10% loss > 3% threshold)
        assert len(candidates) == 1
        assert candidates[0].symbol == "LOSER"

    def test_should_rebalance(self, simulation_config: SimulationConfig):
        """Test rebalance timing logic."""
        engine = SimulationEngine(simulation_config)

        # Should rebalance on first call (no last rebalance)
        assert engine.should_rebalance(date(2024, 1, 2), None) is True

        # Weekly: should not rebalance after 3 days
        assert engine.should_rebalance(date(2024, 1, 5), date(2024, 1, 2)) is False

        # Weekly: should rebalance after 7+ days
        assert engine.should_rebalance(date(2024, 1, 10), date(2024, 1, 2)) is True


class TestSimulationState:
    """Tests for the SimulationState class."""

    def test_get_lots_by_symbol(self):
        """Test getting lots by symbol."""
        state = SimulationState(
            portfolio_id="TEST001",
            current_date=date(2024, 1, 2),
            cash=Decimal("1000"),
            lots=[
                TaxLot(
                    lot_id="lot-001",
                    symbol="AAPL",
                    shares=Decimal("10"),
                    cost_basis=Decimal("100"),
                    acquisition_date=date(2024, 1, 2),
                    portfolio_id="TEST001",
                ),
                TaxLot(
                    lot_id="lot-002",
                    symbol="AAPL",
                    shares=Decimal("5"),
                    cost_basis=Decimal("110"),
                    acquisition_date=date(2024, 2, 1),
                    portfolio_id="TEST001",
                ),
                TaxLot(
                    lot_id="lot-003",
                    symbol="MSFT",
                    shares=Decimal("20"),
                    cost_basis=Decimal("200"),
                    acquisition_date=date(2024, 1, 2),
                    portfolio_id="TEST001",
                ),
            ],
        )

        aapl_lots = state.get_lots_by_symbol("AAPL")
        assert len(aapl_lots) == 2

        msft_lots = state.get_lots_by_symbol("MSFT")
        assert len(msft_lots) == 1

    def test_get_total_shares(self):
        """Test getting total shares for a symbol."""
        state = SimulationState(
            portfolio_id="TEST001",
            current_date=date(2024, 1, 2),
            cash=Decimal("1000"),
            lots=[
                TaxLot(
                    lot_id="lot-001",
                    symbol="AAPL",
                    shares=Decimal("10"),
                    cost_basis=Decimal("100"),
                    acquisition_date=date(2024, 1, 2),
                    portfolio_id="TEST001",
                ),
                TaxLot(
                    lot_id="lot-002",
                    symbol="AAPL",
                    shares=Decimal("5.5"),
                    cost_basis=Decimal("110"),
                    acquisition_date=date(2024, 2, 1),
                    portfolio_id="TEST001",
                ),
            ],
        )

        total = state.get_total_shares("AAPL")
        assert total == Decimal("15.5")

    def test_wash_sale_restriction(self):
        """Test wash sale restriction checking."""
        state = SimulationState(
            portfolio_id="TEST001",
            current_date=date(2024, 6, 15),
            cash=Decimal("1000"),
            wash_sale_symbols={"AAPL": date(2024, 6, 1)},
        )

        # Within 30-day window
        assert state.is_wash_sale_restricted("AAPL", date(2024, 6, 15), 30) is True

        # Outside 30-day window
        assert state.is_wash_sale_restricted("AAPL", date(2024, 7, 5), 30) is False

        # Symbol not in wash sale list
        assert state.is_wash_sale_restricted("MSFT", date(2024, 6, 15), 30) is False


class TestMetricsCalculation:
    """Tests for metrics calculation."""

    def test_basic_metrics(self):
        """Test basic metrics calculation."""
        from di_pilot.simulation.engine import DailySnapshot
        from di_pilot.models import TradeSide

        # Create simple snapshots
        snapshots = [
            DailySnapshot(
                date=date(2024, 1, 2),
                total_value=Decimal("100000"),
                cash=Decimal("1000"),
                holdings_value=Decimal("99000"),
                num_positions=10,
                num_lots=10,
                unrealized_pnl=Decimal("0"),
                realized_pnl=Decimal("0"),
                daily_return=Decimal("0"),
                cumulative_return=Decimal("0"),
            ),
            DailySnapshot(
                date=date(2024, 1, 3),
                total_value=Decimal("101000"),
                cash=Decimal("1000"),
                holdings_value=Decimal("100000"),
                num_positions=10,
                num_lots=10,
                unrealized_pnl=Decimal("1000"),
                realized_pnl=Decimal("0"),
                daily_return=Decimal("0.01"),
                cumulative_return=Decimal("0.01"),
            ),
            DailySnapshot(
                date=date(2024, 1, 4),
                total_value=Decimal("102000"),
                cash=Decimal("1000"),
                holdings_value=Decimal("101000"),
                num_positions=10,
                num_lots=10,
                unrealized_pnl=Decimal("2000"),
                realized_pnl=Decimal("0"),
                daily_return=Decimal("0.0099"),
                cumulative_return=Decimal("0.02"),
            ),
        ]

        # Create some trades
        trades = [
            SimulationTrade.create(
                timestamp=datetime.combine(date(2024, 1, 2), datetime.min.time()),
                symbol="AAPL",
                side=TradeSide.BUY,
                shares=Decimal("100"),
                price=Decimal("100"),
                reason=TradeReason.INITIAL_PURCHASE,
            ),
        ]

        metrics = calculate_metrics(
            snapshots=snapshots,
            trades=trades,
            initial_cash=Decimal("100000"),
        )

        assert isinstance(metrics, SimulationMetrics)
        assert metrics.trading_days == 3
        assert metrics.total_return > 0
        assert metrics.total_trades == 1
        assert metrics.final_value == 102000
