"""
Tests for portfolio initialization functionality.
"""

from datetime import date
from decimal import Decimal

import pytest

from di_pilot.models import BenchmarkConstituent, PortfolioConfig, PriceData
from di_pilot.portfolio.initialize import (
    initialize_portfolio,
    calculate_target_shares,
    validate_initialization_inputs,
    InitializationError,
)


class TestInitializePortfolio:
    """Tests for the initialize_portfolio function."""

    def test_basic_initialization(
        self,
        sample_portfolio_config: PortfolioConfig,
        sample_constituents: list[BenchmarkConstituent],
        sample_prices_jan: dict[str, PriceData],
    ):
        """Test basic portfolio initialization from cash."""
        lots, proposals = initialize_portfolio(
            config=sample_portfolio_config,
            constituents=sample_constituents,
            prices=sample_prices_jan,
        )

        # Should create lots for each constituent with a price
        assert len(lots) > 0
        assert len(proposals) > 0

        # Each lot should have valid data
        for lot in lots:
            assert lot.portfolio_id == sample_portfolio_config.portfolio_id
            assert lot.shares > Decimal("0")
            assert lot.cost_basis > Decimal("0")
            assert lot.acquisition_date == sample_portfolio_config.start_date

    def test_allocation_proportional_to_weights(
        self,
        sample_portfolio_config: PortfolioConfig,
        sample_constituents: list[BenchmarkConstituent],
        sample_prices_jan: dict[str, PriceData],
    ):
        """Test that allocation is roughly proportional to benchmark weights."""
        lots, _ = initialize_portfolio(
            config=sample_portfolio_config,
            constituents=sample_constituents,
            prices=sample_prices_jan,
        )

        # Calculate total allocated
        total_allocated = sum(lot.total_cost for lot in lots)

        # Should have allocated most of the cash (minus rounding)
        assert total_allocated <= sample_portfolio_config.cash
        assert total_allocated >= sample_portfolio_config.cash * Decimal("0.95")

        # Find the lot for AAPL (highest weight in sample)
        aapl_lots = [lot for lot in lots if lot.symbol == "AAPL"]
        assert len(aapl_lots) == 1

        aapl_lot = aapl_lots[0]
        aapl_allocation = aapl_lot.total_cost

        # AAPL should have roughly 7% of total (highest weight)
        aapl_weight = aapl_allocation / total_allocated
        assert aapl_weight > Decimal("0.05")  # At least 5%

    def test_fractional_shares(
        self,
        sample_portfolio_config: PortfolioConfig,
        sample_constituents: list[BenchmarkConstituent],
        sample_prices_jan: dict[str, PriceData],
    ):
        """Test that fractional shares are tracked to 6 decimal places."""
        lots, _ = initialize_portfolio(
            config=sample_portfolio_config,
            constituents=sample_constituents,
            prices=sample_prices_jan,
        )

        # At least some lots should have fractional shares
        fractional_lots = [
            lot for lot in lots
            if lot.shares != lot.shares.to_integral_value()
        ]
        assert len(fractional_lots) > 0

        # Check decimal precision
        for lot in lots:
            # Quantize to 6 decimal places and verify it matches
            quantized = lot.shares.quantize(Decimal("0.000001"))
            assert lot.shares == quantized

    def test_proposals_match_lots(
        self,
        sample_portfolio_config: PortfolioConfig,
        sample_constituents: list[BenchmarkConstituent],
        sample_prices_jan: dict[str, PriceData],
    ):
        """Test that trade proposals match the lots created."""
        lots, proposals = initialize_portfolio(
            config=sample_portfolio_config,
            constituents=sample_constituents,
            prices=sample_prices_jan,
        )

        # Should have same number of proposals as lots
        assert len(proposals) == len(lots)

        # Each proposal should be a BUY
        for proposal in proposals:
            assert proposal.side.value == "BUY"
            assert proposal.rationale.value == "INITIAL_PURCHASE"

        # Symbols should match
        lot_symbols = {lot.symbol for lot in lots}
        proposal_symbols = {p.symbol for p in proposals}
        assert lot_symbols == proposal_symbols

    def test_initialization_with_no_prices_fails(
        self,
        sample_portfolio_config: PortfolioConfig,
        sample_constituents: list[BenchmarkConstituent],
    ):
        """Test that initialization fails if no prices are available."""
        with pytest.raises(InitializationError):
            initialize_portfolio(
                config=sample_portfolio_config,
                constituents=sample_constituents,
                prices={},
            )

    def test_initialization_with_no_constituents_fails(
        self,
        sample_portfolio_config: PortfolioConfig,
        sample_prices_jan: dict[str, PriceData],
    ):
        """Test that initialization fails if no constituents are provided."""
        with pytest.raises(InitializationError):
            initialize_portfolio(
                config=sample_portfolio_config,
                constituents=[],
                prices=sample_prices_jan,
            )


class TestCalculateTargetShares:
    """Tests for the calculate_target_shares function."""

    def test_basic_calculation(self):
        """Test basic share calculation."""
        shares = calculate_target_shares(
            cash=Decimal("100000"),
            weight=Decimal("0.10"),  # 10%
            price=Decimal("100"),
        )
        # $10,000 / $100 = 100 shares
        assert shares == Decimal("100")

    def test_fractional_shares(self):
        """Test calculation resulting in fractional shares."""
        shares = calculate_target_shares(
            cash=Decimal("100000"),
            weight=Decimal("0.10"),
            price=Decimal("333.33"),
        )
        # $10,000 / $333.33 = 30.0003... -> truncated
        assert shares < Decimal("31")
        assert shares > Decimal("29")

    def test_zero_price_returns_zero(self):
        """Test that zero price returns zero shares."""
        shares = calculate_target_shares(
            cash=Decimal("100000"),
            weight=Decimal("0.10"),
            price=Decimal("0"),
        )
        assert shares == Decimal("0")

    def test_negative_price_returns_zero(self):
        """Test that negative price returns zero shares."""
        shares = calculate_target_shares(
            cash=Decimal("100000"),
            weight=Decimal("0.10"),
            price=Decimal("-50"),
        )
        assert shares == Decimal("0")


class TestValidateInitializationInputs:
    """Tests for the validate_initialization_inputs function."""

    def test_valid_inputs_return_no_errors(
        self,
        sample_portfolio_config: PortfolioConfig,
        sample_constituents: list[BenchmarkConstituent],
        sample_prices_jan: dict[str, PriceData],
    ):
        """Test that valid inputs produce no validation errors."""
        errors = validate_initialization_inputs(
            config=sample_portfolio_config,
            constituents=sample_constituents,
            prices=sample_prices_jan,
        )
        # May have warning about weights not summing to 1, but no blocking errors
        blocking_errors = [e for e in errors if "must be" in e.lower()]
        assert len(blocking_errors) == 0

    def test_zero_cash_returns_error(
        self,
        sample_constituents: list[BenchmarkConstituent],
        sample_prices_jan: dict[str, PriceData],
    ):
        """Test that zero cash produces a validation error."""
        config = PortfolioConfig(
            portfolio_id="TEST",
            cash=Decimal("0"),
            start_date=date(2024, 1, 2),
        )
        errors = validate_initialization_inputs(config, sample_constituents, sample_prices_jan)
        assert any("cash" in e.lower() for e in errors)

    def test_empty_constituents_returns_error(
        self,
        sample_portfolio_config: PortfolioConfig,
        sample_prices_jan: dict[str, PriceData],
    ):
        """Test that empty constituents produces a validation error."""
        errors = validate_initialization_inputs(
            sample_portfolio_config, [], sample_prices_jan
        )
        assert any("constituents" in e.lower() for e in errors)

    def test_empty_prices_returns_error(
        self,
        sample_portfolio_config: PortfolioConfig,
        sample_constituents: list[BenchmarkConstituent],
    ):
        """Test that empty prices produces a validation error."""
        errors = validate_initialization_inputs(
            sample_portfolio_config, sample_constituents, {}
        )
        assert any("price" in e.lower() for e in errors)
