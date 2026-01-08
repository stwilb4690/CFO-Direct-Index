"""
Tests for portfolio valuation functionality.
"""

from datetime import date
from decimal import Decimal

import pytest

from di_pilot.models import GainType, TaxLot, PriceData
from di_pilot.portfolio.valuation import (
    value_lots,
    value_portfolio,
    calculate_unrealized_pnl,
    determine_gain_type,
    ValuationError,
)


class TestValueLots:
    """Tests for the value_lots function."""

    def test_basic_valuation(self, sample_lots: list[TaxLot], sample_prices_jan: dict[str, PriceData]):
        """Test basic lot valuation."""
        valuations = value_lots(
            lots=sample_lots,
            prices=sample_prices_jan,
            valuation_date=date(2024, 1, 2),
        )

        assert len(valuations) == len(sample_lots)

        for val in valuations:
            assert val.market_value > Decimal("0")
            assert val.current_price > Decimal("0")

    def test_valuation_with_gains(self, sample_lots: list[TaxLot], sample_prices_jun: dict[str, PriceData]):
        """Test valuation shows gains when prices increase."""
        valuations = value_lots(
            lots=sample_lots,
            prices=sample_prices_jun,
            valuation_date=date(2024, 6, 15),
        )

        # AAPL should have gain (185.64 -> 214.29)
        aapl_vals = [v for v in valuations if v.lot.symbol == "AAPL"]
        assert len(aapl_vals) == 1
        assert aapl_vals[0].unrealized_pnl > Decimal("0")
        assert aapl_vals[0].unrealized_pnl_pct > Decimal("0")

    def test_valuation_with_losses(self, sample_lots: list[TaxLot], sample_prices_jun: dict[str, PriceData]):
        """Test valuation shows losses when prices decrease."""
        valuations = value_lots(
            lots=sample_lots,
            prices=sample_prices_jun,
            valuation_date=date(2024, 6, 15),
        )

        # TSLA should have loss (248.48 -> 182.47)
        tsla_vals = [v for v in valuations if v.lot.symbol == "TSLA"]
        assert len(tsla_vals) == 1
        assert tsla_vals[0].unrealized_pnl < Decimal("0")
        assert tsla_vals[0].unrealized_pnl_pct < Decimal("0")

    def test_valuation_missing_prices_raises_error(self, sample_lots: list[TaxLot]):
        """Test that missing prices raises ValuationError."""
        # Only provide prices for some symbols
        partial_prices = {
            "AAPL": PriceData(symbol="AAPL", date=date(2024, 1, 2), close=Decimal("185.64")),
        }

        with pytest.raises(ValuationError):
            value_lots(
                lots=sample_lots,
                prices=partial_prices,
                valuation_date=date(2024, 1, 2),
            )

    def test_gain_type_short_term(self, sample_lots: list[TaxLot], sample_prices_jun: dict[str, PriceData]):
        """Test that positions held <= 1 year are short-term."""
        valuations = value_lots(
            lots=sample_lots,
            prices=sample_prices_jun,
            valuation_date=date(2024, 6, 15),  # ~5 months from acquisition
        )

        for val in valuations:
            assert val.gain_type == GainType.SHORT_TERM


class TestValuePortfolio:
    """Tests for the value_portfolio function."""

    def test_portfolio_valuation_totals(
        self,
        sample_lots: list[TaxLot],
        sample_prices_jan: dict[str, PriceData],
    ):
        """Test that portfolio valuation calculates correct totals."""
        valuation = value_portfolio(
            lots=sample_lots,
            prices=sample_prices_jan,
            valuation_date=date(2024, 1, 2),
            portfolio_id="TEST001",
        )

        assert valuation.portfolio_id == "TEST001"
        assert valuation.valuation_date == date(2024, 1, 2)

        # Total market value should be sum of lot market values
        lot_total = sum(v.market_value for v in valuation.lot_valuations)
        assert valuation.total_market_value == lot_total

        # Total cost basis should be sum of lot cost bases
        cost_total = sum(v.lot.total_cost for v in valuation.lot_valuations)
        assert valuation.total_cost_basis == cost_total

    def test_portfolio_valuation_with_cash(
        self,
        sample_lots: list[TaxLot],
        sample_prices_jan: dict[str, PriceData],
    ):
        """Test portfolio valuation includes cash balance."""
        cash_balance = Decimal("50000")
        valuation = value_portfolio(
            lots=sample_lots,
            prices=sample_prices_jan,
            valuation_date=date(2024, 1, 2),
            portfolio_id="TEST001",
            cash_balance=cash_balance,
        )

        # Total should include cash
        lot_total = sum(v.market_value for v in valuation.lot_valuations)
        assert valuation.total_market_value == lot_total + cash_balance
        assert valuation.cash_balance == cash_balance

    def test_empty_portfolio_valuation(self, sample_prices_jan: dict[str, PriceData]):
        """Test valuation of empty portfolio."""
        valuation = value_portfolio(
            lots=[],
            prices=sample_prices_jan,
            valuation_date=date(2024, 1, 2),
            portfolio_id="EMPTY001",
        )

        assert valuation.total_market_value == Decimal("0")
        assert valuation.total_cost_basis == Decimal("0")
        assert len(valuation.lot_valuations) == 0

    def test_position_summaries_generated(
        self,
        sample_lots: list[TaxLot],
        sample_prices_jan: dict[str, PriceData],
    ):
        """Test that position summaries are generated."""
        valuation = value_portfolio(
            lots=sample_lots,
            prices=sample_prices_jan,
            valuation_date=date(2024, 1, 2),
            portfolio_id="TEST001",
        )

        # Should have one summary per unique symbol
        unique_symbols = len(set(lot.symbol for lot in sample_lots))
        assert len(valuation.position_summaries) == unique_symbols

        # Each summary should have valid data
        for summary in valuation.position_summaries:
            assert summary.total_shares > Decimal("0")
            assert summary.market_value > Decimal("0")


class TestCalculateUnrealizedPnl:
    """Tests for the calculate_unrealized_pnl function."""

    def test_gain_calculation(self):
        """Test unrealized gain calculation."""
        lot = TaxLot(
            lot_id="test",
            symbol="TEST",
            shares=Decimal("100"),
            cost_basis=Decimal("50"),  # $50/share = $5000 total
            acquisition_date=date(2024, 1, 1),
            portfolio_id="TEST",
        )

        pnl, pnl_pct = calculate_unrealized_pnl(lot, Decimal("60"))  # Now $60/share

        assert pnl == Decimal("1000")  # $6000 - $5000
        assert pnl_pct == Decimal("0.2")  # 20% gain

    def test_loss_calculation(self):
        """Test unrealized loss calculation."""
        lot = TaxLot(
            lot_id="test",
            symbol="TEST",
            shares=Decimal("100"),
            cost_basis=Decimal("50"),  # $50/share = $5000 total
            acquisition_date=date(2024, 1, 1),
            portfolio_id="TEST",
        )

        pnl, pnl_pct = calculate_unrealized_pnl(lot, Decimal("40"))  # Now $40/share

        assert pnl == Decimal("-1000")  # $4000 - $5000
        assert pnl_pct == Decimal("-0.2")  # 20% loss

    def test_breakeven(self):
        """Test breakeven (no P&L) calculation."""
        lot = TaxLot(
            lot_id="test",
            symbol="TEST",
            shares=Decimal("100"),
            cost_basis=Decimal("50"),
            acquisition_date=date(2024, 1, 1),
            portfolio_id="TEST",
        )

        pnl, pnl_pct = calculate_unrealized_pnl(lot, Decimal("50"))  # Same price

        assert pnl == Decimal("0")
        assert pnl_pct == Decimal("0")


class TestDetermineGainType:
    """Tests for the determine_gain_type function."""

    def test_short_term_under_year(self):
        """Test that positions held < 1 year are short-term."""
        gain_type = determine_gain_type(
            acquisition_date=date(2024, 1, 1),
            valuation_date=date(2024, 6, 1),  # ~5 months
        )
        assert gain_type == GainType.SHORT_TERM

    def test_short_term_at_year(self):
        """Test that positions held exactly 365 days are short-term."""
        gain_type = determine_gain_type(
            acquisition_date=date(2024, 1, 1),
            valuation_date=date(2024, 12, 31),  # 365 days
        )
        assert gain_type == GainType.SHORT_TERM

    def test_long_term_over_year(self):
        """Test that positions held > 1 year are long-term."""
        gain_type = determine_gain_type(
            acquisition_date=date(2024, 1, 1),
            valuation_date=date(2025, 1, 2),  # 366 days
        )
        assert gain_type == GainType.LONG_TERM

    def test_long_term_well_over_year(self):
        """Test that positions held well over 1 year are long-term."""
        gain_type = determine_gain_type(
            acquisition_date=date(2022, 1, 1),
            valuation_date=date(2024, 6, 1),  # ~2.5 years
        )
        assert gain_type == GainType.LONG_TERM
