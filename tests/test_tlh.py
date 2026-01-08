"""
Tests for tax-loss harvesting detection functionality.
"""

from datetime import date
from decimal import Decimal

import pytest

from di_pilot.models import GainType, TaxLot, LotValuation, TLHCandidate
from di_pilot.analytics.tlh import (
    identify_tlh_candidates,
    calculate_potential_harvest,
    filter_candidates_by_gain_type,
    get_short_term_candidates,
    get_long_term_candidates,
    group_candidates_by_symbol,
    rank_symbols_by_harvest_potential,
    summarize_tlh_candidates,
)


@pytest.fixture
def sample_valuations_with_losses() -> list[LotValuation]:
    """Create sample valuations with various gains and losses."""
    # Create lots first
    lots = [
        TaxLot(
            lot_id="lot-001",
            symbol="AAPL",
            shares=Decimal("100"),
            cost_basis=Decimal("200"),  # $200/share
            acquisition_date=date(2024, 1, 2),
            portfolio_id="TEST001",
        ),
        TaxLot(
            lot_id="lot-002",
            symbol="MSFT",
            shares=Decimal("50"),
            cost_basis=Decimal("400"),  # $400/share
            acquisition_date=date(2024, 1, 2),
            portfolio_id="TEST001",
        ),
        TaxLot(
            lot_id="lot-003",
            symbol="TSLA",
            shares=Decimal("75"),
            cost_basis=Decimal("250"),  # $250/share = $18,750
            acquisition_date=date(2024, 1, 2),
            portfolio_id="TEST001",
        ),
        TaxLot(
            lot_id="lot-004",
            symbol="NVDA",
            shares=Decimal("20"),
            cost_basis=Decimal("500"),  # $500/share = $10,000
            acquisition_date=date(2023, 1, 2),  # Long-term
            portfolio_id="TEST001",
        ),
    ]

    # Create valuations at current prices
    valuation_date = date(2024, 6, 15)
    return [
        # AAPL: $200 -> $220 = +10% (gain, not a TLH candidate)
        LotValuation(
            lot=lots[0],
            current_price=Decimal("220"),
            valuation_date=valuation_date,
            market_value=Decimal("22000"),
            unrealized_pnl=Decimal("2000"),
            unrealized_pnl_pct=Decimal("0.10"),
            gain_type=GainType.SHORT_TERM,
        ),
        # MSFT: $400 -> $380 = -5% (loss, TLH candidate)
        LotValuation(
            lot=lots[1],
            current_price=Decimal("380"),
            valuation_date=valuation_date,
            market_value=Decimal("19000"),
            unrealized_pnl=Decimal("-1000"),
            unrealized_pnl_pct=Decimal("-0.05"),
            gain_type=GainType.SHORT_TERM,
        ),
        # TSLA: $250 -> $180 = -28% (big loss, TLH candidate)
        LotValuation(
            lot=lots[2],
            current_price=Decimal("180"),
            valuation_date=valuation_date,
            market_value=Decimal("13500"),
            unrealized_pnl=Decimal("-5250"),
            unrealized_pnl_pct=Decimal("-0.28"),
            gain_type=GainType.SHORT_TERM,
        ),
        # NVDA: $500 -> $450 = -10% (loss, long-term TLH candidate)
        LotValuation(
            lot=lots[3],
            current_price=Decimal("450"),
            valuation_date=valuation_date,
            market_value=Decimal("9000"),
            unrealized_pnl=Decimal("-1000"),
            unrealized_pnl_pct=Decimal("-0.10"),
            gain_type=GainType.LONG_TERM,
        ),
    ]


class TestIdentifyTLHCandidates:
    """Tests for the identify_tlh_candidates function."""

    def test_identifies_losses_above_threshold(
        self, sample_valuations_with_losses: list[LotValuation]
    ):
        """Test that losses above threshold are identified."""
        candidates = identify_tlh_candidates(
            valuations=sample_valuations_with_losses,
            loss_threshold=Decimal("0.03"),  # 3% threshold
        )

        # Should find MSFT (-5%), TSLA (-28%), NVDA (-10%)
        assert len(candidates) == 3

        symbols = {c.lot_valuation.lot.symbol for c in candidates}
        assert "MSFT" in symbols
        assert "TSLA" in symbols
        assert "NVDA" in symbols
        assert "AAPL" not in symbols  # AAPL has a gain

    def test_excludes_losses_below_threshold(
        self, sample_valuations_with_losses: list[LotValuation]
    ):
        """Test that losses below threshold are excluded."""
        candidates = identify_tlh_candidates(
            valuations=sample_valuations_with_losses,
            loss_threshold=Decimal("0.06"),  # 6% threshold
        )

        # Should only find TSLA (-28%) and NVDA (-10%)
        # MSFT (-5%) is below threshold
        assert len(candidates) == 2

        symbols = {c.lot_valuation.lot.symbol for c in candidates}
        assert "TSLA" in symbols
        assert "NVDA" in symbols
        assert "MSFT" not in symbols

    def test_sorted_by_loss_magnitude(
        self, sample_valuations_with_losses: list[LotValuation]
    ):
        """Test that candidates are sorted by loss magnitude."""
        candidates = identify_tlh_candidates(
            valuations=sample_valuations_with_losses,
            loss_threshold=Decimal("0.03"),
        )

        # First candidate should be TSLA (largest loss: $5,250)
        assert candidates[0].lot_valuation.lot.symbol == "TSLA"
        assert candidates[0].loss_amount == Decimal("-5250")

    def test_excludes_gains(self, sample_valuations_with_losses: list[LotValuation]):
        """Test that positions with gains are excluded."""
        candidates = identify_tlh_candidates(
            valuations=sample_valuations_with_losses,
            loss_threshold=Decimal("0.01"),  # Very low threshold
        )

        # AAPL has a gain, should not be included
        symbols = {c.lot_valuation.lot.symbol for c in candidates}
        assert "AAPL" not in symbols

    def test_min_loss_amount_filter(
        self, sample_valuations_with_losses: list[LotValuation]
    ):
        """Test minimum loss amount filter."""
        candidates = identify_tlh_candidates(
            valuations=sample_valuations_with_losses,
            loss_threshold=Decimal("0.03"),
            min_loss_amount=Decimal("2000"),  # Minimum $2000 loss
        )

        # Only TSLA has loss > $2000
        assert len(candidates) == 1
        assert candidates[0].lot_valuation.lot.symbol == "TSLA"


class TestCalculatePotentialHarvest:
    """Tests for the calculate_potential_harvest function."""

    def test_calculates_total_harvest(
        self, sample_valuations_with_losses: list[LotValuation]
    ):
        """Test calculation of total potential harvest."""
        candidates = identify_tlh_candidates(
            valuations=sample_valuations_with_losses,
            loss_threshold=Decimal("0.03"),
        )

        harvest = calculate_potential_harvest(candidates)

        # Total: MSFT $1000 + TSLA $5250 + NVDA $1000 = $7250
        assert harvest["total_loss"] == Decimal("7250")

    def test_separates_short_and_long_term(
        self, sample_valuations_with_losses: list[LotValuation]
    ):
        """Test separation of short-term and long-term losses."""
        candidates = identify_tlh_candidates(
            valuations=sample_valuations_with_losses,
            loss_threshold=Decimal("0.03"),
        )

        harvest = calculate_potential_harvest(candidates)

        # Short-term: MSFT $1000 + TSLA $5250 = $6250
        assert harvest["short_term_loss"] == Decimal("6250")

        # Long-term: NVDA $1000
        assert harvest["long_term_loss"] == Decimal("1000")


class TestFilterCandidatesByGainType:
    """Tests for the filter_candidates_by_gain_type function."""

    def test_filters_short_term(
        self, sample_valuations_with_losses: list[LotValuation]
    ):
        """Test filtering to short-term candidates only."""
        candidates = identify_tlh_candidates(
            valuations=sample_valuations_with_losses,
            loss_threshold=Decimal("0.03"),
        )

        short_term = filter_candidates_by_gain_type(candidates, GainType.SHORT_TERM)

        assert len(short_term) == 2
        for c in short_term:
            assert c.gain_type == GainType.SHORT_TERM

    def test_filters_long_term(
        self, sample_valuations_with_losses: list[LotValuation]
    ):
        """Test filtering to long-term candidates only."""
        candidates = identify_tlh_candidates(
            valuations=sample_valuations_with_losses,
            loss_threshold=Decimal("0.03"),
        )

        long_term = filter_candidates_by_gain_type(candidates, GainType.LONG_TERM)

        assert len(long_term) == 1
        assert long_term[0].lot_valuation.lot.symbol == "NVDA"


class TestGetShortTermCandidates:
    """Tests for the get_short_term_candidates function."""

    def test_gets_short_term_only(
        self, sample_valuations_with_losses: list[LotValuation]
    ):
        """Test getting short-term candidates."""
        candidates = identify_tlh_candidates(
            valuations=sample_valuations_with_losses,
            loss_threshold=Decimal("0.03"),
        )

        short_term = get_short_term_candidates(candidates)

        assert len(short_term) == 2
        symbols = {c.lot_valuation.lot.symbol for c in short_term}
        assert "MSFT" in symbols
        assert "TSLA" in symbols


class TestGroupCandidatesBySymbol:
    """Tests for the group_candidates_by_symbol function."""

    def test_groups_by_symbol(
        self, sample_valuations_with_losses: list[LotValuation]
    ):
        """Test grouping candidates by symbol."""
        candidates = identify_tlh_candidates(
            valuations=sample_valuations_with_losses,
            loss_threshold=Decimal("0.03"),
        )

        grouped = group_candidates_by_symbol(candidates)

        assert len(grouped) == 3  # MSFT, TSLA, NVDA
        assert "MSFT" in grouped
        assert "TSLA" in grouped
        assert "NVDA" in grouped


class TestRankSymbolsByHarvestPotential:
    """Tests for the rank_symbols_by_harvest_potential function."""

    def test_ranks_by_harvest_amount(
        self, sample_valuations_with_losses: list[LotValuation]
    ):
        """Test ranking symbols by harvest potential."""
        candidates = identify_tlh_candidates(
            valuations=sample_valuations_with_losses,
            loss_threshold=Decimal("0.03"),
        )

        ranked = rank_symbols_by_harvest_potential(candidates)

        # TSLA should be first (largest loss: $5250)
        assert ranked[0][0] == "TSLA"
        assert ranked[0][1] == Decimal("5250")


class TestSummarizeTLHCandidates:
    """Tests for the summarize_tlh_candidates function."""

    def test_summary_statistics(
        self, sample_valuations_with_losses: list[LotValuation]
    ):
        """Test that summary contains expected statistics."""
        candidates = identify_tlh_candidates(
            valuations=sample_valuations_with_losses,
            loss_threshold=Decimal("0.03"),
        )

        summary = summarize_tlh_candidates(candidates)

        assert summary["candidate_count"] == 3
        assert summary["total_potential_harvest"] == Decimal("7250")
        assert summary["short_term_harvest"] == Decimal("6250")
        assert summary["long_term_harvest"] == Decimal("1000")
        assert summary["symbols_with_candidates"] == 3

    def test_empty_candidates_summary(self):
        """Test summary with no candidates."""
        summary = summarize_tlh_candidates([])

        assert summary["candidate_count"] == 0
        assert summary["total_potential_harvest"] == Decimal("0")
