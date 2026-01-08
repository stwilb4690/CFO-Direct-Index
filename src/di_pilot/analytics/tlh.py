"""
Tax-loss harvesting detection for the Direct Indexing Shadow System.

This module identifies lots with unrealized losses exceeding a configurable
threshold, generating candidates for tax-loss harvesting consideration.

Note: v0.1 does NOT implement:
- Wash-sale rule enforcement
- Automatic replacement trade generation
- Loss carryforward tracking
"""

from decimal import Decimal
from typing import Optional

from di_pilot.models import (
    TaxLot,
    LotValuation,
    TLHCandidate,
    GainType,
)


def identify_tlh_candidates(
    valuations: list[LotValuation],
    loss_threshold: Decimal = Decimal("0.03"),
    min_loss_amount: Decimal = Decimal("0"),
) -> list[TLHCandidate]:
    """
    Identify tax-loss harvesting candidates.

    A lot is a TLH candidate if its unrealized loss percentage exceeds
    the threshold (e.g., down more than 3% from cost basis).

    Args:
        valuations: List of lot valuations
        loss_threshold: Minimum loss percentage to flag (default 3%)
        min_loss_amount: Optional minimum dollar loss to consider

    Returns:
        List of TLHCandidate objects, sorted by loss magnitude (largest first)
    """
    candidates = []

    for val in valuations:
        # Check if lot has a loss
        if val.unrealized_pnl >= Decimal("0"):
            continue

        # Check if loss exceeds threshold
        # unrealized_pnl_pct is negative for losses
        if abs(val.unrealized_pnl_pct) < loss_threshold:
            continue

        # Check minimum loss amount
        loss_amount = abs(val.unrealized_pnl)
        if loss_amount < min_loss_amount:
            continue

        days_held = (val.valuation_date - val.lot.acquisition_date).days

        candidates.append(
            TLHCandidate(
                lot_valuation=val,
                loss_amount=val.unrealized_pnl,  # Negative value
                loss_pct=val.unrealized_pnl_pct,  # Negative value
                days_held=days_held,
                gain_type=val.gain_type,
            )
        )

    # Sort by loss amount (most negative first = largest losses)
    candidates.sort(key=lambda c: c.loss_amount)

    return candidates


def calculate_potential_harvest(
    candidates: list[TLHCandidate],
) -> dict[str, Decimal]:
    """
    Calculate potential tax-loss harvest amounts.

    Args:
        candidates: List of TLH candidates

    Returns:
        Dictionary with harvest summary:
        - total_loss: Total loss amount available
        - short_term_loss: Short-term losses
        - long_term_loss: Long-term losses
        - market_value: Total market value of candidate lots
    """
    total_loss = Decimal("0")
    short_term_loss = Decimal("0")
    long_term_loss = Decimal("0")
    market_value = Decimal("0")

    for cand in candidates:
        loss = abs(cand.loss_amount)
        total_loss += loss
        market_value += cand.lot_valuation.market_value

        if cand.gain_type == GainType.SHORT_TERM:
            short_term_loss += loss
        else:
            long_term_loss += loss

    return {
        "total_loss": total_loss,
        "short_term_loss": short_term_loss,
        "long_term_loss": long_term_loss,
        "market_value": market_value,
    }


def filter_candidates_by_gain_type(
    candidates: list[TLHCandidate],
    gain_type: GainType,
) -> list[TLHCandidate]:
    """
    Filter TLH candidates by gain type (short-term or long-term).

    Args:
        candidates: List of TLH candidates
        gain_type: GainType to filter by

    Returns:
        Filtered list of candidates
    """
    return [c for c in candidates if c.gain_type == gain_type]


def get_short_term_candidates(
    candidates: list[TLHCandidate],
) -> list[TLHCandidate]:
    """
    Get only short-term TLH candidates.

    Short-term losses are generally more tax-efficient as they can
    offset short-term gains (taxed at ordinary income rates).

    Args:
        candidates: List of TLH candidates

    Returns:
        Short-term candidates only
    """
    return filter_candidates_by_gain_type(candidates, GainType.SHORT_TERM)


def get_long_term_candidates(
    candidates: list[TLHCandidate],
) -> list[TLHCandidate]:
    """
    Get only long-term TLH candidates.

    Args:
        candidates: List of TLH candidates

    Returns:
        Long-term candidates only
    """
    return filter_candidates_by_gain_type(candidates, GainType.LONG_TERM)


def group_candidates_by_symbol(
    candidates: list[TLHCandidate],
) -> dict[str, list[TLHCandidate]]:
    """
    Group TLH candidates by symbol.

    Useful for analyzing which positions have the most harvesting potential.

    Args:
        candidates: List of TLH candidates

    Returns:
        Dictionary mapping symbol to list of candidates
    """
    grouped: dict[str, list[TLHCandidate]] = {}
    for cand in candidates:
        symbol = cand.lot_valuation.lot.symbol
        if symbol not in grouped:
            grouped[symbol] = []
        grouped[symbol].append(cand)

    return grouped


def calculate_harvest_by_symbol(
    candidates: list[TLHCandidate],
) -> dict[str, Decimal]:
    """
    Calculate potential harvest amount by symbol.

    Args:
        candidates: List of TLH candidates

    Returns:
        Dictionary mapping symbol to total loss amount
    """
    grouped = group_candidates_by_symbol(candidates)
    return {
        symbol: sum(abs(c.loss_amount) for c in cands)
        for symbol, cands in grouped.items()
    }


def rank_symbols_by_harvest_potential(
    candidates: list[TLHCandidate],
) -> list[tuple[str, Decimal]]:
    """
    Rank symbols by total harvest potential.

    Args:
        candidates: List of TLH candidates

    Returns:
        List of (symbol, total_loss) tuples, sorted by loss descending
    """
    by_symbol = calculate_harvest_by_symbol(candidates)
    ranked = sorted(by_symbol.items(), key=lambda x: x[1], reverse=True)
    return ranked


def summarize_tlh_candidates(
    candidates: list[TLHCandidate],
) -> dict:
    """
    Generate summary statistics for TLH candidates.

    Args:
        candidates: List of TLH candidates

    Returns:
        Dictionary with summary statistics
    """
    if not candidates:
        return {
            "candidate_count": 0,
            "total_potential_harvest": Decimal("0"),
            "short_term_harvest": Decimal("0"),
            "long_term_harvest": Decimal("0"),
            "total_market_value": Decimal("0"),
            "symbols_with_candidates": 0,
            "avg_loss_pct": Decimal("0"),
        }

    harvest = calculate_potential_harvest(candidates)
    symbols = set(c.lot_valuation.lot.symbol for c in candidates)
    avg_loss_pct = sum(c.loss_pct for c in candidates) / len(candidates)

    return {
        "candidate_count": len(candidates),
        "total_potential_harvest": harvest["total_loss"],
        "short_term_harvest": harvest["short_term_loss"],
        "long_term_harvest": harvest["long_term_loss"],
        "total_market_value": harvest["market_value"],
        "symbols_with_candidates": len(symbols),
        "avg_loss_pct": avg_loss_pct,
    }
