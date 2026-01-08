"""
Trade proposal generation for the Direct Indexing Shadow System.

This module generates proposed trades for:
- Rebalancing to reduce drift from benchmark
- Tax-loss harvesting sells

All trades are proposals only and require human review before execution.
"""

from decimal import Decimal, ROUND_DOWN
from typing import Optional

from di_pilot.models import (
    DriftAnalysis,
    LotValuation,
    PriceData,
    TaxLot,
    TLHCandidate,
    TradeProposal,
    TradeSide,
    TradeRationale,
    PortfolioValuation,
)


def generate_rebalance_proposals(
    drift_analyses: list[DriftAnalysis],
    portfolio_value: Decimal,
    prices: dict[str, PriceData],
    portfolio_id: str,
    min_trade_value: Decimal = Decimal("100"),
    max_turnover_pct: Decimal = Decimal("0.05"),
) -> list[TradeProposal]:
    """
    Generate trade proposals to reduce portfolio drift.

    Creates buy proposals for underweight positions and sell proposals
    for overweight positions.

    Args:
        drift_analyses: List of drift analyses with threshold flags
        portfolio_value: Total portfolio value
        prices: Current prices by symbol
        portfolio_id: Portfolio identifier
        min_trade_value: Minimum trade value to propose
        max_turnover_pct: Maximum portfolio turnover per rebalance

    Returns:
        List of TradeProposal objects
    """
    proposals = []
    total_trade_value = Decimal("0")
    max_trade_value = portfolio_value * max_turnover_pct

    # Process positions exceeding threshold
    for drift in drift_analyses:
        if not drift.exceeds_threshold:
            continue

        if drift.symbol not in prices:
            continue

        price = prices[drift.symbol].close
        if price <= Decimal("0"):
            continue

        # Calculate trade value to bring back to target
        trade_value = abs(drift.absolute_drift) * portfolio_value

        # Check minimum trade value
        if trade_value < min_trade_value:
            continue

        # Check turnover limit
        if total_trade_value + trade_value > max_trade_value:
            # Reduce trade size to fit within limit
            remaining_capacity = max_trade_value - total_trade_value
            if remaining_capacity < min_trade_value:
                break
            trade_value = remaining_capacity

        total_trade_value += trade_value

        # Calculate shares
        shares = (trade_value / price).quantize(
            Decimal("0.000001"), rounding=ROUND_DOWN
        )

        if shares <= Decimal("0"):
            continue

        # Determine side based on drift direction
        if drift.absolute_drift > Decimal("0"):
            # Overweight - sell
            side = TradeSide.SELL
            rationale_detail = (
                f"Position overweight by {drift.absolute_drift:.4%} "
                f"(current: {drift.current_weight:.4%}, target: {drift.target_weight:.4%}). "
                f"Selling to reduce drift."
            )
        else:
            # Underweight - buy
            side = TradeSide.BUY
            rationale_detail = (
                f"Position underweight by {abs(drift.absolute_drift):.4%} "
                f"(current: {drift.current_weight:.4%}, target: {drift.target_weight:.4%}). "
                f"Buying to reduce drift."
            )

        proposal = TradeProposal.create(
            portfolio_id=portfolio_id,
            symbol=drift.symbol,
            side=side,
            shares=shares,
            rationale=TradeRationale.DRIFT_CORRECTION,
            rationale_detail=rationale_detail,
            current_price=price,
        )
        proposals.append(proposal)

    return proposals


def generate_tlh_proposals(
    candidates: list[TLHCandidate],
    portfolio_id: str,
    max_harvest_value: Optional[Decimal] = None,
) -> list[TradeProposal]:
    """
    Generate sell proposals for tax-loss harvesting candidates.

    Note: v0.1 does not generate replacement buy trades.
    Wash-sale rules must be considered externally.

    Args:
        candidates: List of TLH candidates
        portfolio_id: Portfolio identifier
        max_harvest_value: Optional cap on total harvest value

    Returns:
        List of TradeProposal objects (sells only)
    """
    proposals = []
    total_harvest_value = Decimal("0")

    for cand in candidates:
        lot = cand.lot_valuation.lot
        current_price = cand.lot_valuation.current_price
        market_value = cand.lot_valuation.market_value

        # Check harvest cap
        if max_harvest_value is not None:
            if total_harvest_value + market_value > max_harvest_value:
                continue

        total_harvest_value += market_value

        rationale_detail = (
            f"Tax-loss harvesting candidate: {cand.loss_pct:.2%} unrealized loss "
            f"(${abs(cand.loss_amount):.2f}). "
            f"Held {cand.days_held} days ({cand.gain_type.value}). "
            f"Consider wash-sale implications before executing."
        )

        proposal = TradeProposal.create(
            portfolio_id=portfolio_id,
            symbol=lot.symbol,
            side=TradeSide.SELL,
            shares=lot.shares,
            rationale=TradeRationale.TAX_LOSS_HARVEST,
            rationale_detail=rationale_detail,
            current_price=current_price,
            lot_id=lot.lot_id,
        )
        proposals.append(proposal)

    return proposals


def generate_all_proposals(
    portfolio_valuation: PortfolioValuation,
    drift_analyses: list[DriftAnalysis],
    tlh_candidates: list[TLHCandidate],
    prices: dict[str, PriceData],
    min_trade_value: Decimal = Decimal("100"),
    max_turnover_pct: Decimal = Decimal("0.05"),
    max_harvest_value: Optional[Decimal] = None,
) -> list[TradeProposal]:
    """
    Generate all trade proposals (rebalance + TLH).

    Coordinates rebalance and TLH proposals to avoid conflicts.
    TLH sells take priority over rebalance sells.

    Args:
        portfolio_valuation: Current portfolio valuation
        drift_analyses: Drift analysis results
        tlh_candidates: TLH candidates
        prices: Current prices
        min_trade_value: Minimum trade value
        max_turnover_pct: Maximum turnover for rebalancing
        max_harvest_value: Optional cap on TLH value

    Returns:
        Combined list of trade proposals
    """
    all_proposals = []
    portfolio_id = portfolio_valuation.portfolio_id
    portfolio_value = portfolio_valuation.total_market_value

    # Generate TLH proposals first (higher priority)
    tlh_proposals = generate_tlh_proposals(
        candidates=tlh_candidates,
        portfolio_id=portfolio_id,
        max_harvest_value=max_harvest_value,
    )
    all_proposals.extend(tlh_proposals)

    # Track symbols being sold for TLH to avoid conflicting rebalance proposals
    tlh_sell_symbols = {p.symbol for p in tlh_proposals}

    # Filter drift analyses to exclude TLH sell symbols
    filtered_drift = [
        d for d in drift_analyses
        if d.symbol not in tlh_sell_symbols or d.absolute_drift < Decimal("0")
    ]

    # Calculate remaining turnover capacity after TLH
    tlh_value = sum(p.estimated_value for p in tlh_proposals)
    remaining_turnover = (portfolio_value * max_turnover_pct) - tlh_value
    remaining_turnover_pct = remaining_turnover / portfolio_value if portfolio_value > 0 else Decimal("0")

    if remaining_turnover_pct > Decimal("0"):
        # Generate rebalance proposals with remaining capacity
        rebalance_proposals = generate_rebalance_proposals(
            drift_analyses=filtered_drift,
            portfolio_value=portfolio_value,
            prices=prices,
            portfolio_id=portfolio_id,
            min_trade_value=min_trade_value,
            max_turnover_pct=remaining_turnover_pct,
        )
        all_proposals.extend(rebalance_proposals)

    return all_proposals


def calculate_proposal_summary(
    proposals: list[TradeProposal],
) -> dict:
    """
    Calculate summary statistics for trade proposals.

    Args:
        proposals: List of trade proposals

    Returns:
        Dictionary with summary statistics
    """
    if not proposals:
        return {
            "total_proposals": 0,
            "buy_count": 0,
            "sell_count": 0,
            "total_buy_value": Decimal("0"),
            "total_sell_value": Decimal("0"),
            "net_trade_value": Decimal("0"),
            "tlh_proposals": 0,
            "rebalance_proposals": 0,
        }

    buys = [p for p in proposals if p.side == TradeSide.BUY]
    sells = [p for p in proposals if p.side == TradeSide.SELL]
    tlh = [p for p in proposals if p.rationale == TradeRationale.TAX_LOSS_HARVEST]
    rebalance = [p for p in proposals if p.rationale in (
        TradeRationale.DRIFT_CORRECTION, TradeRationale.REBALANCE
    )]

    total_buy_value = sum(p.estimated_value for p in buys)
    total_sell_value = sum(p.estimated_value for p in sells)

    return {
        "total_proposals": len(proposals),
        "buy_count": len(buys),
        "sell_count": len(sells),
        "total_buy_value": total_buy_value,
        "total_sell_value": total_sell_value,
        "net_trade_value": total_buy_value - total_sell_value,
        "tlh_proposals": len(tlh),
        "rebalance_proposals": len(rebalance),
    }


def validate_proposals(
    proposals: list[TradeProposal],
    holdings: list[TaxLot],
) -> list[str]:
    """
    Validate trade proposals against current holdings.

    Args:
        proposals: List of trade proposals
        holdings: Current holdings (tax lots)

    Returns:
        List of validation warning messages
    """
    warnings = []

    # Build holdings lookup
    holdings_by_symbol: dict[str, Decimal] = {}
    for lot in holdings:
        holdings_by_symbol[lot.symbol] = holdings_by_symbol.get(
            lot.symbol, Decimal("0")
        ) + lot.shares

    # Check sell proposals
    for proposal in proposals:
        if proposal.side != TradeSide.SELL:
            continue

        current_shares = holdings_by_symbol.get(proposal.symbol, Decimal("0"))
        if proposal.shares > current_shares:
            warnings.append(
                f"Sell proposal for {proposal.symbol} ({proposal.shares} shares) "
                f"exceeds current holdings ({current_shares} shares)"
            )

    # Check for duplicate symbol trades
    symbol_counts: dict[str, int] = {}
    for proposal in proposals:
        key = f"{proposal.symbol}_{proposal.side.value}"
        symbol_counts[key] = symbol_counts.get(key, 0) + 1

    for key, count in symbol_counts.items():
        if count > 1:
            warnings.append(f"Multiple proposals for same symbol/side: {key}")

    return warnings
