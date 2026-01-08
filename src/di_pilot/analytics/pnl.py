"""
P&L (Profit and Loss) calculations for the Direct Indexing Shadow System.

This module provides functions for calculating realized and unrealized P&L,
with breakdowns by gain type (short-term vs long-term).
"""

from collections import defaultdict
from decimal import Decimal
from typing import Optional

from di_pilot.models import (
    TaxLot,
    LotValuation,
    GainType,
)


def calculate_unrealized_pnl_summary(
    valuations: list[LotValuation],
) -> dict[str, Decimal]:
    """
    Calculate summary unrealized P&L statistics.

    Args:
        valuations: List of lot valuations

    Returns:
        Dictionary with P&L summary:
        - total_unrealized_pnl: Total unrealized P&L
        - total_unrealized_gain: Sum of all gains
        - total_unrealized_loss: Sum of all losses (negative)
        - total_cost_basis: Total cost basis
        - total_market_value: Total market value
        - return_pct: Overall return percentage
    """
    total_pnl = Decimal("0")
    total_gain = Decimal("0")
    total_loss = Decimal("0")
    total_cost = Decimal("0")
    total_market_value = Decimal("0")

    for val in valuations:
        total_pnl += val.unrealized_pnl
        total_cost += val.lot.total_cost
        total_market_value += val.market_value

        if val.unrealized_pnl > Decimal("0"):
            total_gain += val.unrealized_pnl
        else:
            total_loss += val.unrealized_pnl

    return_pct = Decimal("0")
    if total_cost != Decimal("0"):
        return_pct = total_pnl / total_cost

    return {
        "total_unrealized_pnl": total_pnl,
        "total_unrealized_gain": total_gain,
        "total_unrealized_loss": total_loss,
        "total_cost_basis": total_cost,
        "total_market_value": total_market_value,
        "return_pct": return_pct,
    }


def calculate_pnl_by_gain_type(
    valuations: list[LotValuation],
) -> dict[str, dict[str, Decimal]]:
    """
    Calculate P&L broken down by short-term vs long-term.

    Args:
        valuations: List of lot valuations

    Returns:
        Dictionary with P&L by gain type:
        - short_term: {unrealized_gain, unrealized_loss, net_pnl}
        - long_term: {unrealized_gain, unrealized_loss, net_pnl}
    """
    result: dict[str, dict[str, Decimal]] = {
        "short_term": {
            "unrealized_gain": Decimal("0"),
            "unrealized_loss": Decimal("0"),
            "net_pnl": Decimal("0"),
            "cost_basis": Decimal("0"),
            "market_value": Decimal("0"),
        },
        "long_term": {
            "unrealized_gain": Decimal("0"),
            "unrealized_loss": Decimal("0"),
            "net_pnl": Decimal("0"),
            "cost_basis": Decimal("0"),
            "market_value": Decimal("0"),
        },
    }

    for val in valuations:
        key = "short_term" if val.gain_type == GainType.SHORT_TERM else "long_term"
        result[key]["net_pnl"] += val.unrealized_pnl
        result[key]["cost_basis"] += val.lot.total_cost
        result[key]["market_value"] += val.market_value

        if val.unrealized_pnl > Decimal("0"):
            result[key]["unrealized_gain"] += val.unrealized_pnl
        else:
            result[key]["unrealized_loss"] += val.unrealized_pnl

    return result


def calculate_pnl_by_symbol(
    valuations: list[LotValuation],
) -> dict[str, dict[str, Decimal]]:
    """
    Calculate P&L by symbol.

    Args:
        valuations: List of lot valuations

    Returns:
        Dictionary mapping symbol to P&L statistics
    """
    by_symbol: dict[str, dict[str, Decimal]] = defaultdict(
        lambda: {
            "unrealized_pnl": Decimal("0"),
            "cost_basis": Decimal("0"),
            "market_value": Decimal("0"),
            "shares": Decimal("0"),
        }
    )

    for val in valuations:
        symbol = val.lot.symbol
        by_symbol[symbol]["unrealized_pnl"] += val.unrealized_pnl
        by_symbol[symbol]["cost_basis"] += val.lot.total_cost
        by_symbol[symbol]["market_value"] += val.market_value
        by_symbol[symbol]["shares"] += val.lot.shares

    # Calculate return percentages
    for symbol, data in by_symbol.items():
        if data["cost_basis"] != Decimal("0"):
            data["return_pct"] = data["unrealized_pnl"] / data["cost_basis"]
        else:
            data["return_pct"] = Decimal("0")

    return dict(by_symbol)


def calculate_realized_pnl(
    sold_lots: list[tuple[TaxLot, Decimal]],
) -> dict[str, Decimal]:
    """
    Calculate realized P&L from sold lots.

    Note: v0.1 does not track realized trades, so this function
    is provided for future use.

    Args:
        sold_lots: List of (TaxLot, sale_price_per_share) tuples

    Returns:
        Dictionary with realized P&L summary
    """
    total_realized = Decimal("0")
    short_term_realized = Decimal("0")
    long_term_realized = Decimal("0")
    total_proceeds = Decimal("0")
    total_cost = Decimal("0")

    for lot, sale_price in sold_lots:
        proceeds = lot.shares * sale_price
        cost = lot.total_cost
        realized = proceeds - cost

        total_realized += realized
        total_proceeds += proceeds
        total_cost += cost

        # Would need sale_date to determine gain type properly
        # For now, assume all are short-term
        short_term_realized += realized

    return {
        "total_realized_pnl": total_realized,
        "short_term_realized": short_term_realized,
        "long_term_realized": long_term_realized,
        "total_proceeds": total_proceeds,
        "total_cost_basis": total_cost,
    }


def get_top_contributors(
    valuations: list[LotValuation],
    n: int = 10,
    by: str = "absolute",
) -> list[LotValuation]:
    """
    Get top P&L contributors.

    Args:
        valuations: List of lot valuations
        n: Number of top contributors to return
        by: Sort by "absolute" (dollar amount) or "percentage"

    Returns:
        Top n contributors by P&L
    """
    if by == "percentage":
        sorted_vals = sorted(
            valuations,
            key=lambda v: v.unrealized_pnl_pct,
            reverse=True,
        )
    else:
        sorted_vals = sorted(
            valuations,
            key=lambda v: v.unrealized_pnl,
            reverse=True,
        )

    return sorted_vals[:n]


def get_bottom_contributors(
    valuations: list[LotValuation],
    n: int = 10,
    by: str = "absolute",
) -> list[LotValuation]:
    """
    Get bottom P&L contributors (largest losses).

    Args:
        valuations: List of lot valuations
        n: Number of bottom contributors to return
        by: Sort by "absolute" (dollar amount) or "percentage"

    Returns:
        Bottom n contributors by P&L (biggest losers)
    """
    if by == "percentage":
        sorted_vals = sorted(
            valuations,
            key=lambda v: v.unrealized_pnl_pct,
        )
    else:
        sorted_vals = sorted(
            valuations,
            key=lambda v: v.unrealized_pnl,
        )

    return sorted_vals[:n]


def calculate_win_rate(
    valuations: list[LotValuation],
) -> dict[str, Decimal]:
    """
    Calculate win/loss statistics.

    Args:
        valuations: List of lot valuations

    Returns:
        Dictionary with win rate statistics
    """
    if not valuations:
        return {
            "win_count": 0,
            "loss_count": 0,
            "breakeven_count": 0,
            "win_rate": Decimal("0"),
            "avg_win": Decimal("0"),
            "avg_loss": Decimal("0"),
        }

    winners = [v for v in valuations if v.unrealized_pnl > Decimal("0")]
    losers = [v for v in valuations if v.unrealized_pnl < Decimal("0")]
    breakeven = [v for v in valuations if v.unrealized_pnl == Decimal("0")]

    win_rate = Decimal(len(winners)) / Decimal(len(valuations))

    avg_win = Decimal("0")
    if winners:
        avg_win = sum(v.unrealized_pnl for v in winners) / Decimal(len(winners))

    avg_loss = Decimal("0")
    if losers:
        avg_loss = sum(v.unrealized_pnl for v in losers) / Decimal(len(losers))

    return {
        "win_count": len(winners),
        "loss_count": len(losers),
        "breakeven_count": len(breakeven),
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
    }
