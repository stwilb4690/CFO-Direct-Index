"""
Holdings and lot-level tracking for the Direct Indexing Shadow System.

Provides utilities for aggregating and analyzing tax lots.
"""

from collections import defaultdict
from decimal import Decimal
from typing import Optional

from di_pilot.models import (
    TaxLot,
    PositionSummary,
    LotValuation,
)


def aggregate_holdings_by_symbol(
    lots: list[TaxLot],
) -> dict[str, list[TaxLot]]:
    """
    Group tax lots by symbol.

    Args:
        lots: List of tax lots

    Returns:
        Dictionary mapping symbol to list of lots for that symbol
    """
    holdings: dict[str, list[TaxLot]] = defaultdict(list)
    for lot in lots:
        holdings[lot.symbol].append(lot)
    return dict(holdings)


def get_portfolio_symbols(lots: list[TaxLot]) -> set[str]:
    """
    Get unique symbols in a portfolio.

    Args:
        lots: List of tax lots

    Returns:
        Set of unique symbols
    """
    return {lot.symbol for lot in lots}


def calculate_total_shares(lots: list[TaxLot], symbol: str) -> Decimal:
    """
    Calculate total shares for a symbol across all lots.

    Args:
        lots: List of tax lots
        symbol: Symbol to calculate shares for

    Returns:
        Total shares for the symbol
    """
    return sum(
        lot.shares for lot in lots if lot.symbol == symbol
    )


def calculate_total_cost_basis(lots: list[TaxLot], symbol: Optional[str] = None) -> Decimal:
    """
    Calculate total cost basis for lots.

    Args:
        lots: List of tax lots
        symbol: Optional symbol to filter by

    Returns:
        Total cost basis
    """
    if symbol:
        filtered_lots = [lot for lot in lots if lot.symbol == symbol]
    else:
        filtered_lots = lots

    return sum(lot.total_cost for lot in filtered_lots)


def calculate_position_weights(
    valuations: list[LotValuation],
    total_portfolio_value: Optional[Decimal] = None,
) -> dict[str, Decimal]:
    """
    Calculate current portfolio weights by symbol.

    Args:
        valuations: List of lot valuations
        total_portfolio_value: Optional pre-calculated total value

    Returns:
        Dictionary mapping symbol to weight (0-1)
    """
    if total_portfolio_value is None:
        total_portfolio_value = sum(v.market_value for v in valuations)

    if total_portfolio_value == Decimal("0"):
        return {}

    # Aggregate market value by symbol
    symbol_values: dict[str, Decimal] = defaultdict(Decimal)
    for val in valuations:
        symbol_values[val.lot.symbol] += val.market_value

    # Calculate weights
    weights = {}
    for symbol, value in symbol_values.items():
        weights[symbol] = value / total_portfolio_value

    return weights


def summarize_positions(
    valuations: list[LotValuation],
    total_portfolio_value: Optional[Decimal] = None,
) -> list[PositionSummary]:
    """
    Create position summaries aggregated by symbol.

    Args:
        valuations: List of lot valuations
        total_portfolio_value: Optional pre-calculated total value

    Returns:
        List of PositionSummary objects, one per symbol
    """
    if total_portfolio_value is None:
        total_portfolio_value = sum(v.market_value for v in valuations)

    # Aggregate by symbol
    symbol_data: dict[str, dict] = defaultdict(
        lambda: {
            "total_shares": Decimal("0"),
            "total_cost": Decimal("0"),
            "market_value": Decimal("0"),
            "num_lots": 0,
        }
    )

    for val in valuations:
        symbol = val.lot.symbol
        symbol_data[symbol]["total_shares"] += val.lot.shares
        symbol_data[symbol]["total_cost"] += val.lot.total_cost
        symbol_data[symbol]["market_value"] += val.market_value
        symbol_data[symbol]["num_lots"] += 1

    summaries = []
    for symbol, data in symbol_data.items():
        weight = Decimal("0")
        if total_portfolio_value > Decimal("0"):
            weight = data["market_value"] / total_portfolio_value

        summaries.append(
            PositionSummary(
                symbol=symbol,
                total_shares=data["total_shares"],
                total_cost=data["total_cost"],
                market_value=data["market_value"],
                unrealized_pnl=data["market_value"] - data["total_cost"],
                num_lots=data["num_lots"],
                current_weight=weight,
            )
        )

    # Sort by market value descending
    summaries.sort(key=lambda x: x.market_value, reverse=True)

    return summaries


def filter_lots_by_portfolio(
    lots: list[TaxLot],
    portfolio_id: str,
) -> list[TaxLot]:
    """
    Filter lots to a specific portfolio.

    Args:
        lots: List of tax lots
        portfolio_id: Portfolio ID to filter by

    Returns:
        Filtered list of lots
    """
    return [lot for lot in lots if lot.portfolio_id == portfolio_id]


def get_lots_for_symbol(
    lots: list[TaxLot],
    symbol: str,
) -> list[TaxLot]:
    """
    Get all lots for a specific symbol.

    Args:
        lots: List of tax lots
        symbol: Symbol to filter by

    Returns:
        List of lots for the symbol
    """
    return [lot for lot in lots if lot.symbol == symbol.upper()]
