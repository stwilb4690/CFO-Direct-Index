"""
Portfolio valuation and mark-to-market calculations.

This module handles valuation of tax lots using current market prices,
calculating unrealized P&L and portfolio totals.
"""

from datetime import date
from decimal import Decimal
from typing import Optional

from di_pilot.models import (
    TaxLot,
    LotValuation,
    PortfolioValuation,
    PriceData,
    GainType,
)
from di_pilot.portfolio.holdings import summarize_positions


class ValuationError(Exception):
    """Raised when valuation cannot be completed."""
    pass


def value_lots(
    lots: list[TaxLot],
    prices: dict[str, PriceData],
    valuation_date: date,
) -> list[LotValuation]:
    """
    Value a list of tax lots at current prices.

    Args:
        lots: List of tax lots to value
        prices: Current prices by symbol
        valuation_date: Date of valuation

    Returns:
        List of LotValuation objects

    Raises:
        ValuationError: If prices are missing for any lots
    """
    valuations = []
    missing_prices = []

    for lot in lots:
        if lot.symbol not in prices:
            missing_prices.append(lot.symbol)
            continue

        price_data = prices[lot.symbol]
        valuation = LotValuation.from_lot(
            lot=lot,
            current_price=price_data.close,
            valuation_date=valuation_date,
        )
        valuations.append(valuation)

    if missing_prices:
        unique_missing = list(set(missing_prices))
        raise ValuationError(
            f"Missing prices for symbols: {unique_missing}"
        )

    return valuations


def value_portfolio(
    lots: list[TaxLot],
    prices: dict[str, PriceData],
    valuation_date: date,
    portfolio_id: str,
    cash_balance: Decimal = Decimal("0"),
) -> PortfolioValuation:
    """
    Create a complete portfolio valuation.

    Args:
        lots: List of tax lots
        prices: Current prices by symbol
        valuation_date: Date of valuation
        portfolio_id: Portfolio identifier
        cash_balance: Any uninvested cash

    Returns:
        PortfolioValuation with all lot valuations and summaries

    Raises:
        ValuationError: If valuation cannot be completed
    """
    # Filter to portfolio
    portfolio_lots = [lot for lot in lots if lot.portfolio_id == portfolio_id]

    if not portfolio_lots:
        # Return empty valuation
        return PortfolioValuation(
            portfolio_id=portfolio_id,
            valuation_date=valuation_date,
            total_market_value=cash_balance,
            total_cost_basis=Decimal("0"),
            total_unrealized_pnl=Decimal("0"),
            cash_balance=cash_balance,
            lot_valuations=[],
            position_summaries=[],
        )

    # Value all lots
    lot_valuations = value_lots(portfolio_lots, prices, valuation_date)

    # Calculate totals
    total_market_value = sum(v.market_value for v in lot_valuations) + cash_balance
    total_cost_basis = sum(v.lot.total_cost for v in lot_valuations)
    total_unrealized_pnl = sum(v.unrealized_pnl for v in lot_valuations)

    # Generate position summaries
    position_summaries = summarize_positions(lot_valuations, total_market_value)

    return PortfolioValuation(
        portfolio_id=portfolio_id,
        valuation_date=valuation_date,
        total_market_value=total_market_value,
        total_cost_basis=total_cost_basis,
        total_unrealized_pnl=total_unrealized_pnl,
        cash_balance=cash_balance,
        lot_valuations=lot_valuations,
        position_summaries=position_summaries,
    )


def calculate_unrealized_pnl(
    lot: TaxLot,
    current_price: Decimal,
) -> tuple[Decimal, Decimal]:
    """
    Calculate unrealized P&L for a single lot.

    Args:
        lot: Tax lot to evaluate
        current_price: Current market price

    Returns:
        Tuple of (unrealized_pnl_amount, unrealized_pnl_pct)
    """
    market_value = lot.shares * current_price
    cost_basis = lot.total_cost
    unrealized_pnl = market_value - cost_basis

    if cost_basis != Decimal("0"):
        unrealized_pnl_pct = unrealized_pnl / cost_basis
    else:
        unrealized_pnl_pct = Decimal("0")

    return unrealized_pnl, unrealized_pnl_pct


def determine_gain_type(
    acquisition_date: date,
    valuation_date: date,
) -> GainType:
    """
    Determine if a position is short-term or long-term.

    Args:
        acquisition_date: Date lot was acquired
        valuation_date: Current valuation date

    Returns:
        GainType.SHORT_TERM if held <= 365 days, else LONG_TERM
    """
    days_held = (valuation_date - acquisition_date).days
    return GainType.LONG_TERM if days_held > 365 else GainType.SHORT_TERM


def calculate_portfolio_return(
    valuation: PortfolioValuation,
) -> Decimal:
    """
    Calculate simple portfolio return.

    Args:
        valuation: Portfolio valuation

    Returns:
        Return as decimal (e.g., 0.05 for 5%)
    """
    if valuation.total_cost_basis == Decimal("0"):
        return Decimal("0")

    return valuation.total_unrealized_pnl / valuation.total_cost_basis


def get_gainers_and_losers(
    valuations: list[LotValuation],
    top_n: int = 10,
) -> tuple[list[LotValuation], list[LotValuation]]:
    """
    Get top gainers and losers by unrealized P&L percentage.

    Args:
        valuations: List of lot valuations
        top_n: Number of top/bottom positions to return

    Returns:
        Tuple of (top_gainers, top_losers) sorted by P&L %
    """
    sorted_by_pnl = sorted(
        valuations,
        key=lambda v: v.unrealized_pnl_pct,
        reverse=True,
    )

    top_gainers = sorted_by_pnl[:top_n]
    top_losers = sorted_by_pnl[-top_n:][::-1]  # Reverse to show worst first

    return top_gainers, top_losers
