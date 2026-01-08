"""
Portfolio initialization from cash.

This module handles the initial deployment of cash into S&P 500 constituents
according to market-cap weights, creating tax lots for each position.
"""

from datetime import date
from decimal import Decimal, ROUND_DOWN
from typing import Optional

from di_pilot.models import (
    BenchmarkConstituent,
    PriceData,
    PortfolioConfig,
    TaxLot,
    TradeProposal,
    TradeSide,
    TradeRationale,
)


class InitializationError(Exception):
    """Raised when portfolio initialization fails."""
    pass


def initialize_portfolio(
    config: PortfolioConfig,
    constituents: list[BenchmarkConstituent],
    prices: dict[str, PriceData],
    min_position_value: Decimal = Decimal("10"),
) -> tuple[list[TaxLot], list[TradeProposal]]:
    """
    Initialize a portfolio from cash into S&P 500 constituents.

    Allocates cash according to benchmark weights, creating one tax lot
    per position. Supports fractional shares tracked to 6 decimal places.

    Args:
        config: Portfolio configuration with cash and start_date
        constituents: S&P 500 constituents with weights
        prices: Price data for the start date, keyed by symbol
        min_position_value: Minimum position value to create (default $10)

    Returns:
        Tuple of (list of TaxLot, list of TradeProposal for initial purchases)

    Raises:
        InitializationError: If initialization fails (missing prices, etc.)
    """
    if not constituents:
        raise InitializationError("No benchmark constituents provided")

    if not prices:
        raise InitializationError("No price data provided")

    cash = config.cash
    start_date = config.start_date
    portfolio_id = config.portfolio_id

    # Validate constituents have prices
    missing_prices = []
    for constituent in constituents:
        if constituent.symbol not in prices:
            missing_prices.append(constituent.symbol)

    if missing_prices:
        # Log warning but continue with available symbols
        available_weight = sum(
            c.weight for c in constituents if c.symbol in prices
        )
        if available_weight == Decimal("0"):
            raise InitializationError(
                f"No price data available for any constituents. "
                f"Missing: {missing_prices[:10]}..."
            )

    # Filter to constituents with prices and renormalize weights
    valid_constituents = [c for c in constituents if c.symbol in prices]
    total_weight = sum(c.weight for c in valid_constituents)

    if total_weight == Decimal("0"):
        raise InitializationError("Total weight of valid constituents is zero")

    # Renormalize weights
    normalized_constituents = []
    for c in valid_constituents:
        normalized_constituents.append(
            BenchmarkConstituent(
                symbol=c.symbol,
                weight=c.weight / total_weight,
                as_of_date=c.as_of_date,
            )
        )

    # Calculate target allocation for each symbol
    lots = []
    proposals = []
    allocated_cash = Decimal("0")

    for constituent in normalized_constituents:
        target_value = cash * constituent.weight

        # Skip positions below minimum value
        if target_value < min_position_value:
            continue

        price = prices[constituent.symbol].close
        if price <= Decimal("0"):
            continue

        # Calculate shares (fractional, 6 decimal places)
        shares = (target_value / price).quantize(
            Decimal("0.000001"), rounding=ROUND_DOWN
        )

        if shares <= Decimal("0"):
            continue

        # Actual cost for this position
        actual_cost = shares * price
        allocated_cash += actual_cost

        # Create tax lot
        lot = TaxLot.create(
            symbol=constituent.symbol,
            shares=shares,
            cost_basis=price,
            acquisition_date=start_date,
            portfolio_id=portfolio_id,
        )
        lots.append(lot)

        # Create trade proposal for initial purchase
        proposal = TradeProposal.create(
            portfolio_id=portfolio_id,
            symbol=constituent.symbol,
            side=TradeSide.BUY,
            shares=shares,
            rationale=TradeRationale.INITIAL_PURCHASE,
            rationale_detail=(
                f"Initial purchase at {constituent.weight:.4%} target weight. "
                f"Deploying ${target_value:.2f} at ${price:.2f}/share."
            ),
            current_price=price,
        )
        proposals.append(proposal)

    if not lots:
        raise InitializationError(
            "No positions could be created. Check prices and minimum position value."
        )

    # Calculate remaining cash (due to rounding)
    remaining_cash = cash - allocated_cash

    return lots, proposals


def calculate_target_shares(
    cash: Decimal,
    weight: Decimal,
    price: Decimal,
    decimal_places: int = 6,
) -> Decimal:
    """
    Calculate target shares for a position.

    Args:
        cash: Total cash to allocate
        weight: Target weight (0-1)
        price: Current price per share
        decimal_places: Precision for fractional shares (default 6)

    Returns:
        Number of shares to purchase
    """
    if price <= Decimal("0"):
        return Decimal("0")

    target_value = cash * weight
    quantize_str = "0." + "0" * decimal_places
    shares = (target_value / price).quantize(
        Decimal(quantize_str), rounding=ROUND_DOWN
    )

    return shares


def validate_initialization_inputs(
    config: PortfolioConfig,
    constituents: list[BenchmarkConstituent],
    prices: dict[str, PriceData],
) -> list[str]:
    """
    Validate inputs for portfolio initialization.

    Args:
        config: Portfolio configuration
        constituents: Benchmark constituents
        prices: Price data

    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []

    if config.cash <= Decimal("0"):
        errors.append("Cash amount must be positive")

    if not constituents:
        errors.append("No benchmark constituents provided")

    if not prices:
        errors.append("No price data provided")

    # Check for duplicate symbols in constituents
    symbols = [c.symbol for c in constituents]
    duplicates = set([s for s in symbols if symbols.count(s) > 1])
    if duplicates:
        errors.append(f"Duplicate symbols in constituents: {duplicates}")

    # Check weights sum approximately to 1
    total_weight = sum(c.weight for c in constituents)
    if abs(total_weight - Decimal("1")) > Decimal("0.01"):
        errors.append(
            f"Constituent weights sum to {total_weight}, expected ~1.0"
        )

    # Check for negative prices
    negative_prices = [s for s, p in prices.items() if p.close <= Decimal("0")]
    if negative_prices:
        errors.append(f"Symbols with non-positive prices: {negative_prices}")

    return errors
