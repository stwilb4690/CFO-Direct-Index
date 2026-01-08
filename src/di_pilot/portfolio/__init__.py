"""
Portfolio management module for the Direct Indexing Shadow System.

Provides functionality for initializing portfolios from cash,
tracking lot-level holdings, and calculating valuations.
"""

from di_pilot.portfolio.initialize import initialize_portfolio
from di_pilot.portfolio.holdings import (
    aggregate_holdings_by_symbol,
    get_portfolio_symbols,
    calculate_position_weights,
)
from di_pilot.portfolio.valuation import (
    value_portfolio,
    value_lots,
)

__all__ = [
    "initialize_portfolio",
    "aggregate_holdings_by_symbol",
    "get_portfolio_symbols",
    "calculate_position_weights",
    "value_portfolio",
    "value_lots",
]
