"""
Data ingestion module for the Direct Indexing Shadow System.

Provides functionality for loading S&P 500 constituents, price data,
and holdings from CSV/Parquet files.
"""

from di_pilot.data.loaders import (
    load_benchmark_constituents,
    load_prices,
    load_holdings,
    save_holdings,
    save_valuations,
    save_drift_report,
    save_tlh_candidates,
    save_trade_proposals,
)
from di_pilot.data.schemas import (
    CONSTITUENTS_SCHEMA,
    PRICES_SCHEMA,
    HOLDINGS_SCHEMA,
)

__all__ = [
    "load_benchmark_constituents",
    "load_prices",
    "load_holdings",
    "save_holdings",
    "save_valuations",
    "save_drift_report",
    "save_tlh_candidates",
    "save_trade_proposals",
    "CONSTITUENTS_SCHEMA",
    "PRICES_SCHEMA",
    "HOLDINGS_SCHEMA",
]
