"""
Analytics module for the Direct Indexing Shadow System.

Provides drift analysis, tax-loss harvesting detection, and P&L calculations.
"""

from di_pilot.analytics.drift import (
    calculate_drift,
    calculate_tracking_error,
    get_positions_exceeding_threshold,
)
from di_pilot.analytics.tlh import (
    identify_tlh_candidates,
    calculate_potential_harvest,
)
from di_pilot.analytics.pnl import (
    calculate_realized_pnl,
    calculate_unrealized_pnl_summary,
    calculate_pnl_by_gain_type,
)

__all__ = [
    "calculate_drift",
    "calculate_tracking_error",
    "get_positions_exceeding_threshold",
    "identify_tlh_candidates",
    "calculate_potential_harvest",
    "calculate_realized_pnl",
    "calculate_unrealized_pnl_summary",
    "calculate_pnl_by_gain_type",
]
