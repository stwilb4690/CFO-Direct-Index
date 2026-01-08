"""
Trade proposal generation module for the Direct Indexing Shadow System.

Generates proposed trades for rebalancing and tax-loss harvesting.
All trades are proposals only - no execution occurs.
"""

from di_pilot.trading.proposals import (
    generate_rebalance_proposals,
    generate_tlh_proposals,
    generate_all_proposals,
)

__all__ = [
    "generate_rebalance_proposals",
    "generate_tlh_proposals",
    "generate_all_proposals",
]
