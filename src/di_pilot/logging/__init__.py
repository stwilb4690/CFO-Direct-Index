"""
Decision logging module for the Direct Indexing Shadow System.

Provides append-only decision logging for audit and reproducibility.
"""

from di_pilot.logging.decision_log import (
    DecisionLogger,
    log_action,
    get_logger,
)

__all__ = [
    "DecisionLogger",
    "log_action",
    "get_logger",
]
