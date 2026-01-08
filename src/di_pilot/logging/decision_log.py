"""
Append-only decision logging for the Direct Indexing Shadow System.

All significant actions are logged with timestamps and rationale
to support auditability and reproducibility.
"""

import json
from datetime import datetime, date
from decimal import Decimal
from pathlib import Path
from typing import Any, Optional

from di_pilot.models import (
    ActionType,
    DecisionLogEntry,
    PortfolioConfig,
    PortfolioValuation,
    TaxLot,
    TradeProposal,
    DriftAnalysis,
    TLHCandidate,
)


class DecisionLogger:
    """
    Append-only decision logger.

    Writes all decisions to a JSONL file for audit purposes.
    Each line is a complete JSON object representing one action.
    """

    def __init__(self, log_path: str | Path):
        """
        Initialize the decision logger.

        Args:
            log_path: Path to the log file (will be created if not exists)
        """
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, entry: DecisionLogEntry) -> None:
        """
        Write a decision log entry.

        Args:
            entry: DecisionLogEntry to write
        """
        record = {
            "timestamp": entry.timestamp.isoformat(),
            "action_type": entry.action_type.value,
            "portfolio_id": entry.portfolio_id,
            "details": entry.details,
        }

        with open(self.log_path, "a") as f:
            f.write(json.dumps(record, cls=DecimalEncoder) + "\n")

    def log_portfolio_initialized(
        self,
        config: PortfolioConfig,
        lots: list[TaxLot],
        proposals: list[TradeProposal],
    ) -> None:
        """
        Log portfolio initialization.

        Args:
            config: Portfolio configuration used
            lots: Tax lots created
            proposals: Initial purchase proposals
        """
        details = {
            "cash": str(config.cash),
            "start_date": config.start_date.isoformat(),
            "num_positions": len(set(lot.symbol for lot in lots)),
            "num_lots": len(lots),
            "total_allocated": str(sum(lot.total_cost for lot in lots)),
            "parameters": {
                "tlh_threshold": str(config.tlh_threshold),
                "drift_threshold": str(config.drift_threshold),
            },
        }

        entry = DecisionLogEntry.create(
            action_type=ActionType.PORTFOLIO_INITIALIZED,
            portfolio_id=config.portfolio_id,
            details=details,
        )
        self.log(entry)

    def log_valuation_calculated(
        self,
        valuation: PortfolioValuation,
    ) -> None:
        """
        Log portfolio valuation.

        Args:
            valuation: Portfolio valuation result
        """
        details = {
            "valuation_date": valuation.valuation_date.isoformat(),
            "total_market_value": str(valuation.total_market_value),
            "total_cost_basis": str(valuation.total_cost_basis),
            "total_unrealized_pnl": str(valuation.total_unrealized_pnl),
            "num_positions": len(valuation.position_summaries),
            "num_lots": len(valuation.lot_valuations),
        }

        entry = DecisionLogEntry.create(
            action_type=ActionType.VALUATION_CALCULATED,
            portfolio_id=valuation.portfolio_id,
            details=details,
        )
        self.log(entry)

    def log_drift_analyzed(
        self,
        portfolio_id: str,
        drift_analyses: list[DriftAnalysis],
        threshold: Decimal,
    ) -> None:
        """
        Log drift analysis.

        Args:
            portfolio_id: Portfolio identifier
            drift_analyses: List of drift analyses
            threshold: Drift threshold used
        """
        exceeding = [d for d in drift_analyses if d.exceeds_threshold]

        details = {
            "threshold": str(threshold),
            "total_positions": len(drift_analyses),
            "positions_exceeding_threshold": len(exceeding),
            "max_drift": str(max(
                (abs(d.absolute_drift) for d in drift_analyses),
                default=Decimal("0"),
            )),
            "exceeding_symbols": [d.symbol for d in exceeding[:10]],  # First 10
        }

        entry = DecisionLogEntry.create(
            action_type=ActionType.DRIFT_ANALYZED,
            portfolio_id=portfolio_id,
            details=details,
        )
        self.log(entry)

    def log_tlh_candidates_identified(
        self,
        portfolio_id: str,
        candidates: list[TLHCandidate],
        threshold: Decimal,
    ) -> None:
        """
        Log TLH candidate identification.

        Args:
            portfolio_id: Portfolio identifier
            candidates: List of TLH candidates
            threshold: Loss threshold used
        """
        total_harvest = sum(abs(c.loss_amount) for c in candidates)

        details = {
            "threshold": str(threshold),
            "candidate_count": len(candidates),
            "total_potential_harvest": str(total_harvest),
            "symbols": list(set(c.lot_valuation.lot.symbol for c in candidates))[:10],
        }

        entry = DecisionLogEntry.create(
            action_type=ActionType.TLH_CANDIDATES_IDENTIFIED,
            portfolio_id=portfolio_id,
            details=details,
        )
        self.log(entry)

    def log_trade_proposals_generated(
        self,
        portfolio_id: str,
        proposals: list[TradeProposal],
    ) -> None:
        """
        Log trade proposal generation.

        Args:
            portfolio_id: Portfolio identifier
            proposals: List of trade proposals
        """
        from di_pilot.trading.proposals import calculate_proposal_summary

        summary = calculate_proposal_summary(proposals)

        details = {
            "total_proposals": summary["total_proposals"],
            "buy_count": summary["buy_count"],
            "sell_count": summary["sell_count"],
            "total_buy_value": str(summary["total_buy_value"]),
            "total_sell_value": str(summary["total_sell_value"]),
            "tlh_proposals": summary["tlh_proposals"],
            "rebalance_proposals": summary["rebalance_proposals"],
            "proposal_ids": [p.proposal_id for p in proposals],
        }

        entry = DecisionLogEntry.create(
            action_type=ActionType.TRADE_PROPOSALS_GENERATED,
            portfolio_id=portfolio_id,
            details=details,
        )
        self.log(entry)

    def log_config_loaded(
        self,
        config: PortfolioConfig,
        config_path: str,
    ) -> None:
        """
        Log configuration loading.

        Args:
            config: Loaded configuration
            config_path: Path to configuration file
        """
        details = {
            "config_path": config_path,
            "portfolio_id": config.portfolio_id,
            "cash": str(config.cash),
            "start_date": config.start_date.isoformat(),
            "tlh_threshold": str(config.tlh_threshold),
            "drift_threshold": str(config.drift_threshold),
        }

        entry = DecisionLogEntry.create(
            action_type=ActionType.CONFIG_LOADED,
            portfolio_id=config.portfolio_id,
            details=details,
        )
        self.log(entry)

    def read_log(self) -> list[DecisionLogEntry]:
        """
        Read all entries from the log file.

        Returns:
            List of DecisionLogEntry objects
        """
        if not self.log_path.exists():
            return []

        entries = []
        with open(self.log_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                entries.append(
                    DecisionLogEntry(
                        timestamp=datetime.fromisoformat(record["timestamp"]),
                        action_type=ActionType(record["action_type"]),
                        portfolio_id=record.get("portfolio_id"),
                        details=record.get("details", {}),
                    )
                )

        return entries

    def filter_by_portfolio(
        self,
        portfolio_id: str,
    ) -> list[DecisionLogEntry]:
        """
        Get log entries for a specific portfolio.

        Args:
            portfolio_id: Portfolio to filter by

        Returns:
            Filtered list of entries
        """
        return [e for e in self.read_log() if e.portfolio_id == portfolio_id]

    def filter_by_action_type(
        self,
        action_type: ActionType,
    ) -> list[DecisionLogEntry]:
        """
        Get log entries of a specific action type.

        Args:
            action_type: Action type to filter by

        Returns:
            Filtered list of entries
        """
        return [e for e in self.read_log() if e.action_type == action_type]


class DecimalEncoder(json.JSONEncoder):
    """JSON encoder that handles Decimal types."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, Decimal):
            return str(obj)
        if isinstance(obj, date):
            return obj.isoformat()
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


# Global logger instance (initialized on first use)
_global_logger: Optional[DecisionLogger] = None


def get_logger(log_path: Optional[str | Path] = None) -> DecisionLogger:
    """
    Get or create the global decision logger.

    Args:
        log_path: Optional path to initialize logger (required on first call)

    Returns:
        DecisionLogger instance
    """
    global _global_logger

    if _global_logger is None:
        if log_path is None:
            log_path = "output/decision_log.jsonl"
        _global_logger = DecisionLogger(log_path)
    elif log_path is not None:
        # Allow reinitializing with new path
        _global_logger = DecisionLogger(log_path)

    return _global_logger


def log_action(
    action_type: ActionType,
    portfolio_id: Optional[str],
    details: dict,
    log_path: Optional[str | Path] = None,
) -> None:
    """
    Convenience function to log an action.

    Args:
        action_type: Type of action
        portfolio_id: Portfolio identifier (optional)
        details: Action details dictionary
        log_path: Optional path to log file
    """
    logger = get_logger(log_path)
    entry = DecisionLogEntry.create(
        action_type=action_type,
        portfolio_id=portfolio_id,
        details=details,
    )
    logger.log(entry)
