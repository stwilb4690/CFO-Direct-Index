"""
Core data models for the Direct Indexing Shadow System.

This module defines the fundamental data structures used throughout the system,
including tax lots, holdings, trade proposals, and portfolio configurations.
All monetary and share quantities use Decimal for precision.
"""

from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Optional
import uuid


class TradeSide(Enum):
    """Trade direction indicator."""
    BUY = "BUY"
    SELL = "SELL"


class TradeRationale(Enum):
    """Reason for trade proposal."""
    INITIAL_PURCHASE = "INITIAL_PURCHASE"
    REBALANCE = "REBALANCE"
    TAX_LOSS_HARVEST = "TAX_LOSS_HARVEST"
    DRIFT_CORRECTION = "DRIFT_CORRECTION"


class GainType(Enum):
    """Classification of capital gain/loss for tax purposes."""
    SHORT_TERM = "SHORT_TERM"  # Held <= 1 year
    LONG_TERM = "LONG_TERM"    # Held > 1 year


class ActionType(Enum):
    """Types of logged actions for the decision log."""
    PORTFOLIO_INITIALIZED = "PORTFOLIO_INITIALIZED"
    VALUATION_CALCULATED = "VALUATION_CALCULATED"
    DRIFT_ANALYZED = "DRIFT_ANALYZED"
    TLH_CANDIDATES_IDENTIFIED = "TLH_CANDIDATES_IDENTIFIED"
    TRADE_PROPOSALS_GENERATED = "TRADE_PROPOSALS_GENERATED"
    CONFIG_LOADED = "CONFIG_LOADED"


@dataclass
class TaxLot:
    """
    Represents a single tax lot within a position.

    A tax lot tracks the acquisition details of shares purchased at a specific
    time and cost, enabling accurate cost basis and holding period calculations.

    Attributes:
        lot_id: Unique identifier for this lot
        symbol: Ticker symbol of the security
        shares: Number of shares (supports fractional, 6 decimal places)
        cost_basis: Per-share cost basis at acquisition
        acquisition_date: Date the shares were acquired
        portfolio_id: ID of the portfolio containing this lot
    """
    lot_id: str
    symbol: str
    shares: Decimal
    cost_basis: Decimal  # per share
    acquisition_date: date
    portfolio_id: str

    @classmethod
    def create(
        cls,
        symbol: str,
        shares: Decimal,
        cost_basis: Decimal,
        acquisition_date: date,
        portfolio_id: str,
    ) -> "TaxLot":
        """Factory method to create a new TaxLot with auto-generated ID."""
        return cls(
            lot_id=str(uuid.uuid4()),
            symbol=symbol,
            shares=shares,
            cost_basis=cost_basis,
            acquisition_date=acquisition_date,
            portfolio_id=portfolio_id,
        )

    @property
    def total_cost(self) -> Decimal:
        """Total cost basis for this lot (shares * cost_basis)."""
        return self.shares * self.cost_basis


@dataclass
class LotValuation:
    """
    Mark-to-market valuation of a single tax lot.

    Extends lot information with current market value and unrealized P&L.

    Attributes:
        lot: The underlying tax lot
        current_price: Current market price per share
        valuation_date: Date of the valuation
        market_value: Current market value (shares * current_price)
        unrealized_pnl: Unrealized profit/loss
        unrealized_pnl_pct: Unrealized P&L as percentage of cost basis
        gain_type: Short-term or long-term classification
    """
    lot: TaxLot
    current_price: Decimal
    valuation_date: date
    market_value: Decimal
    unrealized_pnl: Decimal
    unrealized_pnl_pct: Decimal
    gain_type: GainType

    @classmethod
    def from_lot(
        cls,
        lot: TaxLot,
        current_price: Decimal,
        valuation_date: date,
    ) -> "LotValuation":
        """Create a LotValuation from a TaxLot and current price."""
        market_value = lot.shares * current_price
        total_cost = lot.total_cost
        unrealized_pnl = market_value - total_cost

        # Avoid division by zero
        if total_cost != Decimal("0"):
            unrealized_pnl_pct = unrealized_pnl / total_cost
        else:
            unrealized_pnl_pct = Decimal("0")

        # Determine gain type based on holding period
        days_held = (valuation_date - lot.acquisition_date).days
        gain_type = GainType.LONG_TERM if days_held > 365 else GainType.SHORT_TERM

        return cls(
            lot=lot,
            current_price=current_price,
            valuation_date=valuation_date,
            market_value=market_value,
            unrealized_pnl=unrealized_pnl,
            unrealized_pnl_pct=unrealized_pnl_pct,
            gain_type=gain_type,
        )


@dataclass
class PositionSummary:
    """
    Aggregated position summary for a single symbol.

    Combines all lots for a symbol into a summary view.

    Attributes:
        symbol: Ticker symbol
        total_shares: Total shares across all lots
        total_cost: Total cost basis across all lots
        market_value: Current market value
        unrealized_pnl: Total unrealized P&L
        num_lots: Number of tax lots
        current_weight: Current portfolio weight (0-1)
    """
    symbol: str
    total_shares: Decimal
    total_cost: Decimal
    market_value: Decimal
    unrealized_pnl: Decimal
    num_lots: int
    current_weight: Decimal = Decimal("0")


@dataclass
class DriftAnalysis:
    """
    Drift analysis for a single position vs benchmark.

    Attributes:
        symbol: Ticker symbol
        current_weight: Current portfolio weight (0-1)
        target_weight: Target/benchmark weight (0-1)
        absolute_drift: current_weight - target_weight
        relative_drift: absolute_drift / target_weight (if target > 0)
        exceeds_threshold: Whether drift exceeds configured threshold
    """
    symbol: str
    current_weight: Decimal
    target_weight: Decimal
    absolute_drift: Decimal
    relative_drift: Decimal
    exceeds_threshold: bool


@dataclass
class TLHCandidate:
    """
    Tax-loss harvesting candidate.

    Identifies a lot that has an unrealized loss exceeding the threshold.

    Attributes:
        lot_valuation: The valued lot with loss
        loss_amount: Absolute loss amount (negative)
        loss_pct: Loss as percentage of cost basis (negative)
        days_held: Number of days held
        gain_type: Short-term or long-term
    """
    lot_valuation: LotValuation
    loss_amount: Decimal
    loss_pct: Decimal
    days_held: int
    gain_type: GainType


@dataclass
class TradeProposal:
    """
    Proposed trade (not executed).

    Represents a trade recommendation for review. The system does not
    execute trades; all proposals require human review.

    Attributes:
        proposal_id: Unique identifier for this proposal
        portfolio_id: Portfolio this trade is for
        symbol: Ticker symbol
        side: BUY or SELL
        shares: Number of shares (can be fractional)
        rationale: Reason for the trade
        rationale_detail: Additional explanation
        estimated_value: Estimated trade value at current price
        current_price: Price used for estimation
        lot_id: For sells, the specific lot to sell from (optional)
        generated_at: When the proposal was generated
    """
    proposal_id: str
    portfolio_id: str
    symbol: str
    side: TradeSide
    shares: Decimal
    rationale: TradeRationale
    rationale_detail: str
    estimated_value: Decimal
    current_price: Decimal
    lot_id: Optional[str] = None
    generated_at: datetime = field(default_factory=datetime.now)

    @classmethod
    def create(
        cls,
        portfolio_id: str,
        symbol: str,
        side: TradeSide,
        shares: Decimal,
        rationale: TradeRationale,
        rationale_detail: str,
        current_price: Decimal,
        lot_id: Optional[str] = None,
    ) -> "TradeProposal":
        """Factory method to create a trade proposal with auto-generated ID."""
        return cls(
            proposal_id=str(uuid.uuid4()),
            portfolio_id=portfolio_id,
            symbol=symbol,
            side=side,
            shares=shares,
            rationale=rationale,
            rationale_detail=rationale_detail,
            estimated_value=shares * current_price,
            current_price=current_price,
            lot_id=lot_id,
        )


@dataclass
class PortfolioConfig:
    """
    Portfolio configuration loaded from YAML.

    Attributes:
        portfolio_id: Unique portfolio identifier
        cash: Initial cash amount
        start_date: Portfolio start date
        tlh_threshold: Tax-loss harvesting threshold (default 0.03 = 3%)
        drift_threshold: Drift threshold for rebalancing (default 0.005 = 0.5%)
        min_trade_value: Minimum trade value to propose
        output_dir: Directory for output files
    """
    portfolio_id: str
    cash: Decimal
    start_date: date
    tlh_threshold: Decimal = Decimal("0.03")
    drift_threshold: Decimal = Decimal("0.005")
    min_trade_value: Decimal = Decimal("100")
    output_dir: str = "output"


@dataclass
class BenchmarkConstituent:
    """
    S&P 500 benchmark constituent with weight.

    Attributes:
        symbol: Ticker symbol
        weight: Benchmark weight (0-1)
        as_of_date: Date the weight is effective
    """
    symbol: str
    weight: Decimal
    as_of_date: date


@dataclass
class PriceData:
    """
    Price data for a symbol on a specific date.

    Attributes:
        symbol: Ticker symbol
        date: Price date
        close: Closing price
    """
    symbol: str
    date: date
    close: Decimal


@dataclass
class DecisionLogEntry:
    """
    Entry for the append-only decision log.

    Attributes:
        timestamp: When the action occurred
        action_type: Type of action
        portfolio_id: Portfolio involved (if applicable)
        details: JSON-serializable details dictionary
    """
    timestamp: datetime
    action_type: ActionType
    portfolio_id: Optional[str]
    details: dict

    @classmethod
    def create(
        cls,
        action_type: ActionType,
        portfolio_id: Optional[str],
        details: dict,
    ) -> "DecisionLogEntry":
        """Factory method with auto-generated timestamp."""
        return cls(
            timestamp=datetime.now(),
            action_type=action_type,
            portfolio_id=portfolio_id,
            details=details,
        )


@dataclass
class PortfolioValuation:
    """
    Complete portfolio valuation summary.

    Attributes:
        portfolio_id: Portfolio identifier
        valuation_date: Date of valuation
        total_market_value: Sum of all position market values
        total_cost_basis: Sum of all position cost bases
        total_unrealized_pnl: Total unrealized P&L
        cash_balance: Any remaining cash (usually 0 after init)
        lot_valuations: List of individual lot valuations
        position_summaries: Aggregated by-symbol summaries
    """
    portfolio_id: str
    valuation_date: date
    total_market_value: Decimal
    total_cost_basis: Decimal
    total_unrealized_pnl: Decimal
    cash_balance: Decimal
    lot_valuations: list[LotValuation]
    position_summaries: list[PositionSummary]
