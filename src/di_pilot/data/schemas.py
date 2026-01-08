"""
Data schemas for CSV/Parquet file validation.

Defines expected columns and data types for all input and output files.
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class ColumnSchema:
    """Schema definition for a single column."""
    name: str
    dtype: str  # pandas dtype string
    required: bool = True
    nullable: bool = False


@dataclass
class FileSchema:
    """Schema definition for a file."""
    name: str
    columns: list[ColumnSchema]
    description: str

    @property
    def required_columns(self) -> list[str]:
        """Get list of required column names."""
        return [c.name for c in self.columns if c.required]

    @property
    def all_columns(self) -> list[str]:
        """Get list of all column names."""
        return [c.name for c in self.columns]

    def validate_columns(self, df_columns: list[str]) -> tuple[bool, list[str]]:
        """
        Validate that a dataframe has the required columns.

        Args:
            df_columns: List of column names from the dataframe

        Returns:
            Tuple of (is_valid, list of missing columns)
        """
        missing = [col for col in self.required_columns if col not in df_columns]
        return len(missing) == 0, missing


# S&P 500 Constituents Schema
CONSTITUENTS_SCHEMA = FileSchema(
    name="sp500_constituents",
    description="S&P 500 constituent weights by date",
    columns=[
        ColumnSchema(name="symbol", dtype="str", required=True),
        ColumnSchema(name="weight", dtype="float64", required=True),
        ColumnSchema(name="date", dtype="datetime64[ns]", required=True),
    ],
)

# Price Data Schema
PRICES_SCHEMA = FileSchema(
    name="prices",
    description="Daily closing prices by symbol",
    columns=[
        ColumnSchema(name="symbol", dtype="str", required=True),
        ColumnSchema(name="date", dtype="datetime64[ns]", required=True),
        ColumnSchema(name="close", dtype="float64", required=True),
    ],
)

# Holdings Schema (input/output)
HOLDINGS_SCHEMA = FileSchema(
    name="holdings",
    description="Lot-level holdings with cost basis",
    columns=[
        ColumnSchema(name="lot_id", dtype="str", required=True),
        ColumnSchema(name="symbol", dtype="str", required=True),
        ColumnSchema(name="shares", dtype="float64", required=True),
        ColumnSchema(name="cost_basis", dtype="float64", required=True),
        ColumnSchema(name="acquisition_date", dtype="datetime64[ns]", required=True),
        ColumnSchema(name="portfolio_id", dtype="str", required=True),
    ],
)

# Valuation Output Schema
VALUATION_SCHEMA = FileSchema(
    name="valuation",
    description="Mark-to-market valuation with P&L",
    columns=[
        ColumnSchema(name="lot_id", dtype="str", required=True),
        ColumnSchema(name="symbol", dtype="str", required=True),
        ColumnSchema(name="shares", dtype="float64", required=True),
        ColumnSchema(name="cost_basis", dtype="float64", required=True),
        ColumnSchema(name="acquisition_date", dtype="datetime64[ns]", required=True),
        ColumnSchema(name="current_price", dtype="float64", required=True),
        ColumnSchema(name="market_value", dtype="float64", required=True),
        ColumnSchema(name="unrealized_pnl", dtype="float64", required=True),
        ColumnSchema(name="unrealized_pnl_pct", dtype="float64", required=True),
        ColumnSchema(name="gain_type", dtype="str", required=True),
        ColumnSchema(name="days_held", dtype="int64", required=True),
    ],
)

# Drift Report Schema
DRIFT_SCHEMA = FileSchema(
    name="drift_report",
    description="Per-symbol drift analysis vs benchmark",
    columns=[
        ColumnSchema(name="symbol", dtype="str", required=True),
        ColumnSchema(name="current_weight", dtype="float64", required=True),
        ColumnSchema(name="target_weight", dtype="float64", required=True),
        ColumnSchema(name="absolute_drift", dtype="float64", required=True),
        ColumnSchema(name="relative_drift", dtype="float64", required=True),
        ColumnSchema(name="exceeds_threshold", dtype="bool", required=True),
    ],
)

# TLH Candidates Schema
TLH_SCHEMA = FileSchema(
    name="tlh_candidates",
    description="Tax-loss harvesting candidates",
    columns=[
        ColumnSchema(name="lot_id", dtype="str", required=True),
        ColumnSchema(name="symbol", dtype="str", required=True),
        ColumnSchema(name="shares", dtype="float64", required=True),
        ColumnSchema(name="cost_basis", dtype="float64", required=True),
        ColumnSchema(name="current_price", dtype="float64", required=True),
        ColumnSchema(name="market_value", dtype="float64", required=True),
        ColumnSchema(name="loss_amount", dtype="float64", required=True),
        ColumnSchema(name="loss_pct", dtype="float64", required=True),
        ColumnSchema(name="days_held", dtype="int64", required=True),
        ColumnSchema(name="gain_type", dtype="str", required=True),
    ],
)

# Trade Proposals Schema
TRADE_PROPOSALS_SCHEMA = FileSchema(
    name="trade_proposals",
    description="Proposed trades with rationale",
    columns=[
        ColumnSchema(name="proposal_id", dtype="str", required=True),
        ColumnSchema(name="portfolio_id", dtype="str", required=True),
        ColumnSchema(name="symbol", dtype="str", required=True),
        ColumnSchema(name="side", dtype="str", required=True),
        ColumnSchema(name="shares", dtype="float64", required=True),
        ColumnSchema(name="rationale", dtype="str", required=True),
        ColumnSchema(name="rationale_detail", dtype="str", required=True),
        ColumnSchema(name="estimated_value", dtype="float64", required=True),
        ColumnSchema(name="current_price", dtype="float64", required=True),
        ColumnSchema(name="lot_id", dtype="str", required=False, nullable=True),
        ColumnSchema(name="generated_at", dtype="datetime64[ns]", required=True),
    ],
)
