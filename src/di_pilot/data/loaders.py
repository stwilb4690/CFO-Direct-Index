"""
Data loading and saving functions for CSV/Parquet files.

Handles ingestion of benchmark constituents, prices, and holdings,
as well as output of valuation reports, drift analysis, and trade proposals.
"""

from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Optional

import pandas as pd

from di_pilot.models import (
    BenchmarkConstituent,
    DriftAnalysis,
    LotValuation,
    PriceData,
    TaxLot,
    TLHCandidate,
    TradeProposal,
)
from di_pilot.data.schemas import (
    CONSTITUENTS_SCHEMA,
    PRICES_SCHEMA,
    HOLDINGS_SCHEMA,
)


class DataLoadError(Exception):
    """Raised when data cannot be loaded or is invalid."""
    pass


def load_benchmark_constituents(
    file_path: str | Path,
    as_of_date: Optional[date] = None,
) -> list[BenchmarkConstituent]:
    """
    Load S&P 500 constituent weights from CSV file.

    Args:
        file_path: Path to CSV file with columns: symbol, weight, date
        as_of_date: If provided, filter to constituents on this date.
                   If None, uses the most recent date in the file.

    Returns:
        List of BenchmarkConstituent objects

    Raises:
        DataLoadError: If file cannot be loaded or is invalid
    """
    file_path = Path(file_path)
    df = _load_csv(file_path, CONSTITUENTS_SCHEMA)

    # Parse dates
    df["date"] = pd.to_datetime(df["date"]).dt.date

    # Filter to as_of_date
    if as_of_date is None:
        as_of_date = df["date"].max()
    else:
        if as_of_date not in df["date"].values:
            available_dates = sorted(df["date"].unique())
            # Find the closest date <= as_of_date
            prior_dates = [d for d in available_dates if d <= as_of_date]
            if not prior_dates:
                raise DataLoadError(
                    f"No constituent data available for date {as_of_date} or earlier"
                )
            as_of_date = max(prior_dates)

    df_filtered = df[df["date"] == as_of_date].copy()

    if df_filtered.empty:
        raise DataLoadError(f"No constituents found for date {as_of_date}")

    # Normalize weights to sum to 1
    total_weight = df_filtered["weight"].sum()
    if total_weight > 0:
        df_filtered["weight"] = df_filtered["weight"] / total_weight

    constituents = []
    for _, row in df_filtered.iterrows():
        constituents.append(
            BenchmarkConstituent(
                symbol=str(row["symbol"]).upper().strip(),
                weight=Decimal(str(row["weight"])),
                as_of_date=as_of_date,
            )
        )

    return constituents


def load_prices(
    file_path: str | Path,
    as_of_date: Optional[date] = None,
    symbols: Optional[list[str]] = None,
) -> dict[str, PriceData]:
    """
    Load price data from CSV file.

    Args:
        file_path: Path to CSV file with columns: symbol, date, close
        as_of_date: If provided, get prices for this date.
                   If None, uses the most recent date.
        symbols: Optional list of symbols to filter to

    Returns:
        Dictionary mapping symbol -> PriceData

    Raises:
        DataLoadError: If file cannot be loaded or is invalid
    """
    file_path = Path(file_path)
    df = _load_csv(file_path, PRICES_SCHEMA)

    # Parse dates
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["symbol"] = df["symbol"].str.upper().str.strip()

    # Filter to as_of_date
    if as_of_date is None:
        as_of_date = df["date"].max()
    else:
        if as_of_date not in df["date"].values:
            available_dates = sorted(df["date"].unique())
            # Find the closest date <= as_of_date
            prior_dates = [d for d in available_dates if d <= as_of_date]
            if not prior_dates:
                raise DataLoadError(
                    f"No price data available for date {as_of_date} or earlier"
                )
            as_of_date = max(prior_dates)

    df_filtered = df[df["date"] == as_of_date].copy()

    # Filter to symbols if provided
    if symbols:
        symbols_upper = [s.upper().strip() for s in symbols]
        df_filtered = df_filtered[df_filtered["symbol"].isin(symbols_upper)]

    prices = {}
    for _, row in df_filtered.iterrows():
        symbol = str(row["symbol"])
        prices[symbol] = PriceData(
            symbol=symbol,
            date=as_of_date,
            close=Decimal(str(row["close"])),
        )

    return prices


def load_holdings(
    file_path: str | Path,
    portfolio_id: Optional[str] = None,
) -> list[TaxLot]:
    """
    Load holdings (tax lots) from CSV file.

    Args:
        file_path: Path to CSV file with lot-level holdings
        portfolio_id: If provided, filter to this portfolio only

    Returns:
        List of TaxLot objects

    Raises:
        DataLoadError: If file cannot be loaded or is invalid
    """
    file_path = Path(file_path)
    df = _load_csv(file_path, HOLDINGS_SCHEMA)

    # Parse dates
    df["acquisition_date"] = pd.to_datetime(df["acquisition_date"]).dt.date

    # Filter to portfolio_id if provided
    if portfolio_id:
        df = df[df["portfolio_id"] == portfolio_id]

    lots = []
    for _, row in df.iterrows():
        lots.append(
            TaxLot(
                lot_id=str(row["lot_id"]),
                symbol=str(row["symbol"]).upper().strip(),
                shares=Decimal(str(row["shares"])),
                cost_basis=Decimal(str(row["cost_basis"])),
                acquisition_date=row["acquisition_date"],
                portfolio_id=str(row["portfolio_id"]),
            )
        )

    return lots


def save_holdings(
    lots: list[TaxLot],
    output_path: str | Path,
) -> Path:
    """
    Save holdings (tax lots) to CSV file.

    Args:
        lots: List of TaxLot objects to save
        output_path: Path for output CSV file

    Returns:
        Path to the saved file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = []
    for lot in lots:
        records.append({
            "lot_id": lot.lot_id,
            "symbol": lot.symbol,
            "shares": float(lot.shares),
            "cost_basis": float(lot.cost_basis),
            "acquisition_date": lot.acquisition_date.isoformat(),
            "portfolio_id": lot.portfolio_id,
        })

    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)

    return output_path


def save_valuations(
    valuations: list[LotValuation],
    output_path: str | Path,
) -> Path:
    """
    Save lot valuations to CSV file.

    Args:
        valuations: List of LotValuation objects
        output_path: Path for output CSV file

    Returns:
        Path to the saved file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = []
    for val in valuations:
        days_held = (val.valuation_date - val.lot.acquisition_date).days
        records.append({
            "lot_id": val.lot.lot_id,
            "symbol": val.lot.symbol,
            "shares": float(val.lot.shares),
            "cost_basis": float(val.lot.cost_basis),
            "acquisition_date": val.lot.acquisition_date.isoformat(),
            "current_price": float(val.current_price),
            "market_value": float(val.market_value),
            "unrealized_pnl": float(val.unrealized_pnl),
            "unrealized_pnl_pct": float(val.unrealized_pnl_pct),
            "gain_type": val.gain_type.value,
            "days_held": days_held,
        })

    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)

    return output_path


def save_drift_report(
    drift_analyses: list[DriftAnalysis],
    output_path: str | Path,
) -> Path:
    """
    Save drift analysis to CSV file.

    Args:
        drift_analyses: List of DriftAnalysis objects
        output_path: Path for output CSV file

    Returns:
        Path to the saved file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = []
    for da in drift_analyses:
        records.append({
            "symbol": da.symbol,
            "current_weight": float(da.current_weight),
            "target_weight": float(da.target_weight),
            "absolute_drift": float(da.absolute_drift),
            "relative_drift": float(da.relative_drift),
            "exceeds_threshold": da.exceeds_threshold,
        })

    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)

    return output_path


def save_tlh_candidates(
    candidates: list[TLHCandidate],
    output_path: str | Path,
) -> Path:
    """
    Save TLH candidates to CSV file.

    Args:
        candidates: List of TLHCandidate objects
        output_path: Path for output CSV file

    Returns:
        Path to the saved file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = []
    for cand in candidates:
        records.append({
            "lot_id": cand.lot_valuation.lot.lot_id,
            "symbol": cand.lot_valuation.lot.symbol,
            "shares": float(cand.lot_valuation.lot.shares),
            "cost_basis": float(cand.lot_valuation.lot.cost_basis),
            "current_price": float(cand.lot_valuation.current_price),
            "market_value": float(cand.lot_valuation.market_value),
            "loss_amount": float(cand.loss_amount),
            "loss_pct": float(cand.loss_pct),
            "days_held": cand.days_held,
            "gain_type": cand.gain_type.value,
        })

    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)

    return output_path


def save_trade_proposals(
    proposals: list[TradeProposal],
    output_path: str | Path,
) -> Path:
    """
    Save trade proposals to CSV file.

    Args:
        proposals: List of TradeProposal objects
        output_path: Path for output CSV file

    Returns:
        Path to the saved file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = []
    for prop in proposals:
        records.append({
            "proposal_id": prop.proposal_id,
            "portfolio_id": prop.portfolio_id,
            "symbol": prop.symbol,
            "side": prop.side.value,
            "shares": float(prop.shares),
            "rationale": prop.rationale.value,
            "rationale_detail": prop.rationale_detail,
            "estimated_value": float(prop.estimated_value),
            "current_price": float(prop.current_price),
            "lot_id": prop.lot_id or "",
            "generated_at": prop.generated_at.isoformat(),
        })

    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)

    return output_path


def _load_csv(file_path: Path, schema: "FileSchema") -> pd.DataFrame:
    """
    Load a CSV file and validate against schema.

    Args:
        file_path: Path to CSV file
        schema: Expected file schema

    Returns:
        Loaded DataFrame

    Raises:
        DataLoadError: If file cannot be loaded or has missing columns
    """
    if not file_path.exists():
        raise DataLoadError(f"File not found: {file_path}")

    # Support both CSV and Parquet
    if file_path.suffix.lower() == ".parquet":
        try:
            df = pd.read_parquet(file_path)
        except Exception as e:
            raise DataLoadError(f"Failed to load parquet file {file_path}: {e}")
    else:
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            raise DataLoadError(f"Failed to load CSV file {file_path}: {e}")

    # Validate columns
    is_valid, missing = schema.validate_columns(df.columns.tolist())
    if not is_valid:
        raise DataLoadError(
            f"File {file_path} is missing required columns: {missing}"
        )

    return df


# Import FileSchema type for type hints
from di_pilot.data.schemas import FileSchema
