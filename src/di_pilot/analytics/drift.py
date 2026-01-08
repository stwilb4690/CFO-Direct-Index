"""
Drift analysis for portfolio vs benchmark weights.

This module calculates position-level and portfolio-level drift from
S&P 500 target weights, identifying positions that need rebalancing.
"""

from decimal import Decimal
from typing import Optional

from di_pilot.models import (
    BenchmarkConstituent,
    DriftAnalysis,
    LotValuation,
    PositionSummary,
)
from di_pilot.portfolio.holdings import calculate_position_weights


def calculate_drift(
    current_weights: dict[str, Decimal],
    benchmark_constituents: list[BenchmarkConstituent],
    drift_threshold: Decimal = Decimal("0.005"),
) -> list[DriftAnalysis]:
    """
    Calculate drift for each position vs benchmark.

    Args:
        current_weights: Current portfolio weights by symbol
        benchmark_constituents: Benchmark constituents with target weights
        drift_threshold: Threshold for flagging drift (default 0.5%)

    Returns:
        List of DriftAnalysis objects for all positions
    """
    # Build benchmark weight lookup
    benchmark_weights = {c.symbol: c.weight for c in benchmark_constituents}

    # Get all symbols (union of current and benchmark)
    all_symbols = set(current_weights.keys()) | set(benchmark_weights.keys())

    drift_analyses = []

    for symbol in all_symbols:
        current_weight = current_weights.get(symbol, Decimal("0"))
        target_weight = benchmark_weights.get(symbol, Decimal("0"))

        absolute_drift = current_weight - target_weight

        # Calculate relative drift (vs target weight)
        if target_weight != Decimal("0"):
            relative_drift = absolute_drift / target_weight
        else:
            # Position exists but not in benchmark
            relative_drift = Decimal("1") if current_weight > Decimal("0") else Decimal("0")

        exceeds_threshold = abs(absolute_drift) > drift_threshold

        drift_analyses.append(
            DriftAnalysis(
                symbol=symbol,
                current_weight=current_weight,
                target_weight=target_weight,
                absolute_drift=absolute_drift,
                relative_drift=relative_drift,
                exceeds_threshold=exceeds_threshold,
            )
        )

    # Sort by absolute drift magnitude descending
    drift_analyses.sort(key=lambda x: abs(x.absolute_drift), reverse=True)

    return drift_analyses


def calculate_drift_from_valuations(
    valuations: list[LotValuation],
    benchmark_constituents: list[BenchmarkConstituent],
    drift_threshold: Decimal = Decimal("0.005"),
) -> list[DriftAnalysis]:
    """
    Calculate drift directly from lot valuations.

    Args:
        valuations: List of lot valuations
        benchmark_constituents: Benchmark constituents with target weights
        drift_threshold: Threshold for flagging drift

    Returns:
        List of DriftAnalysis objects
    """
    current_weights = calculate_position_weights(valuations)
    return calculate_drift(current_weights, benchmark_constituents, drift_threshold)


def calculate_tracking_error(
    drift_analyses: list[DriftAnalysis],
) -> Decimal:
    """
    Calculate portfolio-level tracking error.

    Uses simple sum of squared absolute drifts as a tracking error proxy.

    Args:
        drift_analyses: List of drift analyses

    Returns:
        Tracking error measure (sum of squared drifts)
    """
    return sum(da.absolute_drift ** 2 for da in drift_analyses)


def calculate_active_share(
    drift_analyses: list[DriftAnalysis],
) -> Decimal:
    """
    Calculate active share vs benchmark.

    Active share = 0.5 * sum(|current_weight - target_weight|)

    Args:
        drift_analyses: List of drift analyses

    Returns:
        Active share (0 = identical to benchmark, 1 = no overlap)
    """
    total_absolute_drift = sum(abs(da.absolute_drift) for da in drift_analyses)
    return total_absolute_drift / Decimal("2")


def get_positions_exceeding_threshold(
    drift_analyses: list[DriftAnalysis],
) -> list[DriftAnalysis]:
    """
    Get positions with drift exceeding the threshold.

    Args:
        drift_analyses: List of drift analyses

    Returns:
        Filtered list of positions exceeding threshold
    """
    return [da for da in drift_analyses if da.exceeds_threshold]


def get_underweight_positions(
    drift_analyses: list[DriftAnalysis],
    min_drift: Decimal = Decimal("0.001"),
) -> list[DriftAnalysis]:
    """
    Get positions that are underweight vs benchmark.

    Args:
        drift_analyses: List of drift analyses
        min_drift: Minimum drift to consider

    Returns:
        List of underweight positions (sorted by drift magnitude)
    """
    underweight = [
        da for da in drift_analyses
        if da.absolute_drift < -min_drift
    ]
    underweight.sort(key=lambda x: x.absolute_drift)  # Most underweight first
    return underweight


def get_overweight_positions(
    drift_analyses: list[DriftAnalysis],
    min_drift: Decimal = Decimal("0.001"),
) -> list[DriftAnalysis]:
    """
    Get positions that are overweight vs benchmark.

    Args:
        drift_analyses: List of drift analyses
        min_drift: Minimum drift to consider

    Returns:
        List of overweight positions (sorted by drift magnitude descending)
    """
    overweight = [
        da for da in drift_analyses
        if da.absolute_drift > min_drift
    ]
    overweight.sort(key=lambda x: x.absolute_drift, reverse=True)  # Most overweight first
    return overweight


def get_missing_positions(
    drift_analyses: list[DriftAnalysis],
) -> list[DriftAnalysis]:
    """
    Get benchmark positions not held in portfolio.

    Args:
        drift_analyses: List of drift analyses

    Returns:
        List of positions with target > 0 but current = 0
    """
    return [
        da for da in drift_analyses
        if da.current_weight == Decimal("0") and da.target_weight > Decimal("0")
    ]


def get_non_benchmark_positions(
    drift_analyses: list[DriftAnalysis],
) -> list[DriftAnalysis]:
    """
    Get portfolio positions not in benchmark.

    Args:
        drift_analyses: List of drift analyses

    Returns:
        List of positions with current > 0 but target = 0
    """
    return [
        da for da in drift_analyses
        if da.current_weight > Decimal("0") and da.target_weight == Decimal("0")
    ]


def summarize_drift(
    drift_analyses: list[DriftAnalysis],
) -> dict:
    """
    Generate summary statistics for drift analysis.

    Args:
        drift_analyses: List of drift analyses

    Returns:
        Dictionary with summary statistics
    """
    exceeding = get_positions_exceeding_threshold(drift_analyses)
    underweight = get_underweight_positions(drift_analyses)
    overweight = get_overweight_positions(drift_analyses)
    missing = get_missing_positions(drift_analyses)

    return {
        "total_positions": len(drift_analyses),
        "positions_exceeding_threshold": len(exceeding),
        "underweight_positions": len(underweight),
        "overweight_positions": len(overweight),
        "missing_positions": len(missing),
        "tracking_error": calculate_tracking_error(drift_analyses),
        "active_share": calculate_active_share(drift_analyses),
        "max_absolute_drift": max(
            (abs(da.absolute_drift) for da in drift_analyses),
            default=Decimal("0"),
        ),
    }
