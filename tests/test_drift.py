"""
Tests for drift analysis functionality.
"""

from decimal import Decimal

import pytest

from di_pilot.models import BenchmarkConstituent, DriftAnalysis
from di_pilot.analytics.drift import (
    calculate_drift,
    calculate_tracking_error,
    calculate_active_share,
    get_positions_exceeding_threshold,
    get_underweight_positions,
    get_overweight_positions,
    get_missing_positions,
    summarize_drift,
)


class TestCalculateDrift:
    """Tests for the calculate_drift function."""

    def test_perfect_alignment(self):
        """Test drift calculation when weights are perfectly aligned."""
        current_weights = {
            "AAPL": Decimal("0.10"),
            "MSFT": Decimal("0.08"),
            "AMZN": Decimal("0.05"),
        }
        constituents = [
            BenchmarkConstituent(symbol="AAPL", weight=Decimal("0.10"), as_of_date=None),
            BenchmarkConstituent(symbol="MSFT", weight=Decimal("0.08"), as_of_date=None),
            BenchmarkConstituent(symbol="AMZN", weight=Decimal("0.05"), as_of_date=None),
        ]

        drift_analyses = calculate_drift(current_weights, constituents)

        # All drifts should be zero
        for da in drift_analyses:
            assert da.absolute_drift == Decimal("0")
            assert not da.exceeds_threshold

    def test_overweight_detection(self):
        """Test detection of overweight positions."""
        current_weights = {
            "AAPL": Decimal("0.15"),  # 5% overweight
            "MSFT": Decimal("0.08"),
        }
        constituents = [
            BenchmarkConstituent(symbol="AAPL", weight=Decimal("0.10"), as_of_date=None),
            BenchmarkConstituent(symbol="MSFT", weight=Decimal("0.08"), as_of_date=None),
        ]

        drift_analyses = calculate_drift(
            current_weights, constituents, drift_threshold=Decimal("0.01")
        )

        aapl_drift = next(da for da in drift_analyses if da.symbol == "AAPL")
        assert aapl_drift.absolute_drift == Decimal("0.05")
        assert aapl_drift.exceeds_threshold

    def test_underweight_detection(self):
        """Test detection of underweight positions."""
        current_weights = {
            "AAPL": Decimal("0.05"),  # 5% underweight
            "MSFT": Decimal("0.08"),
        }
        constituents = [
            BenchmarkConstituent(symbol="AAPL", weight=Decimal("0.10"), as_of_date=None),
            BenchmarkConstituent(symbol="MSFT", weight=Decimal("0.08"), as_of_date=None),
        ]

        drift_analyses = calculate_drift(
            current_weights, constituents, drift_threshold=Decimal("0.01")
        )

        aapl_drift = next(da for da in drift_analyses if da.symbol == "AAPL")
        assert aapl_drift.absolute_drift == Decimal("-0.05")
        assert aapl_drift.exceeds_threshold

    def test_missing_positions_detected(self):
        """Test detection of benchmark positions not held."""
        current_weights = {
            "AAPL": Decimal("0.10"),
            # MSFT is missing
        }
        constituents = [
            BenchmarkConstituent(symbol="AAPL", weight=Decimal("0.10"), as_of_date=None),
            BenchmarkConstituent(symbol="MSFT", weight=Decimal("0.08"), as_of_date=None),
        ]

        drift_analyses = calculate_drift(current_weights, constituents)

        msft_drift = next(da for da in drift_analyses if da.symbol == "MSFT")
        assert msft_drift.current_weight == Decimal("0")
        assert msft_drift.absolute_drift == Decimal("-0.08")

    def test_extra_positions_detected(self):
        """Test detection of portfolio positions not in benchmark."""
        current_weights = {
            "AAPL": Decimal("0.10"),
            "EXTRA": Decimal("0.05"),  # Not in benchmark
        }
        constituents = [
            BenchmarkConstituent(symbol="AAPL", weight=Decimal("0.10"), as_of_date=None),
        ]

        drift_analyses = calculate_drift(current_weights, constituents)

        extra_drift = next(da for da in drift_analyses if da.symbol == "EXTRA")
        assert extra_drift.target_weight == Decimal("0")
        assert extra_drift.absolute_drift == Decimal("0.05")

    def test_relative_drift_calculation(self):
        """Test relative drift calculation."""
        current_weights = {
            "AAPL": Decimal("0.12"),  # 20% overweight relative to 10%
        }
        constituents = [
            BenchmarkConstituent(symbol="AAPL", weight=Decimal("0.10"), as_of_date=None),
        ]

        drift_analyses = calculate_drift(current_weights, constituents)

        aapl_drift = next(da for da in drift_analyses if da.symbol == "AAPL")
        assert aapl_drift.relative_drift == Decimal("0.2")  # 20% relative drift


class TestCalculateTrackingError:
    """Tests for the calculate_tracking_error function."""

    def test_zero_tracking_error(self):
        """Test tracking error is zero when perfectly aligned."""
        drift_analyses = [
            DriftAnalysis(
                symbol="AAPL",
                current_weight=Decimal("0.10"),
                target_weight=Decimal("0.10"),
                absolute_drift=Decimal("0"),
                relative_drift=Decimal("0"),
                exceeds_threshold=False,
            ),
        ]

        te = calculate_tracking_error(drift_analyses)
        assert te == Decimal("0")

    def test_tracking_error_calculation(self):
        """Test tracking error calculation with drift."""
        drift_analyses = [
            DriftAnalysis(
                symbol="AAPL",
                current_weight=Decimal("0.12"),
                target_weight=Decimal("0.10"),
                absolute_drift=Decimal("0.02"),
                relative_drift=Decimal("0.2"),
                exceeds_threshold=True,
            ),
            DriftAnalysis(
                symbol="MSFT",
                current_weight=Decimal("0.06"),
                target_weight=Decimal("0.08"),
                absolute_drift=Decimal("-0.02"),
                relative_drift=Decimal("-0.25"),
                exceeds_threshold=True,
            ),
        ]

        te = calculate_tracking_error(drift_analyses)
        # Sum of squared drifts: 0.02^2 + 0.02^2 = 0.0004 + 0.0004 = 0.0008
        assert te == Decimal("0.0008")


class TestCalculateActiveShare:
    """Tests for the calculate_active_share function."""

    def test_zero_active_share(self):
        """Test active share is zero when identical to benchmark."""
        drift_analyses = [
            DriftAnalysis(
                symbol="AAPL",
                current_weight=Decimal("0.10"),
                target_weight=Decimal("0.10"),
                absolute_drift=Decimal("0"),
                relative_drift=Decimal("0"),
                exceeds_threshold=False,
            ),
        ]

        active_share = calculate_active_share(drift_analyses)
        assert active_share == Decimal("0")

    def test_active_share_calculation(self):
        """Test active share calculation."""
        drift_analyses = [
            DriftAnalysis(
                symbol="AAPL",
                current_weight=Decimal("0.12"),
                target_weight=Decimal("0.10"),
                absolute_drift=Decimal("0.02"),
                relative_drift=Decimal("0.2"),
                exceeds_threshold=True,
            ),
            DriftAnalysis(
                symbol="MSFT",
                current_weight=Decimal("0.06"),
                target_weight=Decimal("0.08"),
                absolute_drift=Decimal("-0.02"),
                relative_drift=Decimal("-0.25"),
                exceeds_threshold=True,
            ),
        ]

        active_share = calculate_active_share(drift_analyses)
        # Sum of abs drifts / 2 = (0.02 + 0.02) / 2 = 0.02
        assert active_share == Decimal("0.02")


class TestGetPositionsExceedingThreshold:
    """Tests for the get_positions_exceeding_threshold function."""

    def test_filters_exceeding_only(self):
        """Test that only positions exceeding threshold are returned."""
        drift_analyses = [
            DriftAnalysis(
                symbol="AAPL",
                current_weight=Decimal("0.12"),
                target_weight=Decimal("0.10"),
                absolute_drift=Decimal("0.02"),
                relative_drift=Decimal("0.2"),
                exceeds_threshold=True,
            ),
            DriftAnalysis(
                symbol="MSFT",
                current_weight=Decimal("0.081"),
                target_weight=Decimal("0.08"),
                absolute_drift=Decimal("0.001"),
                relative_drift=Decimal("0.0125"),
                exceeds_threshold=False,
            ),
        ]

        exceeding = get_positions_exceeding_threshold(drift_analyses)
        assert len(exceeding) == 1
        assert exceeding[0].symbol == "AAPL"


class TestGetUnderweightPositions:
    """Tests for the get_underweight_positions function."""

    def test_gets_underweight_only(self):
        """Test that only underweight positions are returned."""
        drift_analyses = [
            DriftAnalysis(
                symbol="AAPL",
                current_weight=Decimal("0.12"),
                target_weight=Decimal("0.10"),
                absolute_drift=Decimal("0.02"),
                relative_drift=Decimal("0.2"),
                exceeds_threshold=True,
            ),
            DriftAnalysis(
                symbol="MSFT",
                current_weight=Decimal("0.06"),
                target_weight=Decimal("0.08"),
                absolute_drift=Decimal("-0.02"),
                relative_drift=Decimal("-0.25"),
                exceeds_threshold=True,
            ),
        ]

        underweight = get_underweight_positions(drift_analyses)
        assert len(underweight) == 1
        assert underweight[0].symbol == "MSFT"


class TestGetOverweightPositions:
    """Tests for the get_overweight_positions function."""

    def test_gets_overweight_only(self):
        """Test that only overweight positions are returned."""
        drift_analyses = [
            DriftAnalysis(
                symbol="AAPL",
                current_weight=Decimal("0.12"),
                target_weight=Decimal("0.10"),
                absolute_drift=Decimal("0.02"),
                relative_drift=Decimal("0.2"),
                exceeds_threshold=True,
            ),
            DriftAnalysis(
                symbol="MSFT",
                current_weight=Decimal("0.06"),
                target_weight=Decimal("0.08"),
                absolute_drift=Decimal("-0.02"),
                relative_drift=Decimal("-0.25"),
                exceeds_threshold=True,
            ),
        ]

        overweight = get_overweight_positions(drift_analyses)
        assert len(overweight) == 1
        assert overweight[0].symbol == "AAPL"


class TestSummarizeDrift:
    """Tests for the summarize_drift function."""

    def test_summary_statistics(self):
        """Test that summary contains expected statistics."""
        drift_analyses = [
            DriftAnalysis(
                symbol="AAPL",
                current_weight=Decimal("0.12"),
                target_weight=Decimal("0.10"),
                absolute_drift=Decimal("0.02"),
                relative_drift=Decimal("0.2"),
                exceeds_threshold=True,
            ),
            DriftAnalysis(
                symbol="MSFT",
                current_weight=Decimal("0.08"),
                target_weight=Decimal("0.08"),
                absolute_drift=Decimal("0"),
                relative_drift=Decimal("0"),
                exceeds_threshold=False,
            ),
        ]

        summary = summarize_drift(drift_analyses)

        assert summary["total_positions"] == 2
        assert summary["positions_exceeding_threshold"] == 1
        assert summary["overweight_positions"] == 1
        assert summary["underweight_positions"] == 0
        assert summary["max_absolute_drift"] == Decimal("0.02")
