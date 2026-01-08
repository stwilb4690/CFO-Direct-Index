"""
Simulation module for the Direct Indexing Shadow System.

Provides backtest and forward test capabilities for evaluating
direct indexing portfolio strategies.
"""

from di_pilot.simulation.engine import SimulationEngine, SimulationState
from di_pilot.simulation.backtest import run_backtest
from di_pilot.simulation.forward import run_forward_test
from di_pilot.simulation.metrics import calculate_metrics, SimulationMetrics
from di_pilot.simulation.report import generate_report

__all__ = [
    "SimulationEngine",
    "SimulationState",
    "run_backtest",
    "run_forward_test",
    "calculate_metrics",
    "SimulationMetrics",
    "generate_report",
]
