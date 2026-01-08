"""
Performance metrics calculation for simulation results.

Calculates standard portfolio performance metrics including:
- CAGR (Compound Annual Growth Rate)
- Volatility (annualized)
- Maximum Drawdown
- Sharpe Ratio
- Tracking Error
- Turnover
- Tax-loss harvesting statistics
"""

import json
from dataclasses import dataclass, asdict
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Optional
import math

import numpy as np
import pandas as pd

from di_pilot.simulation.engine import DailySnapshot, SimulationTrade, TradeReason


@dataclass
class SimulationMetrics:
    """Container for simulation performance metrics."""

    # Time period
    start_date: str
    end_date: str
    trading_days: int

    # Returns
    total_return: float
    cagr: float
    annualized_volatility: float
    sharpe_ratio: float

    # Risk
    max_drawdown: float
    max_drawdown_date: str
    avg_drawdown: float

    # Trading
    total_trades: int
    buy_trades: int
    sell_trades: int
    total_turnover: float
    avg_daily_turnover: float

    # Tax-Loss Harvesting
    tlh_trades: int
    harvested_losses: float
    rebalance_trades: int

    # Portfolio
    final_value: float
    final_cash: float
    final_positions: int
    final_lots: int

    # Tracking (if benchmark data available)
    tracking_error: Optional[float] = None
    information_ratio: Optional[float] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)

    def to_json(self, path: Path) -> None:
        """Save metrics to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: Path) -> "SimulationMetrics":
        """Load metrics from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**data)


def calculate_metrics(
    snapshots: list[DailySnapshot],
    trades: list[SimulationTrade],
    initial_cash: Decimal,
    benchmark_returns: Optional[pd.Series] = None,
    risk_free_rate: float = 0.05,
) -> SimulationMetrics:
    """
    Calculate comprehensive performance metrics from simulation results.

    Args:
        snapshots: List of daily portfolio snapshots
        trades: List of executed trades
        initial_cash: Initial investment amount
        benchmark_returns: Optional benchmark daily returns for tracking error
        risk_free_rate: Annual risk-free rate for Sharpe ratio (default 5%)

    Returns:
        SimulationMetrics with all calculated values
    """
    if not snapshots:
        raise ValueError("No snapshots provided for metrics calculation")

    # Convert to DataFrame for easier calculation
    df = pd.DataFrame([
        {
            "date": s.date,
            "total_value": float(s.total_value),
            "daily_return": float(s.daily_return),
            "cash": float(s.cash),
            "num_positions": s.num_positions,
            "num_lots": s.num_lots,
        }
        for s in snapshots
    ])
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")

    # Time period
    start_date = snapshots[0].date
    end_date = snapshots[-1].date
    trading_days = len(snapshots)
    years = trading_days / 252  # Approximate trading days per year

    # Returns
    initial_value = float(initial_cash)
    final_value = df["total_value"].iloc[-1]
    total_return = (final_value - initial_value) / initial_value

    # CAGR
    if years > 0:
        cagr = (final_value / initial_value) ** (1 / years) - 1
    else:
        cagr = 0.0

    # Volatility (annualized)
    daily_returns = df["daily_return"].dropna()
    if len(daily_returns) > 1:
        daily_vol = daily_returns.std()
        annualized_volatility = daily_vol * math.sqrt(252)
    else:
        annualized_volatility = 0.0

    # Sharpe Ratio
    daily_rf = risk_free_rate / 252
    if annualized_volatility > 0 and len(daily_returns) > 0:
        excess_returns = daily_returns - daily_rf
        sharpe_ratio = (excess_returns.mean() * 252) / annualized_volatility
    else:
        sharpe_ratio = 0.0

    # Maximum Drawdown
    cumulative_max = df["total_value"].cummax()
    drawdown = (df["total_value"] - cumulative_max) / cumulative_max
    max_drawdown = drawdown.min()
    max_drawdown_idx = drawdown.idxmin()
    max_drawdown_date = max_drawdown_idx.strftime("%Y-%m-%d") if pd.notna(max_drawdown_idx) else ""
    avg_drawdown = drawdown.mean()

    # Trading statistics
    total_trades = len(trades)
    buy_trades = sum(1 for t in trades if t.side.value == "BUY")
    sell_trades = sum(1 for t in trades if t.side.value == "SELL")

    # TLH and Rebalance trades
    tlh_trades = sum(1 for t in trades if t.reason == TradeReason.TLH_SELL)
    rebalance_trades = sum(
        1 for t in trades
        if t.reason in (TradeReason.REBALANCE_BUY, TradeReason.REBALANCE_SELL)
    )

    # Harvested losses
    harvested_losses = sum(
        float(t.value) for t in trades if t.reason == TradeReason.TLH_SELL
    )

    # Turnover calculation
    total_trade_value = sum(float(t.value) for t in trades)
    avg_portfolio_value = df["total_value"].mean()
    total_turnover = total_trade_value / avg_portfolio_value if avg_portfolio_value > 0 else 0
    avg_daily_turnover = total_turnover / trading_days if trading_days > 0 else 0

    # Final portfolio state
    final_snapshot = snapshots[-1]
    final_cash = float(final_snapshot.cash)
    final_positions = final_snapshot.num_positions
    final_lots = final_snapshot.num_lots

    # Tracking error (if benchmark provided)
    tracking_error = None
    information_ratio = None
    if benchmark_returns is not None and len(benchmark_returns) > 0:
        # Align dates
        aligned = pd.DataFrame({
            "portfolio": daily_returns,
            "benchmark": benchmark_returns,
        }).dropna()

        if len(aligned) > 1:
            active_returns = aligned["portfolio"] - aligned["benchmark"]
            tracking_error = active_returns.std() * math.sqrt(252)

            if tracking_error > 0:
                information_ratio = (active_returns.mean() * 252) / tracking_error

    return SimulationMetrics(
        start_date=start_date.isoformat(),
        end_date=end_date.isoformat(),
        trading_days=trading_days,
        total_return=round(total_return, 6),
        cagr=round(cagr, 6),
        annualized_volatility=round(annualized_volatility, 6),
        sharpe_ratio=round(sharpe_ratio, 4),
        max_drawdown=round(max_drawdown, 6),
        max_drawdown_date=max_drawdown_date,
        avg_drawdown=round(avg_drawdown, 6),
        total_trades=total_trades,
        buy_trades=buy_trades,
        sell_trades=sell_trades,
        total_turnover=round(total_turnover, 4),
        avg_daily_turnover=round(avg_daily_turnover, 6),
        tlh_trades=tlh_trades,
        harvested_losses=round(harvested_losses, 2),
        rebalance_trades=rebalance_trades,
        final_value=round(final_value, 2),
        final_cash=round(final_cash, 2),
        final_positions=final_positions,
        final_lots=final_lots,
        tracking_error=round(tracking_error, 6) if tracking_error else None,
        information_ratio=round(information_ratio, 4) if information_ratio else None,
    )


def calculate_rolling_metrics(
    snapshots: list[DailySnapshot],
    window: int = 20,
) -> pd.DataFrame:
    """
    Calculate rolling performance metrics.

    Args:
        snapshots: List of daily snapshots
        window: Rolling window size in days

    Returns:
        DataFrame with rolling metrics
    """
    df = pd.DataFrame([
        {
            "date": s.date,
            "total_value": float(s.total_value),
            "daily_return": float(s.daily_return),
        }
        for s in snapshots
    ])
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")

    # Rolling calculations
    df["rolling_return"] = df["daily_return"].rolling(window).mean() * 252
    df["rolling_volatility"] = df["daily_return"].rolling(window).std() * math.sqrt(252)
    df["rolling_sharpe"] = df["rolling_return"] / df["rolling_volatility"]

    # Rolling max drawdown
    rolling_max = df["total_value"].rolling(window).max()
    df["rolling_drawdown"] = (df["total_value"] - rolling_max) / rolling_max

    return df.reset_index()


def compare_to_benchmark(
    snapshots: list[DailySnapshot],
    benchmark_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compare portfolio performance to benchmark.

    Args:
        snapshots: Portfolio snapshots
        benchmark_df: DataFrame with 'date' and 'close' columns for benchmark

    Returns:
        DataFrame comparing portfolio and benchmark
    """
    portfolio_df = pd.DataFrame([
        {
            "date": s.date,
            "portfolio_value": float(s.total_value),
            "portfolio_return": float(s.daily_return),
        }
        for s in snapshots
    ])
    portfolio_df["date"] = pd.to_datetime(portfolio_df["date"])

    benchmark_df = benchmark_df.copy()
    benchmark_df["date"] = pd.to_datetime(benchmark_df["date"])
    benchmark_df["benchmark_return"] = benchmark_df["close"].pct_change()

    # Normalize benchmark to start at same value as portfolio
    initial_value = portfolio_df["portfolio_value"].iloc[0]
    benchmark_df["benchmark_value"] = (
        (1 + benchmark_df["benchmark_return"]).cumprod() * initial_value
    )

    # Merge
    merged = pd.merge(
        portfolio_df,
        benchmark_df[["date", "benchmark_value", "benchmark_return"]],
        on="date",
        how="inner",
    )

    # Calculate active return
    merged["active_return"] = merged["portfolio_return"] - merged["benchmark_return"]

    # Calculate cumulative returns
    merged["portfolio_cumulative"] = (
        (1 + merged["portfolio_return"]).cumprod() - 1
    )
    merged["benchmark_cumulative"] = (
        (1 + merged["benchmark_return"]).cumprod() - 1
    )
    merged["active_cumulative"] = merged["portfolio_cumulative"] - merged["benchmark_cumulative"]

    return merged
