"""
Report generation for simulation results.

Generates human-readable markdown reports and JSON metric files
summarizing simulation performance and trading activity.
"""

from datetime import date, datetime
from pathlib import Path
from typing import Optional, Union

from di_pilot.simulation.backtest import BacktestResult
from di_pilot.simulation.forward import ForwardTestResult
from di_pilot.simulation.metrics import SimulationMetrics, calculate_metrics
from di_pilot.simulation.engine import SimulationConfig, TradeReason


def generate_report(
    result: Union[BacktestResult, ForwardTestResult],
    output_dir: str | Path,
    include_trade_details: bool = True,
) -> dict[str, Path]:
    """
    Generate complete simulation report.

    Creates:
    - run_report.md: Human-readable markdown summary
    - metrics.json: Machine-readable metrics
    - trades.csv: All executed trades
    - portfolio_daily.csv: Daily portfolio values
    - professional_report.html: Professional HTML report with charts

    Args:
        result: Backtest or Forward test result
        output_dir: Directory to save outputs
        include_trade_details: Include detailed trade list in markdown

    Returns:
        Dictionary mapping output type to file path
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {}

    # Calculate metrics
    metrics = calculate_metrics(
        snapshots=result.snapshots,
        trades=result.trades,
        initial_cash=result.config.initial_cash,
    )

    # Save metrics JSON
    metrics_path = output_dir / "metrics.json"
    metrics.to_json(metrics_path)
    paths["metrics"] = metrics_path

    # Save CSV files
    csv_paths = result.save_outputs(output_dir)
    paths.update(csv_paths)

    # Generate markdown report
    report_path = output_dir / "run_report.md"
    markdown = _generate_markdown_report(result, metrics, include_trade_details)
    with open(report_path, "w") as f:
        f.write(markdown)
    paths["report"] = report_path

    # Generate professional HTML report
    try:
        from di_pilot.simulation.professional_report import generate_professional_report
        html_path = generate_professional_report(result, output_dir)
        paths["professional_report"] = html_path
    except Exception as e:
        # Don't fail if professional report fails
        print(f"Warning: Could not generate professional report: {e}")

    return paths


def _generate_markdown_report(
    result: Union[BacktestResult, ForwardTestResult],
    metrics: SimulationMetrics,
    include_trade_details: bool,
) -> str:
    """Generate markdown report content."""
    is_backtest = isinstance(result, BacktestResult)
    sim_type = "Backtest" if is_backtest else "Forward Test"

    lines = [
        f"# Direct Indexing {sim_type} Report",
        "",
        f"**Run ID:** `{result.run_id}`",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "---",
        "",
        "## Summary",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Start Date | {metrics.start_date} |",
        f"| End Date | {metrics.end_date} |",
        f"| Trading Days | {metrics.trading_days} |",
        f"| Initial Investment | ${result.config.initial_cash:,.2f} |",
        f"| Final Value | ${metrics.final_value:,.2f} |",
        f"| Total Return | {metrics.total_return:.2%} |",
        f"| CAGR | {metrics.cagr:.2%} |",
        "",
        "---",
        "",
        "## Performance Metrics",
        "",
        "### Returns",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total Return | {metrics.total_return:.2%} |",
        f"| CAGR | {metrics.cagr:.2%} |",
        f"| Annualized Volatility | {metrics.annualized_volatility:.2%} |",
        f"| Sharpe Ratio | {metrics.sharpe_ratio:.2f} |",
        "",
        "### Risk",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Maximum Drawdown | {metrics.max_drawdown:.2%} |",
        f"| Max Drawdown Date | {metrics.max_drawdown_date} |",
        f"| Average Drawdown | {metrics.avg_drawdown:.2%} |",
        "",
    ]

    if metrics.tracking_error is not None:
        lines.extend([
            "### Tracking",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Tracking Error | {metrics.tracking_error:.2%} |",
        ])
        if metrics.information_ratio is not None:
            lines.append(f"| Information Ratio | {metrics.information_ratio:.2f} |")
        lines.append("")

    lines.extend([
        "---",
        "",
        "## Trading Activity",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total Trades | {metrics.total_trades} |",
        f"| Buy Trades | {metrics.buy_trades} |",
        f"| Sell Trades | {metrics.sell_trades} |",
        f"| Total Turnover | {metrics.total_turnover:.1%} |",
        f"| Avg Daily Turnover | {metrics.avg_daily_turnover:.4%} |",
        "",
        "### By Reason",
        "",
        f"| Reason | Count |",
        f"|--------|-------|",
        f"| Initial Purchase | {sum(1 for t in result.trades if t.reason == TradeReason.INITIAL_PURCHASE)} |",
        f"| Rebalance | {metrics.rebalance_trades} |",
        f"| Tax-Loss Harvest | {metrics.tlh_trades} |",
        "",
        "---",
        "",
        "## Tax-Loss Harvesting",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| TLH Trades | {metrics.tlh_trades} |",
        f"| Harvested Losses | ${metrics.harvested_losses:,.2f} |",
        "",
        "---",
        "",
        "## Portfolio State (Final)",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total Value | ${metrics.final_value:,.2f} |",
        f"| Cash | ${metrics.final_cash:,.2f} |",
        f"| Positions | {metrics.final_positions} |",
        f"| Tax Lots | {metrics.final_lots} |",
        "",
        "---",
        "",
        "## Configuration",
        "",
        "```yaml",
        f"initial_cash: {result.config.initial_cash}",
        f"cash_buffer_pct: {result.config.cash_buffer_pct}",
        f"min_trade_value: {result.config.min_trade_value}",
        f"max_turnover_pct: {result.config.max_turnover_pct}",
        f"rebalance_band_pct: {result.config.rebalance_band_pct}",
        f"tlh_loss_threshold: {result.config.tlh_loss_threshold}",
        f"tlh_wash_sale_days: {result.config.tlh_wash_sale_days}",
        f"rebalance_freq: {result.config.rebalance_freq}",
        "```",
        "",
    ])

    if include_trade_details and result.trades:
        lines.extend([
            "---",
            "",
            "## Trade Details",
            "",
            "### Recent Trades (Last 20)",
            "",
            "| Date | Symbol | Side | Shares | Price | Value | Reason |",
            "|------|--------|------|--------|-------|-------|--------|",
        ])

        recent_trades = sorted(result.trades, key=lambda t: t.timestamp, reverse=True)[:20]
        for trade in recent_trades:
            lines.append(
                f"| {trade.timestamp.strftime('%Y-%m-%d')} | {trade.symbol} | "
                f"{trade.side.value} | {trade.shares:.2f} | ${trade.price:.2f} | "
                f"${trade.value:.2f} | {trade.reason.value} |"
            )

        lines.append("")

    lines.extend([
        "---",
        "",
        "## Assumptions & Limitations",
        "",
        "1. **Survivorship Bias**: Uses current S&P 500 constituents for the entire "
        "backtest period. Historical index changes are not reflected.",
        "",
        "2. **Execution**: Assumes perfect execution at closing prices. No slippage "
        "or market impact is modeled.",
        "",
        "3. **Corporate Actions**: Adjusted prices are used, which account for "
        "splits and dividends, but complex corporate actions may not be reflected.",
        "",
        "4. **Wash Sales**: Wash sale risks are flagged but not fully enforced. "
        "A 30-day window is tracked.",
        "",
        "5. **Transaction Costs**: No transaction costs or bid-ask spreads are included.",
        "",
        "---",
        "",
        f"*Report generated by di-pilot v0.2.0*",
    ])

    return "\n".join(lines)


def generate_quick_summary(
    result: Union[BacktestResult, ForwardTestResult],
) -> str:
    """
    Generate a quick text summary for console output.

    Args:
        result: Simulation result

    Returns:
        Formatted summary string
    """
    metrics = calculate_metrics(
        snapshots=result.snapshots,
        trades=result.trades,
        initial_cash=result.config.initial_cash,
    )

    is_backtest = isinstance(result, BacktestResult)
    sim_type = "Backtest" if is_backtest else "Forward Test"

    lines = [
        f"\n{'='*60}",
        f"  {sim_type} Summary: {result.run_id}",
        f"{'='*60}",
        "",
        f"  Period:      {metrics.start_date} to {metrics.end_date}",
        f"  Days:        {metrics.trading_days}",
        "",
        f"  Initial:     ${float(result.config.initial_cash):>14,.2f}",
        f"  Final:       ${metrics.final_value:>14,.2f}",
        f"  Return:      {metrics.total_return:>14.2%}",
        f"  CAGR:        {metrics.cagr:>14.2%}",
        "",
        f"  Volatility:  {metrics.annualized_volatility:>14.2%}",
        f"  Sharpe:      {metrics.sharpe_ratio:>14.2f}",
        f"  Max DD:      {metrics.max_drawdown:>14.2%}",
        "",
        f"  Trades:      {metrics.total_trades:>14}",
        f"  Turnover:    {metrics.total_turnover:>14.1%}",
        f"  TLH Saves:   ${metrics.harvested_losses:>13,.2f}",
        "",
        f"  Positions:   {metrics.final_positions:>14}",
        f"  Cash:        ${metrics.final_cash:>14,.2f}",
        f"{'='*60}\n",
    ]

    return "\n".join(lines)
