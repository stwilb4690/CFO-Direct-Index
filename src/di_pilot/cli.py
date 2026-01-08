"""
Command-line interface for the Direct Indexing Shadow System.

Provides commands for:
- init: Initialize portfolio from cash
- value: Calculate mark-to-market valuation
- drift: Analyze drift vs benchmark
- tlh: Identify tax-loss harvesting candidates
- propose: Generate trade proposals
"""

import sys
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Optional

import click

from di_pilot.config import load_portfolio_config, ConfigurationError
from di_pilot.data import (
    load_benchmark_constituents,
    load_prices,
    load_holdings,
    save_holdings,
    save_valuations,
    save_drift_report,
    save_tlh_candidates,
    save_trade_proposals,
)
from di_pilot.data.loaders import DataLoadError
from di_pilot.portfolio import initialize_portfolio, value_portfolio
from di_pilot.analytics import (
    calculate_drift,
    identify_tlh_candidates,
    calculate_tracking_error,
)
from di_pilot.trading import generate_all_proposals
from di_pilot.logging import DecisionLogger, get_logger


@click.group()
@click.version_option(version="0.2.0", prog_name="di-pilot")
def main():
    """
    S&P 500 Direct Indexing Shadow System.

    A paper-only direct indexing simulation for evaluating S&P 500
    portfolio management strategies. No live trading.
    """
    pass


@main.command()
@click.option(
    "--config", "-c",
    required=True,
    type=click.Path(exists=True),
    help="Path to portfolio configuration YAML file",
)
@click.option(
    "--constituents", "-b",
    required=True,
    type=click.Path(exists=True),
    help="Path to S&P 500 constituents CSV file",
)
@click.option(
    "--prices", "-p",
    required=True,
    type=click.Path(exists=True),
    help="Path to prices CSV file",
)
@click.option(
    "--date", "-d",
    type=str,
    default=None,
    help="Start date (YYYY-MM-DD). Defaults to config start_date.",
)
@click.option(
    "--output-dir", "-o",
    type=click.Path(),
    default=None,
    help="Output directory. Defaults to config output_dir.",
)
def init(config: str, constituents: str, prices: str, date: Optional[str], output_dir: Optional[str]):
    """
    Initialize a portfolio from cash.

    Deploys cash into S&P 500 constituents according to benchmark weights,
    creating tax lots for each position.
    """
    click.echo("Loading configuration...")

    try:
        portfolio_config = load_portfolio_config(config)
    except ConfigurationError as e:
        click.echo(f"Error loading config: {e}", err=True)
        sys.exit(1)

    # Override date if provided
    start_date = portfolio_config.start_date
    if date:
        try:
            start_date = datetime.strptime(date, "%Y-%m-%d").date()
        except ValueError:
            click.echo(f"Invalid date format: {date}. Use YYYY-MM-DD.", err=True)
            sys.exit(1)

    # Set output directory
    out_dir = Path(output_dir or portfolio_config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Initialize logger
    logger = get_logger(out_dir / "decision_log.jsonl")
    logger.log_config_loaded(portfolio_config, config)

    click.echo(f"Loading benchmark constituents for {start_date}...")
    try:
        benchmark = load_benchmark_constituents(constituents, as_of_date=start_date)
    except DataLoadError as e:
        click.echo(f"Error loading constituents: {e}", err=True)
        sys.exit(1)

    click.echo(f"Loading prices for {start_date}...")
    try:
        price_data = load_prices(prices, as_of_date=start_date)
    except DataLoadError as e:
        click.echo(f"Error loading prices: {e}", err=True)
        sys.exit(1)

    click.echo(f"Initializing portfolio {portfolio_config.portfolio_id}...")
    click.echo(f"  Cash: ${portfolio_config.cash:,.2f}")
    click.echo(f"  Date: {start_date}")
    click.echo(f"  Constituents: {len(benchmark)}")

    try:
        lots, proposals = initialize_portfolio(
            config=portfolio_config,
            constituents=benchmark,
            prices=price_data,
        )
    except Exception as e:
        click.echo(f"Error initializing portfolio: {e}", err=True)
        sys.exit(1)

    # Log initialization
    logger.log_portfolio_initialized(portfolio_config, lots, proposals)

    # Save holdings
    holdings_path = out_dir / f"holdings_{portfolio_config.portfolio_id}_{start_date}.csv"
    save_holdings(lots, holdings_path)
    click.echo(f"  Holdings saved: {holdings_path}")

    # Save initial trade proposals
    proposals_path = out_dir / f"trade_proposals_{portfolio_config.portfolio_id}_{start_date}.csv"
    save_trade_proposals(proposals, proposals_path)
    click.echo(f"  Initial purchases: {proposals_path}")

    # Log proposals
    logger.log_trade_proposals_generated(portfolio_config.portfolio_id, proposals)

    # Summary
    total_invested = sum(lot.total_cost for lot in lots)
    click.echo()
    click.echo("Portfolio initialized:")
    click.echo(f"  Positions: {len(set(lot.symbol for lot in lots))}")
    click.echo(f"  Lots: {len(lots)}")
    click.echo(f"  Total invested: ${total_invested:,.2f}")
    click.echo(f"  Remaining cash: ${portfolio_config.cash - total_invested:,.2f}")


@main.command()
@click.option(
    "--portfolio-id", "-i",
    required=True,
    help="Portfolio ID",
)
@click.option(
    "--holdings", "-h",
    required=True,
    type=click.Path(exists=True),
    help="Path to holdings CSV file",
)
@click.option(
    "--prices", "-p",
    required=True,
    type=click.Path(exists=True),
    help="Path to prices CSV file",
)
@click.option(
    "--date", "-d",
    required=True,
    type=str,
    help="Valuation date (YYYY-MM-DD)",
)
@click.option(
    "--output-dir", "-o",
    type=click.Path(),
    default="output",
    help="Output directory",
)
def value(portfolio_id: str, holdings: str, prices: str, date: str, output_dir: str):
    """
    Calculate mark-to-market valuation.

    Values all lots at current prices and calculates unrealized P&L.
    """
    try:
        valuation_date = datetime.strptime(date, "%Y-%m-%d").date()
    except ValueError:
        click.echo(f"Invalid date format: {date}. Use YYYY-MM-DD.", err=True)
        sys.exit(1)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = get_logger(out_dir / "decision_log.jsonl")

    click.echo(f"Loading holdings for {portfolio_id}...")
    try:
        lots = load_holdings(holdings, portfolio_id=portfolio_id)
    except DataLoadError as e:
        click.echo(f"Error loading holdings: {e}", err=True)
        sys.exit(1)

    if not lots:
        click.echo(f"No holdings found for portfolio {portfolio_id}", err=True)
        sys.exit(1)

    click.echo(f"Loading prices for {valuation_date}...")
    try:
        price_data = load_prices(prices, as_of_date=valuation_date)
    except DataLoadError as e:
        click.echo(f"Error loading prices: {e}", err=True)
        sys.exit(1)

    click.echo("Calculating valuation...")
    try:
        valuation = value_portfolio(
            lots=lots,
            prices=price_data,
            valuation_date=valuation_date,
            portfolio_id=portfolio_id,
        )
    except Exception as e:
        click.echo(f"Error calculating valuation: {e}", err=True)
        sys.exit(1)

    # Log valuation
    logger.log_valuation_calculated(valuation)

    # Save valuation
    valuation_path = out_dir / f"valuation_{portfolio_id}_{valuation_date}.csv"
    save_valuations(valuation.lot_valuations, valuation_path)
    click.echo(f"  Valuation saved: {valuation_path}")

    # Summary
    return_pct = Decimal("0")
    if valuation.total_cost_basis != Decimal("0"):
        return_pct = valuation.total_unrealized_pnl / valuation.total_cost_basis

    click.echo()
    click.echo(f"Portfolio Valuation ({valuation_date}):")
    click.echo(f"  Market Value:  ${valuation.total_market_value:,.2f}")
    click.echo(f"  Cost Basis:    ${valuation.total_cost_basis:,.2f}")
    click.echo(f"  Unrealized P&L: ${valuation.total_unrealized_pnl:,.2f} ({return_pct:.2%})")
    click.echo(f"  Positions:     {len(valuation.position_summaries)}")
    click.echo(f"  Lots:          {len(valuation.lot_valuations)}")


@main.command()
@click.option(
    "--portfolio-id", "-i",
    required=True,
    help="Portfolio ID",
)
@click.option(
    "--holdings", "-h",
    required=True,
    type=click.Path(exists=True),
    help="Path to holdings CSV file",
)
@click.option(
    "--constituents", "-b",
    required=True,
    type=click.Path(exists=True),
    help="Path to S&P 500 constituents CSV file",
)
@click.option(
    "--prices", "-p",
    required=True,
    type=click.Path(exists=True),
    help="Path to prices CSV file",
)
@click.option(
    "--date", "-d",
    required=True,
    type=str,
    help="Analysis date (YYYY-MM-DD)",
)
@click.option(
    "--threshold", "-t",
    type=float,
    default=0.005,
    help="Drift threshold (default: 0.005 = 0.5%)",
)
@click.option(
    "--output-dir", "-o",
    type=click.Path(),
    default="output",
    help="Output directory",
)
def drift(
    portfolio_id: str,
    holdings: str,
    constituents: str,
    prices: str,
    date: str,
    threshold: float,
    output_dir: str,
):
    """
    Analyze portfolio drift vs benchmark.

    Compares current portfolio weights to S&P 500 benchmark weights.
    """
    try:
        analysis_date = datetime.strptime(date, "%Y-%m-%d").date()
    except ValueError:
        click.echo(f"Invalid date format: {date}. Use YYYY-MM-DD.", err=True)
        sys.exit(1)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = get_logger(out_dir / "decision_log.jsonl")
    drift_threshold = Decimal(str(threshold))

    click.echo(f"Loading data for {portfolio_id}...")
    try:
        lots = load_holdings(holdings, portfolio_id=portfolio_id)
        benchmark = load_benchmark_constituents(constituents, as_of_date=analysis_date)
        price_data = load_prices(prices, as_of_date=analysis_date)
    except DataLoadError as e:
        click.echo(f"Error loading data: {e}", err=True)
        sys.exit(1)

    if not lots:
        click.echo(f"No holdings found for portfolio {portfolio_id}", err=True)
        sys.exit(1)

    click.echo("Calculating valuation and drift...")
    valuation = value_portfolio(
        lots=lots,
        prices=price_data,
        valuation_date=analysis_date,
        portfolio_id=portfolio_id,
    )

    # Calculate current weights
    from di_pilot.portfolio.holdings import calculate_position_weights
    current_weights = calculate_position_weights(valuation.lot_valuations)

    # Calculate drift
    drift_analyses = calculate_drift(
        current_weights=current_weights,
        benchmark_constituents=benchmark,
        drift_threshold=drift_threshold,
    )

    # Log drift analysis
    logger.log_drift_analyzed(portfolio_id, drift_analyses, drift_threshold)

    # Save drift report
    drift_path = out_dir / f"drift_report_{portfolio_id}_{analysis_date}.csv"
    save_drift_report(drift_analyses, drift_path)
    click.echo(f"  Drift report saved: {drift_path}")

    # Summary
    exceeding = [d for d in drift_analyses if d.exceeds_threshold]
    tracking_error = calculate_tracking_error(drift_analyses)

    click.echo()
    click.echo(f"Drift Analysis ({analysis_date}):")
    click.echo(f"  Threshold: {drift_threshold:.2%}")
    click.echo(f"  Positions exceeding threshold: {len(exceeding)} / {len(drift_analyses)}")
    click.echo(f"  Tracking error (SSE): {tracking_error:.6f}")

    if exceeding:
        click.echo()
        click.echo("  Top positions exceeding threshold:")
        for da in exceeding[:5]:
            click.echo(f"    {da.symbol}: {da.absolute_drift:+.4%} drift")


@main.command()
@click.option(
    "--portfolio-id", "-i",
    required=True,
    help="Portfolio ID",
)
@click.option(
    "--holdings", "-h",
    required=True,
    type=click.Path(exists=True),
    help="Path to holdings CSV file",
)
@click.option(
    "--prices", "-p",
    required=True,
    type=click.Path(exists=True),
    help="Path to prices CSV file",
)
@click.option(
    "--date", "-d",
    required=True,
    type=str,
    help="Analysis date (YYYY-MM-DD)",
)
@click.option(
    "--threshold", "-t",
    type=float,
    default=0.03,
    help="Loss threshold (default: 0.03 = 3%)",
)
@click.option(
    "--output-dir", "-o",
    type=click.Path(),
    default="output",
    help="Output directory",
)
def tlh(
    portfolio_id: str,
    holdings: str,
    prices: str,
    date: str,
    threshold: float,
    output_dir: str,
):
    """
    Identify tax-loss harvesting candidates.

    Finds lots with unrealized losses exceeding the threshold.
    """
    try:
        analysis_date = datetime.strptime(date, "%Y-%m-%d").date()
    except ValueError:
        click.echo(f"Invalid date format: {date}. Use YYYY-MM-DD.", err=True)
        sys.exit(1)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = get_logger(out_dir / "decision_log.jsonl")
    loss_threshold = Decimal(str(threshold))

    click.echo(f"Loading data for {portfolio_id}...")
    try:
        lots = load_holdings(holdings, portfolio_id=portfolio_id)
        price_data = load_prices(prices, as_of_date=analysis_date)
    except DataLoadError as e:
        click.echo(f"Error loading data: {e}", err=True)
        sys.exit(1)

    if not lots:
        click.echo(f"No holdings found for portfolio {portfolio_id}", err=True)
        sys.exit(1)

    click.echo("Calculating valuation...")
    valuation = value_portfolio(
        lots=lots,
        prices=price_data,
        valuation_date=analysis_date,
        portfolio_id=portfolio_id,
    )

    click.echo("Identifying TLH candidates...")
    candidates = identify_tlh_candidates(
        valuations=valuation.lot_valuations,
        loss_threshold=loss_threshold,
    )

    # Log TLH analysis
    logger.log_tlh_candidates_identified(portfolio_id, candidates, loss_threshold)

    # Save candidates
    tlh_path = out_dir / f"tlh_candidates_{portfolio_id}_{analysis_date}.csv"
    save_tlh_candidates(candidates, tlh_path)
    click.echo(f"  TLH candidates saved: {tlh_path}")

    # Summary
    total_harvest = sum(abs(c.loss_amount) for c in candidates)
    short_term = [c for c in candidates if c.gain_type.value == "SHORT_TERM"]
    long_term = [c for c in candidates if c.gain_type.value == "LONG_TERM"]

    click.echo()
    click.echo(f"TLH Analysis ({analysis_date}):")
    click.echo(f"  Loss threshold: {loss_threshold:.1%}")
    click.echo(f"  Candidates found: {len(candidates)}")
    click.echo(f"  Potential harvest: ${total_harvest:,.2f}")
    click.echo(f"    Short-term: {len(short_term)} lots")
    click.echo(f"    Long-term: {len(long_term)} lots")

    if candidates:
        click.echo()
        click.echo("  Top candidates by loss:")
        for cand in candidates[:5]:
            click.echo(
                f"    {cand.lot_valuation.lot.symbol}: "
                f"${abs(cand.loss_amount):,.2f} ({cand.loss_pct:.2%})"
            )


@main.command()
@click.option(
    "--portfolio-id", "-i",
    required=True,
    help="Portfolio ID",
)
@click.option(
    "--holdings", "-h",
    required=True,
    type=click.Path(exists=True),
    help="Path to holdings CSV file",
)
@click.option(
    "--constituents", "-b",
    required=True,
    type=click.Path(exists=True),
    help="Path to S&P 500 constituents CSV file",
)
@click.option(
    "--prices", "-p",
    required=True,
    type=click.Path(exists=True),
    help="Path to prices CSV file",
)
@click.option(
    "--date", "-d",
    required=True,
    type=str,
    help="Proposal date (YYYY-MM-DD)",
)
@click.option(
    "--drift-threshold",
    type=float,
    default=0.005,
    help="Drift threshold (default: 0.005 = 0.5%)",
)
@click.option(
    "--tlh-threshold",
    type=float,
    default=0.03,
    help="TLH loss threshold (default: 0.03 = 3%)",
)
@click.option(
    "--output-dir", "-o",
    type=click.Path(),
    default="output",
    help="Output directory",
)
def propose(
    portfolio_id: str,
    holdings: str,
    constituents: str,
    prices: str,
    date: str,
    drift_threshold: float,
    tlh_threshold: float,
    output_dir: str,
):
    """
    Generate trade proposals.

    Creates proposals for rebalancing and tax-loss harvesting.
    All trades are proposals only - no execution occurs.
    """
    try:
        proposal_date = datetime.strptime(date, "%Y-%m-%d").date()
    except ValueError:
        click.echo(f"Invalid date format: {date}. Use YYYY-MM-DD.", err=True)
        sys.exit(1)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = get_logger(out_dir / "decision_log.jsonl")

    click.echo(f"Loading data for {portfolio_id}...")
    try:
        lots = load_holdings(holdings, portfolio_id=portfolio_id)
        benchmark = load_benchmark_constituents(constituents, as_of_date=proposal_date)
        price_data = load_prices(prices, as_of_date=proposal_date)
    except DataLoadError as e:
        click.echo(f"Error loading data: {e}", err=True)
        sys.exit(1)

    if not lots:
        click.echo(f"No holdings found for portfolio {portfolio_id}", err=True)
        sys.exit(1)

    click.echo("Calculating valuation...")
    valuation = value_portfolio(
        lots=lots,
        prices=price_data,
        valuation_date=proposal_date,
        portfolio_id=portfolio_id,
    )

    # Calculate drift
    from di_pilot.portfolio.holdings import calculate_position_weights
    current_weights = calculate_position_weights(valuation.lot_valuations)
    drift_analyses = calculate_drift(
        current_weights=current_weights,
        benchmark_constituents=benchmark,
        drift_threshold=Decimal(str(drift_threshold)),
    )

    # Identify TLH candidates
    candidates = identify_tlh_candidates(
        valuations=valuation.lot_valuations,
        loss_threshold=Decimal(str(tlh_threshold)),
    )

    click.echo("Generating trade proposals...")
    proposals = generate_all_proposals(
        portfolio_valuation=valuation,
        drift_analyses=drift_analyses,
        tlh_candidates=candidates,
        prices=price_data,
    )

    # Log proposals
    logger.log_trade_proposals_generated(portfolio_id, proposals)

    # Save proposals
    proposals_path = out_dir / f"trade_proposals_{portfolio_id}_{proposal_date}.csv"
    save_trade_proposals(proposals, proposals_path)
    click.echo(f"  Trade proposals saved: {proposals_path}")

    # Summary
    from di_pilot.trading.proposals import calculate_proposal_summary
    summary = calculate_proposal_summary(proposals)

    click.echo()
    click.echo(f"Trade Proposals ({proposal_date}):")
    click.echo(f"  Total proposals: {summary['total_proposals']}")
    click.echo(f"    Buys: {summary['buy_count']} (${summary['total_buy_value']:,.2f})")
    click.echo(f"    Sells: {summary['sell_count']} (${summary['total_sell_value']:,.2f})")
    click.echo(f"    TLH sells: {summary['tlh_proposals']}")
    click.echo(f"    Rebalance trades: {summary['rebalance_proposals']}")
    click.echo()
    click.echo("  Note: All trades are proposals only. Review before execution.")


# =============================================================================
# Simulation Commands (v0.2)
# =============================================================================


@main.command("simulate-backtest")
@click.option(
    "--start-date",
    required=True,
    type=str,
    help="Start date (YYYY-MM-DD)",
)
@click.option(
    "--end-date",
    required=True,
    type=str,
    help="End date (YYYY-MM-DD)",
)
@click.option(
    "--initial-cash",
    type=float,
    default=1000000,
    help="Initial cash amount (default: 1000000)",
)
@click.option(
    "--rebalance-freq",
    type=click.Choice(["daily", "weekly", "monthly", "quarterly"]),
    default="weekly",
    help="Rebalance frequency (default: weekly)",
)
@click.option(
    "--top-n",
    type=int,
    default=None,
    help="Limit to top N symbols (for faster testing)",
)
@click.option(
    "--output-dir", "-o",
    type=click.Path(),
    default="output",
    help="Output directory",
)
@click.option(
    "--no-cache",
    is_flag=True,
    help="Disable data caching",
)
def simulate_backtest(
    start_date: str,
    end_date: str,
    initial_cash: float,
    rebalance_freq: str,
    top_n: Optional[int],
    output_dir: str,
    no_cache: bool,
):
    """
    Run a backtest simulation.

    Simulates deploying cash into S&P 500 from start_date to end_date,
    with periodic rebalancing and tax-loss harvesting.

    Example:
        di-pilot simulate-backtest --start-date 2023-01-01 --end-date 2024-01-01
    """
    from datetime import timedelta

    try:
        start = datetime.strptime(start_date, "%Y-%m-%d").date()
        end = datetime.strptime(end_date, "%Y-%m-%d").date()
    except ValueError as e:
        click.echo(f"Invalid date format: {e}. Use YYYY-MM-DD.", err=True)
        sys.exit(1)

    if end <= start:
        click.echo("End date must be after start date.", err=True)
        sys.exit(1)

    click.echo(f"\n{'='*60}")
    click.echo(f"  Direct Indexing Backtest Simulation")
    click.echo(f"{'='*60}")
    click.echo(f"  Period: {start_date} to {end_date}")
    click.echo(f"  Initial Cash: ${initial_cash:,.2f}")
    click.echo(f"  Rebalance: {rebalance_freq}")
    if top_n:
        click.echo(f"  Symbols: Top {top_n} by weight")
    click.echo(f"{'='*60}\n")

    # Set up data provider
    from di_pilot.data.providers import YFinanceProvider, CachedDataProvider, FileCache
    from di_pilot.simulation import run_backtest, SimulationConfig, generate_report
    from di_pilot.simulation.report import generate_quick_summary

    try:
        provider = YFinanceProvider()
        if not no_cache:
            cache = FileCache("data/cache")
            provider = CachedDataProvider(provider, cache)
    except Exception as e:
        click.echo(f"Error initializing data provider: {e}", err=True)
        sys.exit(1)

    # Configure simulation
    config = SimulationConfig(
        initial_cash=Decimal(str(initial_cash)),
        rebalance_freq=rebalance_freq,
    )

    # Run backtest
    def progress(msg: str):
        click.echo(f"  {msg}")

    try:
        result = run_backtest(
            provider=provider,
            start_date=start,
            end_date=end,
            config=config,
            top_n_symbols=top_n,
            progress_callback=progress,
        )
    except Exception as e:
        click.echo(f"\nError running backtest: {e}", err=True)
        sys.exit(1)

    # Generate outputs
    out_dir = Path(output_dir) / result.run_id
    try:
        paths = generate_report(result, out_dir)
    except Exception as e:
        click.echo(f"\nError generating report: {e}", err=True)
        sys.exit(1)

    # Print summary
    click.echo(generate_quick_summary(result))

    click.echo(f"Outputs saved to: {out_dir}")
    for name, path in paths.items():
        click.echo(f"  - {name}: {path.name}")


@main.command("simulate-forward")
@click.option(
    "--start-date",
    required=True,
    type=str,
    help="Start date (YYYY-MM-DD)",
)
@click.option(
    "--initial-cash",
    type=float,
    default=1000000,
    help="Initial cash amount (default: 1000000)",
)
@click.option(
    "--rebalance-freq",
    type=click.Choice(["daily", "weekly", "monthly", "quarterly"]),
    default="weekly",
    help="Rebalance frequency (default: weekly)",
)
@click.option(
    "--simulate-days",
    type=int,
    default=0,
    help="Days to simulate forward (0 = just initialize)",
)
@click.option(
    "--top-n",
    type=int,
    default=None,
    help="Limit to top N symbols (for faster testing)",
)
@click.option(
    "--output-dir", "-o",
    type=click.Path(),
    default="output",
    help="Output directory",
)
@click.option(
    "--no-cache",
    is_flag=True,
    help="Disable data caching",
)
def simulate_forward(
    start_date: str,
    initial_cash: float,
    rebalance_freq: str,
    simulate_days: int,
    top_n: Optional[int],
    output_dir: str,
    no_cache: bool,
):
    """
    Run a forward test simulation.

    Deploys cash into S&P 500 on start_date and optionally simulates
    forward for the specified number of trading days.

    Example:
        di-pilot simulate-forward --start-date 2024-01-02 --simulate-days 30
    """
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d").date()
    except ValueError as e:
        click.echo(f"Invalid date format: {e}. Use YYYY-MM-DD.", err=True)
        sys.exit(1)

    click.echo(f"\n{'='*60}")
    click.echo(f"  Direct Indexing Forward Test Simulation")
    click.echo(f"{'='*60}")
    click.echo(f"  Start Date: {start_date}")
    click.echo(f"  Initial Cash: ${initial_cash:,.2f}")
    click.echo(f"  Rebalance: {rebalance_freq}")
    if simulate_days > 0:
        click.echo(f"  Simulate: {simulate_days} trading days")
    if top_n:
        click.echo(f"  Symbols: Top {top_n} by weight")
    click.echo(f"{'='*60}\n")

    # Set up data provider
    from di_pilot.data.providers import YFinanceProvider, CachedDataProvider, FileCache
    from di_pilot.simulation import run_forward_test, SimulationConfig, generate_report
    from di_pilot.simulation.report import generate_quick_summary

    try:
        provider = YFinanceProvider()
        if not no_cache:
            cache = FileCache("data/cache")
            provider = CachedDataProvider(provider, cache)
    except Exception as e:
        click.echo(f"Error initializing data provider: {e}", err=True)
        sys.exit(1)

    # Configure simulation
    config = SimulationConfig(
        initial_cash=Decimal(str(initial_cash)),
        rebalance_freq=rebalance_freq,
    )

    # Run forward test
    def progress(msg: str):
        click.echo(f"  {msg}")

    try:
        result = run_forward_test(
            provider=provider,
            start_date=start,
            config=config,
            top_n_symbols=top_n,
            simulate_days=simulate_days,
            progress_callback=progress,
        )
    except Exception as e:
        click.echo(f"\nError running forward test: {e}", err=True)
        sys.exit(1)

    # Generate outputs
    out_dir = Path(output_dir) / result.run_id
    try:
        paths = generate_report(result, out_dir)
    except Exception as e:
        click.echo(f"\nError generating report: {e}", err=True)
        sys.exit(1)

    # Print summary
    click.echo(generate_quick_summary(result))

    click.echo(f"Outputs saved to: {out_dir}")
    for name, path in paths.items():
        click.echo(f"  - {name}: {path.name}")


@main.command("quick-test")
@click.option(
    "--days",
    type=int,
    default=30,
    help="Number of days to backtest (default: 30)",
)
@click.option(
    "--top-n",
    type=int,
    default=20,
    help="Number of top symbols to use (default: 20)",
)
@click.option(
    "--output-dir", "-o",
    type=click.Path(),
    default="output",
    help="Output directory",
)
def quick_test(days: int, top_n: int, output_dir: str):
    """
    Run a quick sanity-check backtest.

    Uses a small universe (top 20 symbols) for fast execution.
    Good for verifying the system works correctly.

    Example:
        di-pilot quick-test --days 30 --top-n 20
    """
    from datetime import timedelta

    end = datetime.now().date() - timedelta(days=1)  # Yesterday
    start = end - timedelta(days=days + 7)  # Add buffer for weekends

    click.echo(f"\n{'='*60}")
    click.echo(f"  Quick Sanity Check Backtest")
    click.echo(f"{'='*60}")
    click.echo(f"  Period: ~{days} trading days")
    click.echo(f"  Symbols: Top {top_n} by weight")
    click.echo(f"{'='*60}\n")

    # Set up data provider
    from di_pilot.data.providers import YFinanceProvider, CachedDataProvider, FileCache
    from di_pilot.simulation import run_quick_backtest, generate_report
    from di_pilot.simulation.report import generate_quick_summary
    from di_pilot.simulation.backtest import run_quick_backtest

    try:
        provider = YFinanceProvider()
        cache = FileCache("data/cache")
        provider = CachedDataProvider(provider, cache)
    except Exception as e:
        click.echo(f"Error initializing data provider: {e}", err=True)
        sys.exit(1)

    click.echo("  Running backtest (this may take a minute)...")

    try:
        result = run_quick_backtest(
            provider=provider,
            start_date=start,
            end_date=end,
            initial_cash=Decimal("1000000"),
            top_n=top_n,
        )
    except Exception as e:
        click.echo(f"\nError running backtest: {e}", err=True)
        sys.exit(1)

    # Generate outputs
    out_dir = Path(output_dir) / result.run_id
    try:
        paths = generate_report(result, out_dir)
    except Exception as e:
        click.echo(f"\nError generating report: {e}", err=True)
        sys.exit(1)

    # Print summary
    click.echo(generate_quick_summary(result))

    click.echo(f"Outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
