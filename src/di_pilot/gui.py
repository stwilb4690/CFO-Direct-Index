"""
Streamlit GUI for Direct Indexing Pilot.

Launch with: di-pilot-gui
Or: streamlit run src/di_pilot/gui.py
"""

import json
import subprocess
import sys
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Direct Indexing Pilot",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


def get_project_root() -> Path:
    """Get the project root directory."""
    # Try to find the project root by looking for pyproject.toml
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent
    return Path.cwd()


def get_outputs_dir() -> Path:
    """Get the outputs directory."""
    return get_project_root() / "outputs"


def list_existing_runs() -> list[str]:
    """List existing simulation runs."""
    outputs_dir = get_outputs_dir()
    if not outputs_dir.exists():
        return []

    runs = []
    for path in sorted(outputs_dir.iterdir(), reverse=True):
        if path.is_dir() and (path / "metrics.json").exists():
            runs.append(path.name)
    return runs


def load_run_results(run_id: str) -> dict:
    """Load results from a simulation run."""
    run_dir = get_outputs_dir() / run_id
    results = {"run_id": run_id, "run_dir": run_dir}

    # Load metrics
    metrics_file = run_dir / "metrics.json"
    if metrics_file.exists():
        with open(metrics_file) as f:
            results["metrics"] = json.load(f)

    # Load trades
    trades_file = run_dir / "trades.csv"
    if trades_file.exists():
        results["trades"] = pd.read_csv(trades_file)

    # Load daily snapshots
    portfolio_file = run_dir / "portfolio_daily.csv"
    if portfolio_file.exists():
        results["portfolio"] = pd.read_csv(portfolio_file, parse_dates=["date"])

    # Load report
    report_file = run_dir / "run_report.md"
    if report_file.exists():
        results["report"] = report_file.read_text()

    return results


def run_simulation(
    sim_type: str,
    start_date: date,
    end_date: date | None,
    initial_cash: int,
    rebalance_freq: str,
    top_n: int | None,
    simulate_days: int | None,
) -> tuple[bool, str, str | None]:
    """Run a simulation and return (success, output, run_id)."""

    cmd = ["di-pilot"]

    if sim_type == "backtest":
        cmd.extend(["simulate-backtest"])
        cmd.extend(["--start-date", start_date.isoformat()])
        if end_date:
            cmd.extend(["--end-date", end_date.isoformat()])
        cmd.extend(["--initial-cash", str(initial_cash)])
        cmd.extend(["--rebalance-freq", rebalance_freq])
        if top_n and top_n > 0:
            cmd.extend(["--top-n", str(top_n)])

    elif sim_type == "forward":
        cmd.extend(["simulate-forward"])
        cmd.extend(["--start-date", start_date.isoformat()])
        cmd.extend(["--initial-cash", str(initial_cash)])
        if simulate_days and simulate_days > 0:
            cmd.extend(["--simulate-days", str(simulate_days)])

    elif sim_type == "quick":
        cmd.extend(["quick-test"])
        cmd.extend(["--start-date", start_date.isoformat()])
        if end_date:
            cmd.extend(["--end-date", end_date.isoformat()])
        if top_n and top_n > 0:
            cmd.extend(["--top-n", str(top_n)])

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=get_project_root(),
            timeout=600,  # 10 minute timeout
        )

        output = result.stdout + result.stderr

        # Try to extract run_id from output
        run_id = None
        for line in output.split("\n"):
            if "outputs/" in line or "Run ID:" in line:
                # Extract the run ID
                import re
                match = re.search(r"(backtest_|forward_|quick_)[\w\-]+", line)
                if match:
                    run_id = match.group(0)
                    break

        return result.returncode == 0, output, run_id

    except subprocess.TimeoutExpired:
        return False, "Simulation timed out after 10 minutes", None
    except Exception as e:
        return False, f"Error running simulation: {e}", None


def render_metrics_cards(metrics: dict):
    """Render metrics as card-style display."""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_return = metrics.get("total_return", 0)
        st.metric(
            "Total Return",
            f"{total_return:.2%}",
            delta=None,
        )

        cagr = metrics.get("cagr", 0)
        st.metric("CAGR", f"{cagr:.2%}")

    with col2:
        sharpe = metrics.get("sharpe_ratio", 0)
        st.metric("Sharpe Ratio", f"{sharpe:.2f}")

        volatility = metrics.get("annualized_volatility", 0)
        st.metric("Volatility", f"{volatility:.2%}")

    with col3:
        max_dd = metrics.get("max_drawdown", 0)
        st.metric(
            "Max Drawdown",
            f"{max_dd:.2%}",
            delta=None,
        )

        final_value = metrics.get("final_value", 0)
        st.metric("Final Value", f"${final_value:,.0f}")

    with col4:
        total_trades = metrics.get("total_trades", 0)
        st.metric("Total Trades", f"{total_trades:,}")

        tlh_trades = metrics.get("tlh_trades", 0)
        harvested = metrics.get("harvested_losses", 0)
        st.metric("TLH Trades", f"{tlh_trades:,}", delta=f"${harvested:,.0f} harvested")


def render_portfolio_chart(portfolio_df: pd.DataFrame):
    """Render portfolio value chart."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=portfolio_df["date"],
        y=portfolio_df["total_value"],
        mode="lines",
        name="Portfolio Value",
        line=dict(color="#2E86AB", width=2),
    ))

    fig.update_layout(
        title="Portfolio Value Over Time",
        xaxis_title="Date",
        yaxis_title="Value ($)",
        hovermode="x unified",
        template="plotly_white",
        height=400,
    )

    fig.update_yaxes(tickformat="$,.0f")

    st.plotly_chart(fig, use_container_width=True)


def render_returns_chart(portfolio_df: pd.DataFrame):
    """Render cumulative returns chart."""
    if "daily_return" not in portfolio_df.columns:
        return

    # Calculate cumulative return
    portfolio_df = portfolio_df.copy()
    portfolio_df["cumulative_return"] = (1 + portfolio_df["daily_return"]).cumprod() - 1

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=portfolio_df["date"],
        y=portfolio_df["cumulative_return"],
        mode="lines",
        name="Cumulative Return",
        line=dict(color="#28A745", width=2),
        fill="tozeroy",
        fillcolor="rgba(40, 167, 69, 0.1)",
    ))

    fig.update_layout(
        title="Cumulative Returns",
        xaxis_title="Date",
        yaxis_title="Return",
        hovermode="x unified",
        template="plotly_white",
        height=350,
    )

    fig.update_yaxes(tickformat=".1%")

    st.plotly_chart(fig, use_container_width=True)


def render_drawdown_chart(portfolio_df: pd.DataFrame):
    """Render drawdown chart."""
    if "total_value" not in portfolio_df.columns:
        return

    portfolio_df = portfolio_df.copy()
    portfolio_df["peak"] = portfolio_df["total_value"].cummax()
    portfolio_df["drawdown"] = (portfolio_df["total_value"] - portfolio_df["peak"]) / portfolio_df["peak"]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=portfolio_df["date"],
        y=portfolio_df["drawdown"],
        mode="lines",
        name="Drawdown",
        line=dict(color="#DC3545", width=2),
        fill="tozeroy",
        fillcolor="rgba(220, 53, 69, 0.2)",
    ))

    fig.update_layout(
        title="Drawdown",
        xaxis_title="Date",
        yaxis_title="Drawdown",
        hovermode="x unified",
        template="plotly_white",
        height=300,
    )

    fig.update_yaxes(tickformat=".1%")

    st.plotly_chart(fig, use_container_width=True)


def render_trades_analysis(trades_df: pd.DataFrame):
    """Render trades analysis."""
    if trades_df.empty:
        st.info("No trades recorded.")
        return

    col1, col2 = st.columns(2)

    with col1:
        # Trades by reason
        if "reason" in trades_df.columns:
            reason_counts = trades_df["reason"].value_counts()
            fig = px.pie(
                values=reason_counts.values,
                names=reason_counts.index,
                title="Trades by Reason",
                hole=0.4,
            )
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Trades by side
        if "side" in trades_df.columns:
            side_counts = trades_df["side"].value_counts()
            fig = px.bar(
                x=side_counts.index,
                y=side_counts.values,
                title="Trades by Side",
                color=side_counts.index,
                color_discrete_map={"buy": "#28A745", "sell": "#DC3545"},
            )
            fig.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    # Recent trades table
    st.subheader("Recent Trades")
    display_df = trades_df.tail(50).copy()
    if "shares" in display_df.columns:
        display_df["shares"] = display_df["shares"].round(4)
    if "price" in display_df.columns:
        display_df["price"] = display_df["price"].round(2)
    if "value" in display_df.columns:
        display_df["value"] = display_df["value"].round(2)
    st.dataframe(display_df, use_container_width=True, hide_index=True)


def render_sidebar():
    """Render the sidebar configuration."""
    st.sidebar.title("📈 DI Pilot")
    st.sidebar.markdown("---")

    # Simulation type
    sim_type = st.sidebar.selectbox(
        "Simulation Type",
        options=["backtest", "forward", "quick"],
        format_func=lambda x: {
            "backtest": "📊 Backtest",
            "forward": "🔮 Forward Test",
            "quick": "⚡ Quick Test",
        }.get(x, x),
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Configuration")

    # Date inputs
    default_start = date.today() - timedelta(days=365)
    default_end = date.today() - timedelta(days=1)

    start_date = st.sidebar.date_input(
        "Start Date",
        value=default_start,
        max_value=date.today(),
    )

    end_date = None
    if sim_type in ["backtest", "quick"]:
        end_date = st.sidebar.date_input(
            "End Date",
            value=default_end,
            max_value=date.today(),
        )

    # Initial cash
    initial_cash = st.sidebar.number_input(
        "Initial Cash ($)",
        min_value=10000,
        max_value=100000000,
        value=1000000,
        step=100000,
        format="%d",
    )

    # Rebalance frequency
    rebalance_freq = "weekly"
    if sim_type == "backtest":
        rebalance_freq = st.sidebar.selectbox(
            "Rebalance Frequency",
            options=["daily", "weekly", "monthly"],
            index=1,
        )

    # Top N symbols
    top_n = None
    if sim_type in ["backtest", "quick"]:
        use_top_n = st.sidebar.checkbox(
            "Limit to Top N Symbols",
            value=sim_type == "quick",
        )
        if use_top_n:
            top_n = st.sidebar.slider(
                "Number of Symbols",
                min_value=5,
                max_value=100,
                value=20 if sim_type == "quick" else 50,
            )

    # Simulate days (forward test only)
    simulate_days = None
    if sim_type == "forward":
        simulate_days = st.sidebar.number_input(
            "Simulate Days (0 for init only)",
            min_value=0,
            max_value=365,
            value=0,
        )

    return {
        "sim_type": sim_type,
        "start_date": start_date,
        "end_date": end_date,
        "initial_cash": initial_cash,
        "rebalance_freq": rebalance_freq,
        "top_n": top_n,
        "simulate_days": simulate_days,
    }


def main_page():
    """Render the main page."""
    st.title("Direct Indexing Pilot")
    st.markdown("*S&P 500 Direct Indexing Shadow System - Paper Trading Simulation*")

    # Get sidebar configuration
    config = render_sidebar()

    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["🚀 Run Simulation", "📂 View Results", "📖 Documentation"])

    with tab1:
        st.header("Run New Simulation")

        # Show current configuration
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info(f"**Type:** {config['sim_type'].title()}")
        with col2:
            st.info(f"**Cash:** ${config['initial_cash']:,}")
        with col3:
            if config["top_n"]:
                st.info(f"**Symbols:** Top {config['top_n']}")
            else:
                st.info("**Symbols:** All S&P 500")

        # Warnings
        if config["sim_type"] == "backtest" and not config["top_n"]:
            st.warning(
                "⚠️ Running a full backtest with all S&P 500 symbols may take several minutes. "
                "Consider using 'Limit to Top N Symbols' for faster testing."
            )

        # Run button
        if st.button("🚀 Run Simulation", type="primary", use_container_width=True):
            with st.spinner("Running simulation... This may take a few minutes."):
                success, output, run_id = run_simulation(
                    sim_type=config["sim_type"],
                    start_date=config["start_date"],
                    end_date=config["end_date"],
                    initial_cash=config["initial_cash"],
                    rebalance_freq=config["rebalance_freq"],
                    top_n=config["top_n"],
                    simulate_days=config["simulate_days"],
                )

            if success:
                st.success(f"✅ Simulation completed successfully!")
                if run_id:
                    st.info(f"**Run ID:** {run_id}")
                    st.session_state["last_run_id"] = run_id

                with st.expander("View Output"):
                    st.code(output)
            else:
                st.error("❌ Simulation failed")
                with st.expander("View Error Output"):
                    st.code(output)

    with tab2:
        st.header("View Simulation Results")

        # Get list of existing runs
        runs = list_existing_runs()

        if not runs:
            st.info("No simulation results found. Run a simulation first!")
        else:
            # Select run
            default_idx = 0
            if "last_run_id" in st.session_state:
                try:
                    default_idx = runs.index(st.session_state["last_run_id"])
                except ValueError:
                    pass

            selected_run = st.selectbox(
                "Select Run",
                options=runs,
                index=default_idx,
            )

            if selected_run:
                results = load_run_results(selected_run)

                # Metrics
                if "metrics" in results:
                    st.subheader("Performance Metrics")
                    render_metrics_cards(results["metrics"])

                # Charts
                if "portfolio" in results:
                    st.markdown("---")
                    render_portfolio_chart(results["portfolio"])

                    col1, col2 = st.columns(2)
                    with col1:
                        render_returns_chart(results["portfolio"])
                    with col2:
                        render_drawdown_chart(results["portfolio"])

                # Trades
                if "trades" in results:
                    st.markdown("---")
                    st.subheader("Trade Analysis")
                    render_trades_analysis(results["trades"])

                # Report
                if "report" in results:
                    st.markdown("---")
                    with st.expander("📄 Full Report"):
                        st.markdown(results["report"])

                # Raw metrics JSON
                if "metrics" in results:
                    with st.expander("📊 Raw Metrics (JSON)"):
                        st.json(results["metrics"])

    with tab3:
        st.header("Documentation")

        st.markdown("""
        ## Quick Start

        1. **Quick Test**: Use the "Quick Test" simulation type with 20 symbols to verify everything works
        2. **Backtest**: Run historical simulations to test strategy performance
        3. **Forward Test**: Initialize a portfolio for paper trading

        ## Simulation Types

        ### 📊 Backtest
        Simulates historical portfolio performance from a start date to end date.
        - Initializes portfolio from cash on start date
        - Rebalances at configured frequency
        - Executes tax-loss harvesting when opportunities arise
        - Tracks all trades and daily snapshots

        ### 🔮 Forward Test
        Initializes a portfolio for paper trading.
        - Creates initial positions based on current S&P 500 weights
        - Optionally simulates forward for specified days
        - Saves state for future resumption

        ### ⚡ Quick Test
        Fast sanity check with limited symbols.
        - Uses top N symbols by weight (default: 20)
        - Shorter time periods for faster execution
        - Good for verifying setup works correctly

        ## Configuration Options

        | Parameter | Description |
        |-----------|-------------|
        | Start Date | When to begin the simulation |
        | End Date | When to end (backtest/quick only) |
        | Initial Cash | Starting investment amount |
        | Rebalance Frequency | How often to rebalance (daily/weekly/monthly) |
        | Top N Symbols | Limit universe for faster execution |

        ## Output Files

        All results are saved to `outputs/<run_id>/`:
        - `metrics.json` - Performance metrics
        - `trades.csv` - All executed trades
        - `portfolio_daily.csv` - Daily snapshots
        - `run_report.md` - Human-readable summary

        ## Assumptions & Limitations

        - **Survivorship Bias**: Uses current S&P 500 constituents for entire period
        - **Execution**: Trades at closing prices, no slippage
        - **Costs**: No transaction costs modeled
        - **Wash Sales**: Flagged but not enforced
        """)


def main():
    """Entry point for the GUI."""
    # Check if running via streamlit
    if "streamlit" in sys.modules:
        main_page()
    else:
        # Launch streamlit
        import subprocess
        gui_path = Path(__file__).resolve()
        subprocess.run([
            sys.executable, "-m", "streamlit", "run",
            str(gui_path),
            "--server.headless", "true",
        ])


if __name__ == "__main__":
    main_page()
