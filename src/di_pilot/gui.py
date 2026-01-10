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

from di_pilot.config import load_api_keys

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
    # Also consider `output/` (singular) since CLI historically saved there.
    candidate_dirs = [get_project_root() / "outputs", get_project_root() / "output"]

    runs_info: list[tuple[str, str]] = []  # (sort_key, run_name)
    seen = set()

    for out_dir in candidate_dirs:
        if not out_dir.exists():
            continue

        for path in out_dir.iterdir():
            if not path.is_dir():
                continue

            run_name = path.name
            if run_name in seen:
                continue
            seen.add(run_name)

            metrics_file = path / "metrics.json"
            sort_key = None

            if metrics_file.exists():
                try:
                    with open(metrics_file, "r") as f:
                        data = json.load(f)
                    # Prefer end_date then start_date
                    sort_key = data.get("end_date") or data.get("start_date")
                except Exception:
                    sort_key = None

            if not sort_key:
                # Fallback to directory modification time
                try:
                    sort_key = str(path.stat().st_mtime)
                except Exception:
                    sort_key = "0"

            runs_info.append((sort_key, run_name))

    # Sort runs by sort_key descending (newest first)
    runs_info.sort(key=lambda x: x[0], reverse=True)

    return [r[1] for r in runs_info]


def load_run_results(run_id: str) -> dict:
    """Load results from a simulation run."""
    # Support both `outputs/` and legacy `output/` directories
    candidate_dirs = [get_outputs_dir(), get_project_root() / "output"]
    run_dir = None
    for d in candidate_dirs:
        p = d / run_id
        if p.exists():
            run_dir = p
            break
    if run_dir is None:
        # Fallback to outputs/ path even if it doesn't exist
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
    provider: str = "eodhd",
    run_name: str | None = None,
) -> tuple[bool, str, str | None]:
    """Run a simulation and return (success, output, run_id)."""

    # Use the module entry point via the current Python interpreter so the
    # console script is not required to be installed in PATH.
    cmd = [sys.executable, "-m", "di_pilot.cli"]

    if sim_type == "backtest":
        cmd.extend(["simulate-backtest"])
        cmd.extend(["--start-date", start_date.isoformat()])
        if end_date:
            cmd.extend(["--end-date", end_date.isoformat()])
        cmd.extend(["--initial-cash", str(initial_cash)])
        cmd.extend(["--rebalance-freq", rebalance_freq])
        if top_n and top_n > 0:
            cmd.extend(["--top-n", str(top_n)])
        cmd.extend(["--provider", provider])
        if run_name:
            cmd.extend(["--run-name", run_name])

    elif sim_type == "forward":
        cmd.extend(["simulate-forward"])
        cmd.extend(["--start-date", start_date.isoformat()])
        cmd.extend(["--initial-cash", str(initial_cash)])
        if simulate_days and simulate_days > 0:
            cmd.extend(["--simulate-days", str(simulate_days)])
        cmd.extend(["--provider", provider])
        if run_name:
            cmd.extend(["--run-name", run_name])

    elif sim_type == "quick":
        cmd.extend(["quick-test"])
        # Calculate days from date range
        if end_date and start_date:
            days = (end_date - start_date).days
            cmd.extend(["--days", str(max(days, 5))])  # Minimum 5 days
        else:
            cmd.extend(["--days", "30"])  # Default 30 days
        if top_n and top_n > 0:
            cmd.extend(["--top-n", str(top_n)])
        cmd.extend(["--provider", provider])
        if run_name:
            cmd.extend(["--run-name", run_name])

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=get_project_root(),
            timeout=1800,  # 30 minute timeout (first run with 500 symbols can be slow)
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


def save_api_key_to_env(api_key: str) -> bool:
    """Save EODHD API key to .env file.
    
    Args:
        api_key: The API key to save
        
    Returns:
        True if successful, False otherwise
    """
    try:
        env_path = get_project_root() / ".env"
        
        # Strip whitespace from API key
        api_key = api_key.strip()
        
        # Read existing content
        existing_lines = []
        if env_path.exists():
            with open(env_path, "r") as f:
                existing_lines = f.readlines()
        
        # Update or add the API key
        key_found = False
        new_lines = []
        for line in existing_lines:
            if line.strip().startswith("EODHD_API_KEY="):
                new_lines.append(f"EODHD_API_KEY={api_key}\n")
                key_found = True
            else:
                new_lines.append(line)
        
        # Add key if not found
        if not key_found:
            if new_lines and not new_lines[-1].endswith("\n"):
                new_lines.append("\n")
            new_lines.append(f"EODHD_API_KEY={api_key}\n")
        
        # Write back
        with open(env_path, "w") as f:
            f.writelines(new_lines)
        
        return True
    except Exception:
        return False


def check_eodhd_api_key() -> tuple[bool, str]:
    """Check if EODHD API key is configured.

    Returns:
        Tuple of (is_configured, status_message)
    """
    try:
        api_keys = load_api_keys()
        key = api_keys.get("eodhd_api_key", "")
        if key and key != "your-api-key-here":
            # Mask the key for display
            masked = key[:4] + "..." + key[-4:] if len(key) > 8 else "****"
            return True, f"Configured ({masked})"
        return False, "Not configured"
    except Exception as e:
        return False, f"Error: {e}"


def test_eodhd_connection() -> tuple[bool, str]:
    """Test connection to EODHD API.

    Returns:
        Tuple of (success, message)
    """
    try:
        from di_pilot.data.providers import get_eodhd_provider
        import pandas as pd

        provider = get_eodhd_provider(use_cache=False)
        # Try to get a single day of SPY data as a connectivity test
        from datetime import date, timedelta
        test_date = date.today() - timedelta(days=7)
        end_date = date.today()

        prices = provider.get_prices(
            symbols=["SPY"],
            start_date=test_date,
            end_date=end_date,
        )

        # Support providers that return dicts (symbol -> records)
        if isinstance(prices, dict):
            if "SPY" in prices and len(prices["SPY"]) > 0:
                return True, f"Connected! Retrieved {len(prices['SPY'])} price records for SPY"
            return False, "Connected but no data returned"

        # Support providers that return pandas DataFrame
        if isinstance(prices, pd.DataFrame):
            if prices.empty:
                return False, "Connected but no data returned"

            # Count rows for SPY if symbol column exists
            if "symbol" in prices.columns:
                # Sum the boolean mask (fast and avoids summing DataFrame columns)
                count = int((prices["symbol"].astype(str).str.upper() == "SPY").sum())
            else:
                count = len(prices)

            if count > 0:
                return True, f"Connected! Retrieved {count} price records for SPY"
            return False, "Connected but no data returned"

        # Unknown response type
        return False, "Connected but no data returned"

    except Exception as e:
        import traceback
        error_msg = str(e)
        tb = traceback.format_exc()
        if "API key" in error_msg:
            return False, "API key not configured or invalid"
        elif "429" in error_msg or "rate limit" in error_msg.lower():
            return False, "Rate limited - try again later"
        else:
            # Include brief traceback to help debugging in GUI logs
            short_tb = tb.splitlines()[-3:]
            return False, f"Connection failed: {error_msg} -- {' | '.join(short_tb)}"


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

    # Data Provider selection
    st.sidebar.subheader("Data Provider")
    provider = st.sidebar.selectbox(
        "Provider",
        options=["eodhd"],
        format_func=lambda x: {
            "eodhd": "EODHD (API Key Required)",
        }.get(x, x),
    )

    # Show API key status and test button for EODHD
    if provider == "eodhd":
        is_configured, status_msg = check_eodhd_api_key()
        if is_configured:
            st.sidebar.success(f"API Key: {status_msg}")
        else:
            st.sidebar.warning(f"API Key: {status_msg}")
        
        # API Key input field
        with st.sidebar.expander("🔑 Configure API Key", expanded=not is_configured):
            with st.form("api_key_form"):
                api_key_input = st.text_input(
                    "EODHD API Key",
                    type="password",
                    placeholder="Paste your EODHD API key here",
                    help="Get your API key at https://eodhd.com/",
                )
                
                submitted = st.form_submit_button("💾 Save API Key", use_container_width=True)
                
                if submitted:
                    if api_key_input:
                        # Check for whitespace issues
                        has_whitespace = api_key_input != api_key_input.strip()
                        if has_whitespace:
                            st.warning("⚠️ Your API key had leading/trailing spaces - they have been removed.")
                        
                        if save_api_key_to_env(api_key_input):
                            st.success("✅ API key saved!")
                            st.rerun()
                        else:
                            st.error("❌ Failed to save. Check file permissions.")
                    else:
                        st.warning("Please enter an API key first.")

        if st.sidebar.button("Test Connection"):
            with st.sidebar:
                with st.spinner("Testing connection..."):
                    success, message = test_eodhd_connection()
                if success:
                    st.success(message)
                else:
                    st.error(message)

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


    # Date inputs - default to Jan 2025 for testing
    default_start = date(2025, 1, 1)
    default_end = date(2025, 1, 31)

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

    # Top N symbols - only for Quick Test mode (Backtest uses full S&P 500)
    top_n = None
    if sim_type == "quick":
        use_top_n = st.sidebar.checkbox(
            "Limit to Top N Symbols",
            value=True,  # Default to checked for quick test
            help="Quick Test mode uses top N symbols by weight for faster testing",
        )
        if use_top_n:
            top_n = st.sidebar.slider(
                "Number of Symbols",
                min_value=5,
                max_value=100,
                value=20,
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

    # Advanced Settings (TLH and Dividends)
    st.sidebar.markdown("---")
    with st.sidebar.expander("⚙️ Advanced Settings"):
        st.markdown("**TLH Behavior**")
        tlh_skip_replacement = st.checkbox(
            "Skip TLH Replacement Buys",
            value=False,
            help="When enabled, TLH sells just add cash (Fidelity pattern). "
                 "When disabled, immediately buy correlated replacements.",
        )
        
        st.markdown("**Dividend Simulation**")
        enable_dividends = st.checkbox(
            "Enable Dividend Simulation",
            value=False,
            help="Simulate quarterly dividend payments that add to cash balance.",
        )
        
        dividend_yield = 1.5
        if enable_dividends:
            dividend_yield = st.slider(
                "Average Dividend Yield (%)",
                min_value=0.5,
                max_value=4.0,
                value=1.5,
                step=0.1,
                help="Average S&P 500 yield is ~1.5%",
            )

    # Optional run name to prefix output folder
    run_name = st.sidebar.text_input(
        "Run Name (optional)",
        value="",
        help="Optional label to prefix the output folder name for easier identification",
    )

    # Cache management
    st.sidebar.markdown("---")
    st.sidebar.subheader("Data Cache")
    if st.sidebar.button("🗑️ Clear Data Cache", help="Clear cached price and constituent data to fetch fresh data"):
        try:
            from di_pilot.data.providers.cache import FileCache
            cache = FileCache()
            cache.clear()
            st.sidebar.success("Cache cleared! Fresh data will be fetched on next run.")
        except Exception as e:
            st.sidebar.error(f"Failed to clear cache: {e}")
    
    if st.sidebar.button("📥 Preload 3 Years Data", help="Download all S&P 500 data for 2022-2025 to speed up full backtests"):
        with st.sidebar:
            with st.spinner("Downloading S&P 500 data (this takes 10-20 min first time)..."):
                try:
                    from di_pilot.scripts.preload_data import preload_sp500_data
                    end = date.today() - timedelta(days=1)
                    start = date(end.year - 3, end.month, end.day)
                    stats = preload_sp500_data(start, end)
                    st.success(f"Downloaded {stats['price_records']:,} records for {stats['symbols']} symbols!")
                except Exception as e:
                    st.error(f"Preload failed: {e}")

    return {
        "sim_type": sim_type,
        "start_date": start_date,
        "end_date": end_date,
        "initial_cash": initial_cash,
        "rebalance_freq": rebalance_freq,
        "top_n": top_n,
        "simulate_days": simulate_days,
        "run_name": run_name.strip() if isinstance(run_name, str) and run_name.strip() != "" else None,
        "provider": provider,
        "tlh_skip_replacement": tlh_skip_replacement,
        "enable_dividends": enable_dividends,
        "dividend_yield": dividend_yield,
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
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.info(f"**Type:** {config['sim_type'].title()}")
        with col2:
            st.info(f"**Cash:** ${config['initial_cash']:,}")
        with col3:
            if config["top_n"]:
                st.info(f"**Symbols:** Top {config['top_n']}")
            else:
                st.info("**Symbols:** All S&P 500")
        with col4:
            provider_name = "EODHD"
            st.info(f"**Provider:** {provider_name}")

        # Warnings
        if config["sim_type"] == "backtest":
            st.warning(
                "⚠️ **Full Backtest Mode**: Running with all S&P 500 symbols (~500). "
                "This may take 10-15 minutes depending on data provider. "
                "Use **Quick Test** mode for faster iteration with a subset of symbols."
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
                    provider=config["provider"],
                    run_name=config.get("run_name"),
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
            # Create columns for run selection and actions
            col_select, col_actions = st.columns([3, 1])
            
            with col_select:
                # Select run - default to most recent
                default_idx = 0
                if "last_run_id" in st.session_state:
                    try:
                        default_idx = runs.index(st.session_state["last_run_id"])
                    except ValueError:
                        pass

                selected_run = st.selectbox(
                    "Select Run (newest first)",
                    options=runs,
                    index=default_idx,
                    format_func=lambda x: f"📊 {x}" if x == runs[0] else x,
                )
            
            with col_actions:
                st.write("")  # Spacer
                if st.button("🗑️ Delete Run", help="Delete the selected run"):
                    if selected_run:
                        try:
                            import shutil
                            for out_dir in ["outputs", "output"]:
                                run_path = get_project_root() / out_dir / selected_run
                                if run_path.exists():
                                    shutil.rmtree(run_path)
                                    st.success(f"Deleted {selected_run}")
                                    st.rerun()
                        except Exception as e:
                            st.error(f"Failed to delete: {e}")

            if selected_run:
                results = load_run_results(selected_run)
                
                # Quick info bar
                st.markdown("---")
                if "metrics" in results:
                    m = results["metrics"]
                    info_cols = st.columns(5)
                    with info_cols[0]:
                        st.metric("Return", f"{m.get('total_return', 0):.1%}")
                    with info_cols[1]:
                        st.metric("vs S&P", f"{m.get('total_return', 0) - 0.1863:.1%}" if m.get('total_return') else "N/A")
                    with info_cols[2]:
                        st.metric("Sharpe", f"{m.get('sharpe_ratio', 0):.2f}")
                    with info_cols[3]:
                        st.metric("Max DD", f"{m.get('max_drawdown', 0):.1%}")
                    with info_cols[4]:
                        st.metric("Trades", f"{m.get('total_trades', 0)}")
                
                # Open HTML report button
                run_dir = results.get("run_dir")
                if run_dir:
                    html_report = run_dir / "professional_report.html"
                    if html_report.exists():
                        st.markdown(f"📄 **[Open Professional Report](file:///{html_report})**")

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
                    with st.expander("📄 Full Markdown Report"):
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
