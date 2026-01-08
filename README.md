# Direct Indexing Pilot (Internal)

## Purpose
This repository supports an internal evaluation of managing S&P 500 direct indexing portfolios in-house rather than through a third-party provider.

The goal is to replicate and evaluate a rules-based direct indexing strategy while maintaining:
- Tax efficiency
- Benchmark fidelity to the S&P 500
- Conservative compliance posture
- Human-in-the-loop execution

This repository is for research, simulation, and evaluation purposes only.

---

## Version History

| Version | Features |
|---------|----------|
| v0.1 | Portfolio init, holdings tracking, TLH detection, drift analysis, trade proposals |
| v0.2 | Backtest engine, forward test engine, data providers, metrics calculation, reports |
| v0.3 | Web-based GUI for running simulations without code |

---

## v0.3 Features (NEW)

### Web-Based GUI

Run simulations through a user-friendly web interface - no code required!

**Launch the GUI:**
```bash
di-pilot gui
```

Or with a custom port:
```bash
di-pilot gui --port 8080
```

**Features:**
- Configure and run backtests, forward tests, and quick tests
- View results with interactive charts (portfolio value, returns, drawdowns)
- Analyze trades by type and reason
- Export metrics and reports
- Browse historical simulation runs

**Screenshot of GUI capabilities:**
- Sidebar configuration for simulation parameters
- Performance metrics cards (return, Sharpe, drawdown, etc.)
- Interactive portfolio value and returns charts
- Trade analysis with pie charts and tables
- Full documentation built-in

---

## v0.2 Features

### Simulation Capabilities

- **Backtest**: Simulate historical portfolio performance from any start date
- **Forward Test**: Initialize portfolio today and track going forward (paper trading)
- **Data Providers**: Pluggable data source interface with yfinance (free, no API key)
- **Caching**: File-based cache for reproducibility and faster reruns
- **Metrics**: CAGR, volatility, Sharpe ratio, max drawdown, tracking error
- **Reports**: Markdown summaries, CSV trades/snapshots, JSON metrics

### Simulation Logic

1. Initialize portfolio from cash, allocating to constituents by weight
2. Daily: Mark portfolio to market, record snapshot
3. Weekly (configurable): Check rebalance triggers and TLH opportunities
4. Execute rebalance trades to bring weights within bands
5. Execute TLH trades for lots with losses exceeding threshold
6. Track wash sale windows (30-day flagging)

---

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd CFO-Direct-Index

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install package with dependencies
pip install -e .

# Install dev dependencies for testing
pip install -e ".[dev]"
```

---

## Quick Start: Simulation (v0.2)

### 1. Quick Sanity Check (Recommended First Run)

Run a fast backtest with top 20 symbols to verify setup:

```bash
di-pilot quick-test \
  --start-date 2024-01-02 \
  --end-date 2024-03-31 \
  --top-n 20
```

This runs in ~30 seconds and produces outputs in `outputs/quick_<date>_<id>/`.

### 2. Full Backtest

Run a backtest over a historical period:

```bash
di-pilot simulate-backtest \
  --start-date 2023-01-03 \
  --end-date 2024-01-02 \
  --initial-cash 1000000 \
  --rebalance-freq weekly
```

Options:
- `--start-date`: Backtest start date (YYYY-MM-DD)
- `--end-date`: Backtest end date (YYYY-MM-DD), default=today
- `--initial-cash`: Starting investment (default: $1,000,000)
- `--rebalance-freq`: `daily`, `weekly`, or `monthly`
- `--top-n`: Limit to top N symbols by weight (for faster testing)
- `--output-dir`: Custom output directory

### 3. Forward Test

Initialize a portfolio today and optionally simulate forward:

```bash
# Initialize only
di-pilot simulate-forward \
  --start-date 2024-06-01 \
  --initial-cash 1000000

# Initialize and simulate 30 days forward
di-pilot simulate-forward \
  --start-date 2024-01-02 \
  --initial-cash 1000000 \
  --simulate-days 30
```

---

## Simulation Outputs

All outputs are written to `outputs/<run_id>/`:

| File | Format | Description |
|------|--------|-------------|
| `trades.csv` | CSV | All executed trades with timestamp, symbol, side, shares, price, reason |
| `portfolio_daily.csv` | CSV | Daily snapshots: value, cash, positions, returns |
| `metrics.json` | JSON | Performance metrics for programmatic use |
| `run_report.md` | Markdown | Human-readable summary report |
| `state.json` | JSON | Forward test only: resumable state |

### Sample Output: metrics.json

```json
{
  "start_date": "2024-01-02",
  "end_date": "2024-12-31",
  "trading_days": 252,
  "total_return": 0.2534,
  "cagr": 0.2534,
  "annualized_volatility": 0.1823,
  "sharpe_ratio": 1.12,
  "max_drawdown": -0.0892,
  "total_trades": 1547,
  "tlh_trades": 89,
  "harvested_losses": 45230.50,
  "final_value": 1253400.00
}
```

---

## Configuration Parameters

Simulation behavior is controlled by these parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `initial_cash` | $1,000,000 | Starting investment amount |
| `cash_buffer_pct` | 1% | Cash reserve for trading friction |
| `min_trade_value` | $100 | Minimum trade size |
| `max_turnover_pct` | 10% | Maximum daily turnover |
| `rebalance_band_pct` | 2% | Drift threshold to trigger rebalance |
| `tlh_loss_threshold` | 3% | Loss percentage to trigger TLH |
| `tlh_wash_sale_days` | 30 | Wash sale window (flagging only) |
| `rebalance_freq` | weekly | Rebalance frequency |

---

## Data Sources

### Default: Yahoo Finance (yfinance)

- **Free**: No API key required
- **Prices**: Adjusted close prices (accounts for splits/dividends)
- **Constituents**: Scraped from Wikipedia (current S&P 500 members)

### Caching

Data is cached in `data/cache/` for:
- Reproducibility: Same inputs produce same outputs
- Speed: Avoid re-downloading on subsequent runs
- Offline: Work with previously cached data

To clear cache:
```bash
rm -rf data/cache/
```

---

## Assumptions & Limitations

### Survivorship Bias (Mode 1)

The backtest uses **current S&P 500 constituents for the entire historical period**. This introduces survivorship bias:
- Companies that were removed from the index are excluded
- Companies that were added later are included from the start

This is a known limitation. Mode 2 (historical reconstitution) is planned for future versions.

### Other Assumptions

1. **Execution**: Trades execute at closing prices. No slippage or market impact.
2. **Transaction Costs**: No commissions or bid-ask spreads modeled.
3. **Corporate Actions**: Adjusted prices handle splits/dividends, but complex actions (spinoffs, mergers) may not be fully reflected.
4. **Wash Sales**: The 30-day window is tracked and flagged, but not enforced.
5. **Fractional Shares**: Supported (6 decimal places) for accurate weight matching.

---

## v0.1 Features

### Portfolio Management
- **Portfolio Initialization**: Deploy cash into S&P 500 constituents by market-cap weight
- **Lot-Level Holdings Tracking**: Track shares, cost basis, and acquisition date per lot
- **Mark-to-Market Valuation**: Calculate current value and unrealized P&L

### Analytics
- **Tax-Loss Harvesting Detection**: Identify lots with losses exceeding threshold
- **Drift Analysis**: Compare portfolio weights to benchmark
- **Trade Proposal Generation**: Generate rebalance and TLH proposals (paper only)

### Infrastructure
- **Decision Logging**: Append-only JSONL audit log
- **CLI Interface**: Command-line tools for all operations

---

## CLI Commands Reference

### v0.1 Commands

```bash
# Initialize portfolio from cash
di-pilot init --config <config.yaml> --constituents <csv> --prices <csv> --date <YYYY-MM-DD>

# Calculate current valuation
di-pilot value --portfolio-id <id> --holdings <csv> --prices <csv> --date <YYYY-MM-DD>

# Analyze drift from benchmark
di-pilot drift --portfolio-id <id> --holdings <csv> --constituents <csv> --prices <csv> --date <YYYY-MM-DD>

# Find TLH candidates
di-pilot tlh --portfolio-id <id> --holdings <csv> --prices <csv> --date <YYYY-MM-DD> --threshold 0.03

# Generate trade proposals
di-pilot propose --portfolio-id <id> --holdings <csv> --constituents <csv> --prices <csv> --date <YYYY-MM-DD>
```

### v0.2 Commands

```bash
# Quick sanity check (top 20 symbols, 3 months)
di-pilot quick-test --start-date 2024-01-02 --end-date 2024-03-31

# Full backtest
di-pilot simulate-backtest --start-date 2023-01-03 --end-date 2024-01-02

# Forward test (paper trading)
di-pilot simulate-forward --start-date 2024-06-01 --simulate-days 30
```

### v0.3 Commands

```bash
# Launch web GUI (default port 8501)
di-pilot gui

# Launch on custom port
di-pilot gui --port 8080
```

---

## Project Structure

```
CFO-Direct-Index/
├── README.md
├── pyproject.toml
├── src/
│   └── di_pilot/
│       ├── __init__.py
│       ├── cli.py                 # CLI entry point
│       ├── gui.py                 # v0.3: Streamlit web GUI
│       ├── config.py              # Config loading (YAML)
│       ├── models.py              # Data classes
│       ├── data/
│       │   ├── loaders.py         # CSV/Parquet I/O
│       │   ├── schemas.py         # Column schemas
│       │   └── providers/         # v0.2: Data provider layer
│       │       ├── base.py        # Abstract provider interface
│       │       ├── cache.py       # File-based caching
│       │       └── yfinance_provider.py  # Yahoo Finance impl
│       ├── portfolio/
│       │   ├── initialize.py      # Cash → holdings
│       │   ├── holdings.py        # Lot tracking
│       │   └── valuation.py       # Mark-to-market
│       ├── analytics/
│       │   ├── drift.py           # Drift calculation
│       │   ├── tlh.py             # TLH detection
│       │   └── pnl.py             # P&L calculations
│       ├── trading/
│       │   └── proposals.py       # Trade proposals
│       ├── simulation/            # v0.2: Simulation engine
│       │   ├── engine.py          # Core simulation logic
│       │   ├── backtest.py        # Backtest orchestration
│       │   ├── forward.py         # Forward test orchestration
│       │   ├── metrics.py         # Performance calculations
│       │   └── report.py          # Report generation
│       └── logging/
│           └── decision_log.py    # Audit logging
├── tests/
│   ├── conftest.py
│   ├── test_initialize.py
│   ├── test_valuation.py
│   ├── test_drift.py
│   ├── test_tlh.py
│   └── test_simulation.py         # v0.2 simulation tests
├── data/
│   ├── sample/                    # Sample input files
│   └── cache/                     # v0.2: Cached data (gitignored)
└── outputs/                       # Generated outputs (gitignored)
```

---

## Testing

```bash
# Run all tests
pytest -o addopts=""

# Run simulation tests only
pytest tests/test_simulation.py -v -o addopts=""

# Run with coverage (requires pytest-cov)
pytest --cov=di_pilot --cov-report=term-missing
```

---

## Constraints & Guardrails

- **No live trading**: No brokerage API integrations
- **Paper only**: All trades are proposals requiring human review
- **Auditability**: All decisions logged with rationale
- **Deterministic**: Same inputs (with cache) produce identical outputs
- **Conservative**: Default thresholds favor safety over optimization

---

## Technology Stack

- Python 3.11+
- `decimal.Decimal` for all monetary/share calculations
- pandas / numpy for data manipulation
- yfinance for market data
- beautifulsoup4 for web scraping
- click for CLI
- streamlit / plotly for web GUI
- pyyaml for configuration
- pytest for testing

---

## Status

**v0.3** - Web GUI complete.

- Web-based GUI for running simulations (Streamlit)
- Interactive charts (portfolio value, returns, drawdowns)
- Trade analysis with visualizations
- Results browser for historical runs
- No code required for basic operations

**v0.2** - Simulation capabilities.

- Backtest engine with configurable parameters
- Forward test with state persistence
- Yahoo Finance data provider (free, no API key)
- File-based caching for reproducibility
- Performance metrics (CAGR, Sharpe, drawdown, etc.)

**Known Limitations:**
- Survivorship bias in backtest (uses current constituents)
- No transaction costs modeled
- Wash sales flagged but not enforced

No production deployment.
