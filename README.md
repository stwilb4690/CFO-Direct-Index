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

## v0.1 Features

### Implemented
- **Portfolio Initialization**: Deploy cash into S&P 500 constituents by market-cap weight
- **Lot-Level Holdings Tracking**: Track shares, cost basis, and acquisition date per lot
- **Mark-to-Market Valuation**: Calculate current value and unrealized P&L
- **Tax-Loss Harvesting Detection**: Identify lots with losses exceeding threshold
- **Drift Analysis**: Compare portfolio weights to benchmark
- **Trade Proposal Generation**: Generate rebalance and TLH proposals (paper only)
- **Decision Logging**: Append-only JSONL audit log
- **CLI Interface**: Command-line tools for all operations

### Out of Scope (v0.2+)
- Wash-sale rule enforcement
- Index reconstitution handling
- Corporate actions (splits, dividends, spinoffs)
- Interactive dashboard
- Multi-day simulation engine
- Historical backfill of existing portfolios

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

## Quick Start

### 1. Initialize a Portfolio

```bash
di-pilot init \
  --config data/sample/portfolio_config.yaml \
  --constituents data/sample/sp500_constituents.csv \
  --prices data/sample/prices.csv \
  --date 2024-01-02
```

### 2. Calculate Valuation

```bash
di-pilot value \
  --portfolio-id P001 \
  --holdings output/holdings_P001_2024-01-02.csv \
  --prices data/sample/prices.csv \
  --date 2024-06-15
```

### 3. Analyze Drift

```bash
di-pilot drift \
  --portfolio-id P001 \
  --holdings output/holdings_P001_2024-01-02.csv \
  --constituents data/sample/sp500_constituents.csv \
  --prices data/sample/prices.csv \
  --date 2024-06-15
```

### 4. Identify TLH Candidates

```bash
di-pilot tlh \
  --portfolio-id P001 \
  --holdings output/holdings_P001_2024-01-02.csv \
  --prices data/sample/prices.csv \
  --date 2024-06-15 \
  --threshold 0.03
```

### 5. Generate Trade Proposals

```bash
di-pilot propose \
  --portfolio-id P001 \
  --holdings output/holdings_P001_2024-01-02.csv \
  --constituents data/sample/sp500_constituents.csv \
  --prices data/sample/prices.csv \
  --date 2024-06-15
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
│       ├── cli.py                 # Entry point
│       ├── config.py              # Config loading (YAML)
│       ├── models.py              # Data classes
│       ├── data/
│       │   ├── loaders.py         # CSV/Parquet I/O
│       │   └── schemas.py         # Column schemas
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
│       └── logging/
│           └── decision_log.py    # Audit logging
├── tests/
│   ├── conftest.py
│   ├── test_initialize.py
│   ├── test_valuation.py
│   ├── test_drift.py
│   └── test_tlh.py
├── data/
│   └── sample/                    # Sample input files
└── output/                        # Generated outputs (gitignored)
```

---

## Configuration

Portfolio configuration is defined in YAML:

```yaml
portfolio_id: P001
cash: 1000000              # Initial cash amount
start_date: 2024-01-02     # Portfolio start date

# Optional parameters
tlh_threshold: 0.03        # Tax-loss harvesting threshold (3%)
drift_threshold: 0.005     # Drift threshold (0.5%)
min_trade_value: 100       # Minimum trade value
output_dir: output         # Output directory
```

---

## Output Files

| Output | Format | Description |
|--------|--------|-------------|
| `holdings_{portfolio_id}_{date}.csv` | CSV | Lot-level holdings |
| `valuation_{portfolio_id}_{date}.csv` | CSV | Mark-to-market with P&L |
| `drift_report_{portfolio_id}_{date}.csv` | CSV | Per-symbol drift analysis |
| `tlh_candidates_{portfolio_id}_{date}.csv` | CSV | Tax-loss harvesting candidates |
| `trade_proposals_{portfolio_id}_{date}.csv` | CSV | Proposed trades |
| `decision_log.jsonl` | JSONL | Append-only audit log |

---

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=di_pilot --cov-report=term-missing

# Run specific test file
pytest tests/test_initialize.py -v
```

---

## Constraints & Guardrails

- **No live trading**: No brokerage API integrations
- **Paper only**: All trades are proposals requiring human review
- **Auditability**: All decisions logged with rationale
- **Deterministic**: Same inputs produce identical outputs
- **Conservative**: Default thresholds favor safety over optimization

---

## Technology Stack

- Python 3.11+
- `decimal.Decimal` for all monetary/share calculations
- pandas for data manipulation
- click for CLI
- pyyaml for configuration
- pytest for testing

---

## Status

**v0.1** - Initial implementation complete.

- Portfolio initialization from cash
- Lot-level holdings tracking
- Mark-to-market valuation
- TLH candidate detection
- Drift analysis
- Trade proposal generation
- Decision logging

No production deployment.
