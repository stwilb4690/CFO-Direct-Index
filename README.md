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

## Scope
The project will implement a **shadow (paper) direct indexing system** capable of:

- Initializing portfolios from fresh cash into the S&P 500
- Simulating portfolio behavior over time
- Applying tax-loss harvesting rules
- Handling index additions and deletions
- Managing drift and tracking error
- Generating proposed trades (paper only)
- Producing analytics and decision logs
- Displaying results via an internal dashboard

No live trading or brokerage execution is permitted.

---

## Initialization Mode (Default)

The default simulation mode assumes portfolios begin with **100% cash** deployed into the S&P 500 on a specified start date.

### Behavior
- Portfolio starts with cash only
- Holdings are constructed using S&P 500 constituent weights on the start date
- Initial purchases occur on the start date
- All tax lots originate on the start date
- No legacy holdings or historical wash-sale constraints are assumed unless explicitly enabled

This mode is intended to:
- Establish a clean baseline
- Measure tracking error, turnover, and tax outcomes
- Evaluate strategy behavior over time

---

## Core Components (Planned)

- **Data Ingestion**
  - Portfolio definitions
  - Benchmark constituent data
  - Market price data
  - Corporate actions (if available)

- **Portfolio Engine**
  - Target weight construction
  - Lot-level holdings tracking
  - Drift monitoring

- **Tax-Loss Harvesting Engine**
  - Loss detection thresholds
  - Wash-sale avoidance
  - Lot selection rules
  - Trade deferral logic

- **Index Implementation Logic**
  - Handling S&P 500 additions and deletions
  - Conservative implementation buffers
  - Liquidity and tracking error controls

- **Simulation Engine**
  - Daily or periodic rebalancing
  - Paper trade generation
  - Reproducible runs

- **Analytics & Reporting**
  - Realized and unrealized P&L
  - Harvested losses
  - Tracking error metrics
  - Turnover statistics
  - Wash-sale flags
  - Decision rationale logs

- **Dashboard**
  - Interactive internal dashboard
  - Portfolio summary views
  - Time-series metrics
  - Trade review panels

---

## Constraints & Guardrails

- No live order routing
- No brokerage integrations
- All trades are simulated only
- All decisions must be explainable and logged
- Human review assumed before any real-world action
- Designed for use in a regulated RIA environment

---

## Technology Assumptions

- Python-based implementation
- File-based inputs (CSV/Parquet)
- Local execution
- Dashboard framework suitable for internal use (e.g., Streamlit)

---

## Status

Design and evaluation phase only.  
No production deployment.
