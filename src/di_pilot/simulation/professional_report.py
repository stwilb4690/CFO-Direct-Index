"""
Professional report generation for backtest results.

Generates a comprehensive HTML report similar to institutional investment reports,
including:
- Portfolio summary with inception date and current value
- Realized gains/losses breakdown (short-term, long-term)
- Pre-tax and after-tax performance comparison
- Tax alpha calculations
- Performance charts (strategy vs benchmark)
- Period returns table (MTD, QTD, YTD, 1Y, 3Y, 5Y, Since Inception)
"""

import base64
from datetime import date, datetime, timedelta
from decimal import Decimal
from io import BytesIO
from pathlib import Path
from typing import Optional, Union
import math

import pandas as pd
import numpy as np

from di_pilot.simulation.backtest import BacktestResult
from di_pilot.simulation.forward import ForwardTestResult
from di_pilot.simulation.metrics import SimulationMetrics, calculate_metrics
from di_pilot.simulation.engine import SimulationConfig, TradeReason, TradeSide
from di_pilot.data.sectors import (
    calculate_sector_weights, 
    calculate_sector_drift, 
    GICS_SECTORS,
    SECTOR_MAPPINGS
)


# Tax rates (can be configured per client)
DEFAULT_SHORT_TERM_RATE = 0.37  # Federal + state approximation
DEFAULT_LONG_TERM_RATE = 0.238  # 20% + 3.8% NIIT


def calculate_tax_metrics(
    result: Union[BacktestResult, ForwardTestResult],
    short_term_rate: float = DEFAULT_SHORT_TERM_RATE,
    long_term_rate: float = DEFAULT_LONG_TERM_RATE,
) -> dict:
    """
    Calculate tax-related metrics from simulation results.
    
    Returns:
        Dictionary with realized gains/losses and tax estimates
    """
    trades = result.trades
    snapshots = result.snapshots
    initial_cash = float(result.config.initial_cash)
    
    # Separate short-term and long-term realized gains/losses
    # For now, we track harvested losses from TLH
    tlh_trades = [t for t in trades if t.reason == TradeReason.TLH_SELL]
    
    harvested_short_term = 0.0
    harvested_long_term = 0.0
    
    # Calculate based on holding period assumption
    # TLH typically captures short-term losses for maximum tax benefit
    total_harvested = sum(float(t.value) for t in tlh_trades)
    # Assume 70% short-term, 30% long-term for harvested losses
    harvested_short_term = total_harvested * 0.7
    harvested_long_term = total_harvested * 0.3
    
    # Tax savings estimate
    tax_savings_short = abs(harvested_short_term) * short_term_rate
    tax_savings_long = abs(harvested_long_term) * long_term_rate
    total_tax_savings = tax_savings_short + tax_savings_long
    
    # Pre-tax return
    if snapshots:
        final_value = float(snapshots[-1].total_value)
        pre_tax_return = (final_value - initial_cash) / initial_cash
    else:
        final_value = initial_cash
        pre_tax_return = 0.0
    
    # After-tax return estimate (adding back tax savings)
    after_tax_value = final_value + total_tax_savings
    after_tax_return = (after_tax_value - initial_cash) / initial_cash
    
    # Tax alpha = after-tax return - pre-tax return
    tax_alpha = after_tax_return - pre_tax_return
    
    return {
        "short_term_harvested": harvested_short_term,
        "long_term_harvested": harvested_long_term,
        "net_harvested": total_harvested,
        "tax_savings_short_term": tax_savings_short,
        "tax_savings_long_term": tax_savings_long,
        "total_tax_savings": total_tax_savings,
        "pre_tax_return": pre_tax_return,
        "after_tax_return": after_tax_return,
        "tax_alpha": tax_alpha,
        "final_value": final_value,
        "after_tax_value": after_tax_value,
        "short_term_rate": short_term_rate,
        "long_term_rate": long_term_rate,
    }


def calculate_period_returns(
    snapshots: list,
    end_date: date,
) -> dict:
    """
    Calculate returns for various time periods.
    
    Returns:
        Dictionary with MTD, QTD, YTD, 1Y, 3Y, 5Y, Since Inception returns
    """
    if not snapshots:
        return {}
    
    # Build DataFrame
    df = pd.DataFrame([
        {"date": s.date, "value": float(s.total_value)}
        for s in snapshots
    ])
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    
    periods = {}
    start_value = df["value"].iloc[0]
    end_value = df["value"].iloc[-1]
    
    # Calculate Since Inception
    inception_date = df.index[0].date()
    days_since_inception = (end_date - inception_date).days
    periods["since_inception"] = (end_value / start_value) - 1
    periods["inception_date"] = inception_date
    
    # Month-to-Date (MTD)
    mtd_start = date(end_date.year, end_date.month, 1)
    mtd_data = df[df.index >= pd.Timestamp(mtd_start)]
    if len(mtd_data) > 0 and mtd_data["value"].iloc[0] > 0:
        periods["mtd"] = (end_value / mtd_data["value"].iloc[0]) - 1
    else:
        periods["mtd"] = 0.0
    
    # Quarter-to-Date (QTD)
    quarter_month = ((end_date.month - 1) // 3) * 3 + 1
    qtd_start = date(end_date.year, quarter_month, 1)
    qtd_data = df[df.index >= pd.Timestamp(qtd_start)]
    if len(qtd_data) > 0 and qtd_data["value"].iloc[0] > 0:
        periods["qtd"] = (end_value / qtd_data["value"].iloc[0]) - 1
    else:
        periods["qtd"] = 0.0
    
    # Year-to-Date (YTD)
    ytd_start = date(end_date.year, 1, 1)
    ytd_data = df[df.index >= pd.Timestamp(ytd_start)]
    if len(ytd_data) > 0 and ytd_data["value"].iloc[0] > 0:
        periods["ytd"] = (end_value / ytd_data["value"].iloc[0]) - 1
    else:
        periods["ytd"] = 0.0
    
    # 1-Year
    one_year_start = end_date - timedelta(days=365)
    one_year_data = df[df.index >= pd.Timestamp(one_year_start)]
    if len(one_year_data) > 0 and one_year_data["value"].iloc[0] > 0:
        periods["1y"] = (end_value / one_year_data["value"].iloc[0]) - 1
    else:
        periods["1y"] = None
    
    # 3-Year (annualized)
    three_year_start = end_date - timedelta(days=365 * 3)
    three_year_data = df[df.index >= pd.Timestamp(three_year_start)]
    if len(three_year_data) > 0 and days_since_inception >= 365 * 3:
        ratio = end_value / three_year_data["value"].iloc[0]
        periods["3y"] = ratio ** (1/3) - 1  # Annualized
    else:
        periods["3y"] = None
    
    # 5-Year (annualized)
    five_year_start = end_date - timedelta(days=365 * 5)
    five_year_data = df[df.index >= pd.Timestamp(five_year_start)]
    if len(five_year_data) > 0 and days_since_inception >= 365 * 5:
        ratio = end_value / five_year_data["value"].iloc[0]
        periods["5y"] = ratio ** (1/5) - 1  # Annualized
    else:
        periods["5y"] = None
    
    return periods


def generate_performance_chart_html(
    snapshots: list,
    initial_cash: float,
    tax_metrics: dict,
    benchmark_prices: dict[date, Decimal] = None,
) -> str:
    """
    Generate HTML/JavaScript for performance chart using Plotly.
    
    Shows:
    - Strategy performance (pre-tax)
    - Strategy performance + tax alpha (after-tax equivalent)
    - Benchmark performance (e.g. S&P 500)
    """
    if not snapshots:
        return "<p>No data available for chart</p>"
    
    # Build data series
    dates = []
    strategy_values = []
    strategy_indexed = []
    benchmark_indexed = []
    
    # Build benchmark series
    bench_start_price = None
    if benchmark_prices:
        sorted_dates = sorted(benchmark_prices.keys())
        if sorted_dates:
            bench_start_price = float(benchmark_prices[sorted_dates[0]])

    for snap in snapshots:
        dates.append(snap.date.isoformat())
        strategy_values.append(float(snap.total_value))
        
        # Benchmark indexing
        if bench_start_price and snap.date in benchmark_prices:
            curr_bench = float(benchmark_prices[snap.date])
            benchmark_indexed.append(100 * curr_bench / bench_start_price)
        else:
            # Fallback if missing data: use last known or 100
            last_val = benchmark_indexed[-1] if benchmark_indexed else 100
            benchmark_indexed.append(last_val)
    
    # Index to 100 at start
    base_value = strategy_values[0] if strategy_values else 1
    strategy_indexed = [100 * v / base_value for v in strategy_values]
    
    # Calculate after-tax indexed (adding proportional tax benefit over time)
    total_tax_benefit_pct = tax_metrics.get("tax_alpha", 0) * 100
    after_tax_indexed = []
    for i, val in enumerate(strategy_indexed):
        # Pro-rate tax benefit over the period
        progress = i / max(len(strategy_indexed) - 1, 1)
        tax_benefit = progress * total_tax_benefit_pct
        after_tax_indexed.append(val + tax_benefit)
    
    # Generate Plotly chart
    chart_html = f"""
    <div id="performance-chart" style="width:100%; height:400px;"></div>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <script>
        var dates = {dates};
        var strategyValues = {strategy_indexed};
        var afterTaxValues = {after_tax_indexed};
        var benchmarkValues = {benchmark_indexed};
        
        var strategyTrace = {{
            x: dates,
            y: strategyValues,
            type: 'scatter',
            mode: 'lines',
            name: 'Strategy (Pre-Tax)',
            line: {{color: '#2E86AB', width: 2}}
        }};
        
        var afterTaxTrace = {{
            x: dates,
            y: afterTaxValues,
            type: 'scatter',
            mode: 'lines',
            name: 'Strategy + Tax Alpha',
            line: {{color: '#28A745', width: 2, dash: 'dot'}}
        }};
        
        var benchmarkTrace = {{
            x: dates,
            y: benchmarkValues,
            type: 'scatter',
            mode: 'lines',
            name: 'S&P 500 (Benchmark)',
            line: {{color: '#999999', width: 2, dash: 'dash'}}
        }};
        
        var layout = {{
            title: 'Portfolio Performance (Indexed to 100)',
            xaxis: {{title: 'Date'}},
            yaxis: {{title: 'Indexed Value'}},
            legend: {{x: 0, y: 1, orientation: 'h'}},
            hovermode: 'x unified',
            margin: {{t: 50, r: 50, b: 60, l: 60}}
        }};
        
        Plotly.newPlot('performance-chart', [strategyTrace, afterTaxTrace, benchmarkTrace], layout);
    </script>
    """
    
    return chart_html



def generate_analysis_section(
    result: Union[BacktestResult, ForwardTestResult],
    benchmark_return: float,
) -> str:
    """Generate analysis section calling out specific issues or highlights."""
    issues = []
    highlights = []
    
    # 1. Performance vs Benchmark
    strategy_return = float(result.total_return)
    active_return = strategy_return - benchmark_return
    
    if active_return < -0.02:  # Underperformed by > 2%
        issues.append(f"Strategy significantly underperformed benchmark by {active_return*100:.2f}%.")
    elif active_return > 0.02:
        highlights.append(f"Strategy outperformed benchmark by {active_return*100:.2f}%.")
        
    # 2. Unwanted Gains
    net_harvested = float(result.harvested_losses)
    if net_harvested < 0:  # Negative harvested means realized GAIN
        issues.append(f"Realized capital GAINS of ${abs(net_harvested):,.2f}. Ideally, the strategy should only harvest losses.")
        
    # 3. Cash Drag
    avg_cash = np.mean([float(s.cash) for s in result.snapshots])
    avg_cash_pct = avg_cash / float(result.config.initial_cash)
    if avg_cash_pct > 0.05:  # > 5% cash
        issues.append(f"High average cash balance ({avg_cash_pct*100:.1f}%) created cash drag.")

    # 4. Turnover
    trades_count = result.total_trades
    if trades_count > 500: # High turnover check
        issues.append(f"High trading activity ({trades_count} trades). Ensure transaction costs are manageable.")

    html = '<div class="section"><h2>Analysis & Notes</h2>'
    
    if issues:
        html += '<div style="background: #fff3f3; border-left: 4px solid #dc3545; padding: 15px; margin-bottom: 20px;">'
        html += '<h3 style="color: #dc3545; font-size: 14px; margin-bottom: 10px;">⚠️ Issues Detected</h3><ul>'
        for issue in issues:
            html += f'<li style="margin-bottom: 5px;">{issue}</li>'
        html += '</ul></div>'
        
    if highlights:
        html += '<div style="background: #f0f9ff; border-left: 4px solid #0066a1; padding: 15px;">'
        html += '<h3 style="color: #0066a1; font-size: 14px; margin-bottom: 10px;">✅ Highlights</h3><ul>'
        for highlight in highlights:
            html += f'<li style="margin-bottom: 5px;">{highlight}</li>'
        html += '</ul></div>'
        
    if not issues and not highlights:
        html += '<p>Strategy performed within expected parameters tracking the benchmark closely.</p>'
        
    html += '</div>'
    return html


def generate_professional_html_report(
    result: Union[BacktestResult, ForwardTestResult],
    metrics: SimulationMetrics,
    tax_metrics: dict,
    period_returns: dict,
) -> str:
    """
    Generate a professional HTML report similar to institutional investment reports.
    """
    is_backtest = isinstance(result, BacktestResult)
    sim_type = "Backtest" if is_backtest else "Forward Test"
    
    report_date = datetime.now().strftime("%B %d, %Y")
    inception_date = period_returns.get("inception_date", metrics.start_date)
    
    # Calculate benchmark metrics
    benchmark_total_return = 0.0
    if is_backtest:
        benchmark_total_return = float(result.benchmark_return)
    
    strategy_total_return = float(result.total_return)
    active_return = strategy_total_return - benchmark_total_return
    
    # Format helper
    def fmt_pct(val, decimals=2):
        if val is None:
            return "—"
        return f"{val * 100:.{decimals}f}%"
    
    def fmt_money(val):
        if val is None:
            return "—"
        return f"${val:,.2f}"
    
    def color_cls(val, invert=False):
        if val is None: return ""
        if invert:
            return 'negative' if val > 0 else 'positive' if val < 0 else ''
        return 'positive' if val > 0 else 'negative' if val < 0 else ''

    # Generate chart
    chart_html = generate_performance_chart_html(
        result.snapshots,
        float(result.config.initial_cash),
        tax_metrics,
        result.benchmark_prices if is_backtest else None,
    )
    
    # Generate analysis
    analysis_html = generate_analysis_section(result, benchmark_total_return)

    # Generate sector analysis
    if isinstance(result, BacktestResult):
        sector_html = generate_sector_analysis_html(result)
    else:
        sector_html = ""
    
    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Direct Indexing {sim_type} Report</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: 'Segoe UI', Arial, sans-serif;
            font-size: 14px;
            line-height: 1.5;
            color: #333;
            background: #fff;
            padding: 40px;
            max-width: 1100px;
            margin: 0 auto;
        }}
        .header {{
            border-bottom: 3px solid #0066a1;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }}
        .header h1 {{
            color: #0066a1;
            font-size: 24px;
            font-weight: 600;
            margin-bottom: 5px;
        }}
        .header .subtitle {{
            color: #666;
            font-size: 14px;
        }}
        .header .report-date {{
            float: right;
            text-align: right;
            color: #666;
        }}
        .summary-box {{
            background: #f8f9fa;
            border-left: 4px solid #0066a1;
            padding: 15px 20px;
            margin-bottom: 30px;
        }}
        .summary-box h2 {{
            color: #0066a1;
            font-size: 16px;
            margin-bottom: 10px;
        }}
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 20px;
        }}
        .summary-item {{
            text-align: center;
        }}
        .summary-item .label {{
            color: #666;
            font-size: 12px;
            text-transform: uppercase;
        }}
        .summary-item .value {{
            font-size: 24px;
            font-weight: 600;
            color: #0066a1;
        }}
        .section {{
            margin-bottom: 30px;
        }}
        .section h2 {{
            color: #0066a1;
            font-size: 16px;
            border-bottom: 2px solid #0066a1;
            padding-bottom: 8px;
            margin-bottom: 15px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 20px;
        }}
        th, td {{
            padding: 10px 12px;
            text-align: right;
            border-bottom: 1px solid #e0e0e0;
        }}
        th {{
            background: #f5f5f5;
            color: #333;
            font-weight: 600;
            font-size: 12px;
            text-transform: uppercase;
        }}
        td:first-child, th:first-child {{
            text-align: left;
        }}
        tr:hover {{
            background: #f9f9f9;
        }}
        .positive {{
            color: #28a745;
        }}
        .negative {{
            color: #dc3545;
        }}
        .highlight-row {{
            background: #e8f4f8;
            font-weight: 600;
        }}
        .tax-alpha-row {{
            background: #d4edda;
            font-weight: 600;
        }}
        .metric-card {{
            display: inline-block;
            background: #f8f9fa;
            border: 1px solid #e0e0e0;
            border-radius: 4px;
            padding: 15px 20px;
            margin: 5px;
            min-width: 150px;
            text-align: center;
        }}
        .metric-card .label {{
            font-size: 11px;
            color: #666;
            text-transform: uppercase;
        }}
        .metric-card .value {{
            font-size: 20px;
            font-weight: 600;
            margin-top: 5px;
        }}
        .two-columns {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
        }}
        .chart-container {{
            margin: 20px 0;
            border: 1px solid #e0e0e0;
            border-radius: 4px;
            padding: 20px;
        }}
        .footer {{
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #e0e0e0;
            font-size: 12px;
            color: #666;
        }}
        @media print {{
            body {{
                padding: 20px;
            }}
        }}
        .tax-alpha-row {
            background-color: #e3f2fd;
        }
        .sector-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
            margin-top: 15px;
        }
        .sector-table th, .sector-table td {
            text-align: left;
            padding: 8px;
            border-bottom: 1px solid #eee;
        }
        .sector-table th {
            font-weight: 600;
            background: #f8f9fa;
        }
        .sector-table td:not(:first-child) {
            text-align: right;
        }
        .sector-table th:not(:first-child) {
            text-align: right;
        }
        .tax-alpha-row {
            background-color: #e3f2fd;
        }
        .sector-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
            margin-top: 15px;
        }
        .sector-table th, .sector-table td {
            padding: 8px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        .sector-table th {
            font-weight: 600;
            background-color: #f8f9fa;
        }
        .sector-table td:not(:first-child), .sector-table th:not(:first-child) {
            text-align: right;
        }
    </style>
</head>
<body>
    <div class="header">
        <div class="report-date">
            <strong>{report_date}</strong>
        </div>
        <h1>Direct Indexing Portfolio {sim_type} Report</h1>
        <div class="subtitle">S&P 500 Tax-Managed Strategy</div>
    </div>
    
    <div class="summary-box">
        <div class="summary-grid">
            <div class="summary-item">
                <div class="label">Portfolio Inception</div>
                <div class="value">{inception_date}</div>
            </div>
            <div class="summary-item">
                <div class="label">Total Account Value</div>
                <div class="value">{fmt_money(metrics.final_value)}</div>
            </div>
            <div class="summary-item">
                <div class="label">Tax Rate</div>
                <div class="value">{fmt_pct(tax_metrics['short_term_rate'])} ST / {fmt_pct(tax_metrics['long_term_rate'])} LT</div>
            </div>
        </div>
    </div>
    
    <div class="two-columns">
        <div class="section">
            <h2>Benchmark Comparison</h2>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Strategy</th>
                    <th>S&P 500</th>
                    <th>Difference</th>
                </tr>
                <tr>
                    <td>Total Return</td>
                    <td>{fmt_pct(strategy_total_return)}</td>
                    <td>{fmt_pct(benchmark_total_return)}</td>
                    <td class="{color_cls(active_return)}">{fmt_pct(active_return)}</td>
                </tr>
                <tr>
                    <td>Initial Value</td>
                    <td>{fmt_money(result.initial_value)}</td>
                    <td>—</td>
                    <td>—</td>
                </tr>
                <tr>
                    <td>Current Value</td>
                    <td>{fmt_money(result.final_value)}</td>
                    <td>—</td>
                    <td>—</td>
                </tr>
            </table>

            <h2>Report Summary</h2>
            <h3 style="font-size: 14px; margin-bottom: 10px; color: #333;">Realized Gains / Losses ($)</h3>
            <table>
                <tr>
                    <th></th>
                    <th>YTD</th>
                    <th>Since Inception</th>
                </tr>
                <tr>
                    <td>Short Term</td>
                    <td class="{'negative' if tax_metrics['short_term_harvested'] < 0 else 'positive'}">{fmt_money(tax_metrics['short_term_harvested'])}</td>
                    <td class="{'negative' if tax_metrics['short_term_harvested'] < 0 else 'positive'}">{fmt_money(tax_metrics['short_term_harvested'])}</td>
                </tr>
                <tr>
                    <td>Long Term</td>
                    <td class="{'negative' if tax_metrics['long_term_harvested'] < 0 else 'positive'}">{fmt_money(tax_metrics['long_term_harvested'])}</td>
                    <td class="{'negative' if tax_metrics['long_term_harvested'] < 0 else 'positive'}">{fmt_money(tax_metrics['long_term_harvested'])}</td>
                </tr>
                <tr class="highlight-row">
                    <td>Net Realized</td>
                    <td class="{'negative' if tax_metrics['net_harvested'] < 0 else 'positive'}">{fmt_money(tax_metrics['net_harvested'])}</td>
                    <td class="{'negative' if tax_metrics['net_harvested'] < 0 else 'positive'}">{fmt_money(tax_metrics['net_harvested'])}</td>
                </tr>
            </table>
            
            <h3 style="font-size: 14px; margin-bottom: 10px; color: #333;">Tax Savings Estimate</h3>
            <table>
                <tr>
                    <td>Short-Term Tax Savings</td>
                    <td class="positive">{fmt_money(tax_metrics['tax_savings_short_term'])}</td>
                </tr>
                <tr>
                    <td>Long-Term Tax Savings</td>
                    <td class="positive">{fmt_money(tax_metrics['tax_savings_long_term'])}</td>
                </tr>
                <tr class="highlight-row">
                    <td>Total Tax Savings</td>
                    <td class="positive">{fmt_money(tax_metrics['total_tax_savings'])}</td>
                </tr>
            </table>
        </div>
        
        <div class="section">
            <h2>Pre-Tax / After-Tax Returns</h2>
            <table>
                <tr>
                    <th></th>
                    <th>MTD</th>
                    <th>QTD</th>
                    <th>YTD</th>
                    <th>Inception</th>
                </tr>
                <tr>
                    <td>Pre-Tax Return</td>
                    <td>{fmt_pct(period_returns.get('mtd'))}</td>
                    <td>{fmt_pct(period_returns.get('qtd'))}</td>
                    <td>{fmt_pct(period_returns.get('ytd'))}</td>
                    <td>{fmt_pct(tax_metrics['pre_tax_return'])}</td>
                </tr>
                <tr>
                    <td>After-Tax Return (Est.)</td>
                    <td>—</td>
                    <td>—</td>
                    <td>—</td>
                    <td>{fmt_pct(tax_metrics['after_tax_return'])}</td>
                </tr>
                <tr class="tax-alpha-row">
                    <td>Tax Alpha</td>
                    <td>—</td>
                    <td>—</td>
                    <td>—</td>
                    <td class="positive">{fmt_pct(tax_metrics['tax_alpha'])}</td>
                </tr>
            </table>
            
            <div style="margin-top: 20px;">
                <div class="metric-card">
                    <div class="label">Total Return</div>
                    <div class="value {'positive' if metrics.total_return > 0 else 'negative'}">{fmt_pct(metrics.total_return)}</div>
                </div>
                <div class="metric-card">
                    <div class="label">Sharpe Ratio</div>
                    <div class="value">{metrics.sharpe_ratio:.2f}</div>
                </div>
                <div class="metric-card">
                    <div class="label">Max Drawdown</div>
                    <div class="value negative">{fmt_pct(metrics.max_drawdown)}</div>
                </div>
            </div>
            
            <!-- Analysis Section inserted here -->
            {analysis_html}
            
            <!-- Sector Analysis -->
            {sector_html}
        </div>
    </div>
    
    <div class="section">
        <h2>Performance Chart</h2>
        <div class="chart-container">
            {chart_html}
        </div>
    </div>
    
    <div class="section">
        <h2>Performance Summary</h2>
        <table>
            <tr>
                <th>Strategy</th>
                <th>MTD</th>
                <th>QTD</th>
                <th>YTD</th>
                <th>1-Year</th>
                <th>3-Year</th>
                <th>5-Year</th>
                <th>Since Inception</th>
            </tr>
            <tr>
                <td>Direct Index Strategy (Pre-Tax)</td>
                <td>{fmt_pct(period_returns.get('mtd'))}</td>
                <td>{fmt_pct(period_returns.get('qtd'))}</td>
                <td>{fmt_pct(period_returns.get('ytd'))}</td>
                <td>{fmt_pct(period_returns.get('1y'))}</td>
                <td>{fmt_pct(period_returns.get('3y'))}</td>
                <td>{fmt_pct(period_returns.get('5y'))}</td>
                <td><strong>{fmt_pct(period_returns.get('since_inception'))}</strong></td>
            </tr>
            <tr class="tax-alpha-row">
                <td>Direct Index Strategy + Tax Alpha</td>
                <td>—</td>
                <td>—</td>
                <td>—</td>
                <td>—</td>
                <td>—</td>
                <td>—</td>
                <td><strong>{fmt_pct(tax_metrics['after_tax_return'])}</strong></td>
            </tr>
        </table>
    </div>
    
    <div class="section">
        <h2>Trading Activity</h2>
        <div class="two-columns">
            <table>
                <tr>
                    <td>Total Trades</td>
                    <td>{metrics.total_trades:,}</td>
                </tr>
                <tr>
                    <td>Buy Trades</td>
                    <td>{metrics.buy_trades:,}</td>
                </tr>
                <tr>
                    <td>Sell Trades</td>
                    <td>{metrics.sell_trades:,}</td>
                </tr>
                <tr>
                    <td>TLH Trades</td>
                    <td>{metrics.tlh_trades:,}</td>
                </tr>
            </table>
            <table>
                <tr>
                    <td>Total Turnover</td>
                    <td>{fmt_pct(metrics.total_turnover)}</td>
                </tr>
                <tr>
                    <td>Avg Daily Turnover</td>
                    <td>{fmt_pct(metrics.avg_daily_turnover, 4)}</td>
                </tr>
                <tr>
                    <td>Rebalance Trades</td>
                    <td>{metrics.rebalance_trades:,}</td>
                </tr>
                <tr>
                    <td>Harvested Losses</td>
                    <td>{fmt_money(metrics.harvested_losses)}</td>
                </tr>
            </table>
        </div>
    </div>
    
    <div class="section">
        <h2>Risk Metrics</h2>
        <table>
            <tr>
                <td>Annualized Volatility</td>
                <td>{fmt_pct(metrics.annualized_volatility)}</td>
            </tr>
            <tr>
                <td>Maximum Drawdown</td>
                <td class="negative">{fmt_pct(metrics.max_drawdown)}</td>
            </tr>
            <tr>
                <td>Max Drawdown Date</td>
                <td>{metrics.max_drawdown_date}</td>
            </tr>
            <tr>
                <td>Average Drawdown</td>
                <td class="negative">{fmt_pct(metrics.avg_drawdown)}</td>
            </tr>
            <tr>
                <td>CAGR</td>
                <td class="{'positive' if metrics.cagr > 0 else 'negative'}">{fmt_pct(metrics.cagr)}</td>
            </tr>
        </table>
    </div>
    
    <div class="section">
        <h2>Portfolio Holdings (Final)</h2>
        <table>
            <tr>
                <td>Total Value</td>
                <td>{fmt_money(metrics.final_value)}</td>
            </tr>
            <tr>
                <td>Cash Balance</td>
                <td>{fmt_money(metrics.final_cash)}</td>
            </tr>
            <tr>
                <td>Number of Positions</td>
                <td>{metrics.final_positions}</td>
            </tr>
            <tr>
                <td>Number of Tax Lots</td>
                <td>{metrics.final_lots}</td>
            </tr>
        </table>
    </div>
    
    <div class="footer">
        <p><strong>Run ID:</strong> {result.run_id}</p>
        <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p><em>Report generated by Direct Indexing Pilot v0.3.0</em></p>
    </div>
</body>
</html>
"""
    
    return html



def generate_sector_analysis_html(
    result: BacktestResult,
) -> str:
    """Generate HTML for sector allocation and drift analysis."""
    
    # Calculate Benchmark Sector Weights
    benchmark_weights = {}
    if result.constituents:
        # Convert constituents to holdings-like dict for sector calc
        # Assuming constituents list represents the target weights
        bench_holdings = {c.symbol: Decimal(str(c.weight)) for c in result.constituents}
        benchmark_weights = calculate_sector_weights(bench_holdings)
    
    # Get Current Portfolio Sector Weights (from final snapshot)
    if result.snapshots and hasattr(result.snapshots[-1], 'sector_weights'):
        portfolio_weights = result.snapshots[-1].sector_weights or {}
    else:
        portfolio_weights = {}
        
    # Calculate drift
    drift = calculate_sector_drift(portfolio_weights, benchmark_weights)
    
    # Generate HTML Table
    rows = []
    
    # Sort by absolute drift magnitude
    sorted_sectors = sorted(
        GICS_SECTORS, 
        key=lambda s: abs(drift.get(s, 0.0)), 
        reverse=True
    )
    
    for sector in sorted_sectors:
        port_w = portfolio_weights.get(sector, 0.0)
        bench_w = benchmark_weights.get(sector, 0.0)
        drift_w = drift.get(sector, 0.0)
        
        drift_cls = 'positive' if drift_w > 0 else 'negative' if drift_w < 0 else ''
        drift_sign = "+" if drift_w > 0 else ""
        
        # Highlight significant drift (>1%)
        row_style = "background-color: #fff3cd;" if abs(drift_w) > 0.01 else ""
        
        rows.append(f"""
            <tr style="{row_style}">
                <td>{sector}</td>
                <td>{port_w:.2%}</td>
                <td>{bench_w:.2%}</td>
                <td class="{drift_cls}"><strong>{drift_sign}{drift_w:.2%}</strong></td>
            </tr>
        """)
    
    rows_html = "\n".join(rows)
    
    return f"""
    <div class="section">
        <h2>Sector Allocation & Drift</h2>
        <p class="section-desc">Comparison of portfolio sector weights vs benchmark targets.</p>
        
        <table class="sector-table">
            <thead>
                <tr>
                    <th>Sector</th>
                    <th>Portfolio</th>
                    <th>Benchmark</th>
                    <th>Drift</th>
                </tr>
            </thead>
            <tbody>
                {rows_html}
            </tbody>
        </table>
        <p style="font-size: 12px; color: #666; margin-top: 10px;">
            * Drift > 1.0% highlighted. Positive drift means overweight, negative means underweight.
            Drift allows for tax-loss harvesting while maintaining broad correlation.
        </p>
    </div>
    """


def generate_professional_report(
    result: Union[BacktestResult, ForwardTestResult],
    output_dir: Path,
    short_term_rate: float = DEFAULT_SHORT_TERM_RATE,
    long_term_rate: float = DEFAULT_LONG_TERM_RATE,
) -> Path:
    """
    Generate a professional HTML report and save it.
    
    Args:
        result: Backtest or Forward test result
        output_dir: Directory to save the report
        short_term_rate: Short-term capital gains tax rate
        long_term_rate: Long-term capital gains tax rate
        
    Returns:
        Path to the generated HTML report
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Calculate all metrics
    metrics = calculate_metrics(
        snapshots=result.snapshots,
        trades=result.trades,
        initial_cash=result.config.initial_cash,
    )
    
    tax_metrics = calculate_tax_metrics(
        result=result,
        short_term_rate=short_term_rate,
        long_term_rate=long_term_rate,
    )
    
    end_date = result.snapshots[-1].date if result.snapshots else date.today()
    period_returns = calculate_period_returns(result.snapshots, end_date)
    
    # Generate HTML
    html_content = generate_professional_html_report(
        result=result,
        metrics=metrics,
        tax_metrics=tax_metrics,
        period_returns=period_returns,
    )
    
    # Save report
    report_path = output_dir / "professional_report.html"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    return report_path
