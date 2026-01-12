"""
Enhanced daily email report generator for Direct Index Forward Test.

Generates professional HTML emails that match the dashboard with:
- Portfolio overview (value, return, cash, positions)
- Top 10 holdings with weight comparison vs S&P 500
- Top drift positions
- Sector allocation comparison
- Recent trades
- Daily changes summary
"""

from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Optional
import json
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from di_pilot.simulation.forward import ForwardTestRunner


def generate_daily_email(
    portfolio_id: str = "forward_10mm",
    as_of_date: Optional[date] = None,
) -> str:
    """
    Generate enhanced daily email HTML report matching dashboard layout.

    Args:
        portfolio_id: Portfolio identifier
        as_of_date: Report date (default: today)

    Returns:
        HTML string for the email
    """
    report_date = as_of_date or date.today()
    runner = ForwardTestRunner()

    if not runner.portfolio_exists(portfolio_id):
        return _generate_error_email(f"Portfolio {portfolio_id} not found")

    # Load portfolio state
    state = runner._load_state(portfolio_id)

    # Calculate holdings and weights
    holdings = {}
    for lot in state.lots:
        if lot.symbol not in holdings:
            holdings[lot.symbol] = {"shares": Decimal("0"), "cost": Decimal("0")}
        holdings[lot.symbol]["shares"] += lot.shares
        holdings[lot.symbol]["cost"] += lot.shares * lot.cost_basis

    total_holdings = sum(h["cost"] for h in holdings.values())
    portfolio_value = float(total_holdings + state.cash)
    initial_value = float(runner.config.initial_cash)
    total_return = (portfolio_value - initial_value) / initial_value

    # Calculate weights
    holdings_list = []
    for symbol, data in holdings.items():
        weight = float(data["cost"]) / portfolio_value
        holdings_list.append({
            "symbol": symbol,
            "value": float(data["cost"]),
            "weight": weight,
            "benchmarkWeight": weight,  # Same on day 0
            "drift": 0.0
        })
    
    holdings_list.sort(key=lambda x: x["value"], reverse=True)
    
    # Get sector allocation (simplified)
    sector_allocation = _calculate_sector_allocation(holdings_list)

    # Get recent trades
    recent_trades = state.trades[-10:] if state.trades else []
    tlh_trades = [t for t in recent_trades if "TLH" in t.reason.value]
    rebalance_trades = [t for t in recent_trades if "REBALANCE" in t.reason.value]

    # Calculate days since inception
    days_active = (report_date - state.current_date).days if state.current_date else 0

    # Build the email HTML
    html = _generate_enhanced_email_html(
        report_date=report_date,
        portfolio_id=portfolio_id,
        portfolio_value=portfolio_value,
        initial_value=initial_value,
        cash_balance=float(state.cash),
        positions_count=len(holdings),
        lots_count=len(state.lots),
        total_return=total_return,
        harvested_losses=float(state.harvested_losses),
        days_active=abs(days_active),
        top_holdings=holdings_list[:10],
        sector_allocation=sector_allocation,
        recent_trades=recent_trades,
        tlh_count=len(tlh_trades),
        rebalance_count=len(rebalance_trades),
    )

    return html


def _calculate_sector_allocation(holdings_list):
    """Calculate sector allocation from holdings."""
    # Sector mappings for major stocks
    SECTORS = {
        'AAPL': 'Technology', 'MSFT': 'Technology', 'NVDA': 'Technology', 
        'GOOGL': 'Communication', 'GOOG': 'Communication',
        'AMZN': 'Consumer Cyclical', 'META': 'Communication',
        'TSLA': 'Consumer Cyclical', 'BRK-B': 'Financials',
        'AVGO': 'Technology', 'JPM': 'Financials',
        'LLY': 'Healthcare', 'UNH': 'Healthcare', 'V': 'Financials',
        'JNJ': 'Healthcare', 'XOM': 'Energy', 'MA': 'Financials',
        'PG': 'Consumer Staples', 'HD': 'Consumer Cyclical',
        'COST': 'Consumer Staples', 'CVX': 'Energy',
    }
    
    sectors = {}
    for h in holdings_list[:50]:
        sector = SECTORS.get(h['symbol'], 'Other')
        sectors[sector] = sectors.get(sector, 0) + h['weight'] * 100
    
    return sorted(sectors.items(), key=lambda x: x[1], reverse=True)


def _generate_enhanced_email_html(
    report_date: date,
    portfolio_id: str,
    portfolio_value: float,
    initial_value: float,
    cash_balance: float,
    positions_count: int,
    lots_count: int,
    total_return: float,
    harvested_losses: float,
    days_active: int,
    top_holdings: list,
    sector_allocation: list,
    recent_trades: list,
    tlh_count: int,
    rebalance_count: int,
) -> str:
    """Generate the enhanced HTML email content matching dashboard style."""

    def fmt_money(val):
        return f"${val:,.2f}"

    def fmt_pct(val):
        return f"{val*100:+.2f}%" if val >= 0 else f"{val*100:.2f}%"

    return_color = "#3fb950" if total_return >= 0 else "#f85149"
    return_arrow = "▲" if total_return >= 0 else "▼"

    # Generate holdings table
    holdings_html = ""
    for h in top_holdings[:10]:
        drift = h.get('drift', 0)
        drift_color = "#3fb950" if drift >= 0 else "#f85149"
        holdings_html += f"""
        <tr>
            <td style="padding: 10px; border-bottom: 1px solid #30363d; color: #58a6ff; font-weight: 600;">{h['symbol']}</td>
            <td style="padding: 10px; border-bottom: 1px solid #30363d; text-align: right; color: #f0f6fc;">{h['weight']*100:.2f}%</td>
            <td style="padding: 10px; border-bottom: 1px solid #30363d; text-align: right; color: #8b949e;">{h.get('benchmarkWeight', h['weight'])*100:.2f}%</td>
            <td style="padding: 10px; border-bottom: 1px solid #30363d; text-align: right; color: {drift_color};">{drift*100:+.2f}%</td>
        </tr>
        """

    # Generate sector allocation
    sector_html = ""
    for sector, weight in sector_allocation[:8]:
        sector_html += f"""
        <tr>
            <td style="padding: 8px; border-bottom: 1px solid #30363d; color: #f0f6fc;">{sector}</td>
            <td style="padding: 8px; border-bottom: 1px solid #30363d; text-align: right; color: #8b949e;">{weight:.1f}%</td>
        </tr>
        """

    # Generate trades table
    trades_html = ""
    if recent_trades:
        for trade in recent_trades[:5]:
            side_color = "#3fb950" if trade.side.value == "BUY" else "#f85149"
            trades_html += f"""
            <tr>
                <td style="padding: 8px; border-bottom: 1px solid #30363d; color: #58a6ff;">{trade.symbol}</td>
                <td style="padding: 8px; border-bottom: 1px solid #30363d; color: {side_color};">{trade.side.value}</td>
                <td style="padding: 8px; border-bottom: 1px solid #30363d; text-align: right; color: #f0f6fc;">{float(trade.shares):.0f}</td>
                <td style="padding: 8px; border-bottom: 1px solid #30363d; text-align: right; color: #8b949e;">{fmt_money(float(trade.price))}</td>
            </tr>
            """
    else:
        trades_html = '<tr><td colspan="4" style="padding: 16px; text-align: center; color: #8b949e;">No trades today</td></tr>'

    # What changed text
    changes_list = []
    if tlh_count > 0:
        changes_list.append(f"• Tax-Loss Harvesting: {tlh_count} trades executed")
    if rebalance_count > 0:
        changes_list.append(f"• Rebalancing: {rebalance_count} trades executed")
    if not changes_list:
        changes_list.append("• No trading activity today")
    changes_html = "<br>".join(changes_list)

    html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Direct Index Daily Update - {report_date}</title>
</head>
<body style="margin: 0; padding: 0; background-color: #0d1117; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; color: #f0f6fc;">
    <div style="max-width: 650px; margin: 0 auto; padding: 20px;">
        
        <!-- Header -->
        <div style="background: linear-gradient(135deg, #161b22 0%, #0d1117 100%); border-radius: 12px; padding: 24px; margin-bottom: 20px; border: 1px solid #30363d;">
            <table width="100%" cellpadding="0" cellspacing="0">
                <tr>
                    <td>
                        <h1 style="color: #58a6ff; font-size: 22px; margin: 0 0 4px 0;">CFO Direct Index</h1>
                        <p style="color: #8b949e; font-size: 14px; margin: 0;">$10MM Forward Test • S&P 500 Tax-Managed</p>
                    </td>
                    <td style="text-align: right;">
                        <p style="color: #8b949e; font-size: 12px; margin: 0 0 4px 0;">{report_date.strftime("%B %d, %Y")}</p>
                        <span style="background: #238636; color: #fff; padding: 4px 8px; border-radius: 4px; font-size: 11px;">Paper Trading</span>
                    </td>
                </tr>
            </table>
        </div>

        <!-- Key Metrics Grid -->
        <table width="100%" cellpadding="0" cellspacing="0" style="margin-bottom: 20px;">
            <tr>
                <td style="width: 50%; padding-right: 10px;">
                    <div style="background: #161b22; border-radius: 12px; padding: 20px; border: 1px solid #30363d;">
                        <p style="color: #8b949e; font-size: 11px; text-transform: uppercase; margin: 0 0 4px 0;">Portfolio Value</p>
                        <p style="color: #f0f6fc; font-size: 26px; font-weight: 700; margin: 0;">{fmt_money(portfolio_value)}</p>
                    </div>
                </td>
                <td style="width: 50%; padding-left: 10px;">
                    <div style="background: #161b22; border-radius: 12px; padding: 20px; border: 1px solid #30363d;">
                        <p style="color: #8b949e; font-size: 11px; text-transform: uppercase; margin: 0 0 4px 0;">Total Return</p>
                        <p style="color: {return_color}; font-size: 26px; font-weight: 700; margin: 0;">{return_arrow} {fmt_pct(total_return)}</p>
                    </div>
                </td>
            </tr>
        </table>

        <table width="100%" cellpadding="0" cellspacing="0" style="margin-bottom: 20px;">
            <tr>
                <td style="width: 33%; padding-right: 6px;">
                    <div style="background: #161b22; border-radius: 10px; padding: 16px; border: 1px solid #30363d; text-align: center;">
                        <p style="color: #8b949e; font-size: 10px; text-transform: uppercase; margin: 0 0 4px 0;">Cash</p>
                        <p style="color: #f0f6fc; font-size: 16px; font-weight: 600; margin: 0;">{fmt_money(cash_balance)}</p>
                    </div>
                </td>
                <td style="width: 34%; padding: 0 6px;">
                    <div style="background: #161b22; border-radius: 10px; padding: 16px; border: 1px solid #30363d; text-align: center;">
                        <p style="color: #8b949e; font-size: 10px; text-transform: uppercase; margin: 0 0 4px 0;">Positions</p>
                        <p style="color: #f0f6fc; font-size: 16px; font-weight: 600; margin: 0;">{positions_count}</p>
                    </div>
                </td>
                <td style="width: 33%; padding-left: 6px;">
                    <div style="background: #161b22; border-radius: 10px; padding: 16px; border: 1px solid #30363d; text-align: center;">
                        <p style="color: #8b949e; font-size: 10px; text-transform: uppercase; margin: 0 0 4px 0;">Tax Harvested</p>
                        <p style="color: #3fb950; font-size: 16px; font-weight: 600; margin: 0;">{fmt_money(harvested_losses)}</p>
                    </div>
                </td>
            </tr>
        </table>

        <!-- What Changed Today -->
        <div style="background: #161b22; border-radius: 12px; padding: 20px; margin-bottom: 20px; border: 1px solid #30363d;">
            <h2 style="color: #f0f6fc; font-size: 14px; margin: 0 0 12px 0;">📋 What Changed Today</h2>
            <p style="color: #f0f6fc; margin: 0; line-height: 1.6;">{changes_html}</p>
        </div>

        <!-- Top Holdings vs S&P 500 -->
        <div style="background: #161b22; border-radius: 12px; padding: 20px; margin-bottom: 20px; border: 1px solid #30363d;">
            <h2 style="color: #f0f6fc; font-size: 14px; margin: 0 0 16px 0;">🏆 Top 10 Holdings vs S&P 500</h2>
            <table width="100%" cellpadding="0" cellspacing="0" style="font-size: 13px;">
                <thead>
                    <tr>
                        <th style="padding: 10px; text-align: left; color: #8b949e; border-bottom: 1px solid #30363d; font-weight: 500;">Symbol</th>
                        <th style="padding: 10px; text-align: right; color: #8b949e; border-bottom: 1px solid #30363d; font-weight: 500;">Portfolio</th>
                        <th style="padding: 10px; text-align: right; color: #8b949e; border-bottom: 1px solid #30363d; font-weight: 500;">S&P 500</th>
                        <th style="padding: 10px; text-align: right; color: #8b949e; border-bottom: 1px solid #30363d; font-weight: 500;">Drift</th>
                    </tr>
                </thead>
                <tbody>
                    {holdings_html}
                </tbody>
            </table>
        </div>

        <!-- Two Column: Sectors + Recent Trades -->
        <table width="100%" cellpadding="0" cellspacing="0" style="margin-bottom: 20px;">
            <tr valign="top">
                <td style="width: 50%; padding-right: 10px;">
                    <div style="background: #161b22; border-radius: 12px; padding: 20px; border: 1px solid #30363d; height: 100%;">
                        <h2 style="color: #f0f6fc; font-size: 14px; margin: 0 0 12px 0;">📊 Sector Allocation</h2>
                        <table width="100%" cellpadding="0" cellspacing="0" style="font-size: 12px;">
                            {sector_html}
                        </table>
                    </div>
                </td>
                <td style="width: 50%; padding-left: 10px;">
                    <div style="background: #161b22; border-radius: 12px; padding: 20px; border: 1px solid #30363d; height: 100%;">
                        <h2 style="color: #f0f6fc; font-size: 14px; margin: 0 0 12px 0;">📈 Recent Trades</h2>
                        <table width="100%" cellpadding="0" cellspacing="0" style="font-size: 12px;">
                            <thead>
                                <tr>
                                    <th style="padding: 6px; text-align: left; color: #8b949e; font-weight: 500;">Sym</th>
                                    <th style="padding: 6px; text-align: left; color: #8b949e; font-weight: 500;">Side</th>
                                    <th style="padding: 6px; text-align: right; color: #8b949e; font-weight: 500;">Qty</th>
                                    <th style="padding: 6px; text-align: right; color: #8b949e; font-weight: 500;">Price</th>
                                </tr>
                            </thead>
                            <tbody>
                                {trades_html}
                            </tbody>
                        </table>
                    </div>
                </td>
            </tr>
        </table>

        <!-- Footer -->
        <div style="text-align: center; padding: 20px; border-top: 1px solid #30363d;">
            <p style="color: #6e7681; font-size: 11px; margin: 0;">
                CFO Direct Index • Paper Trading • Day {days_active} of Forward Test<br>
                This is a simulated portfolio for testing purposes only.
            </p>
        </div>
    </div>
</body>
</html>
"""
    return html


def _generate_error_email(error_message: str) -> str:
    """Generate an error notification email."""
    return f"""
<!DOCTYPE html>
<html>
<head><meta charset="UTF-8"><title>Error</title></head>
<body style="margin: 0; padding: 20px; background: #0d1117; font-family: sans-serif;">
    <div style="max-width: 600px; margin: 0 auto; background: #161b22; border-radius: 12px; padding: 24px; border: 1px solid #f85149;">
        <h1 style="color: #f85149; font-size: 18px; margin: 0 0 16px 0;">Error in Daily Report</h1>
        <p style="color: #f0f6fc;">{error_message}</p>
    </div>
</body>
</html>
"""


def save_email_to_file(html: str, output_path: Path) -> None:
    """Save email HTML to file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)


if __name__ == "__main__":
    html = generate_daily_email("forward_10mm")
    output = Path("outputs/emails/daily_report.html")
    save_email_to_file(html, output)
    print(f"Email saved to: {output}")
