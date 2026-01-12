"""
Daily email report generator for Direct Index Forward Test.

Generates professional HTML emails with:
- Portfolio overview (value, return, cash)
- Daily changes summary
- Recent trades
- TLH opportunities
- Market context
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
    Generate daily email HTML report.

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

    # Calculate metrics
    holdings = {}
    for lot in state.lots:
        if lot.symbol not in holdings:
            holdings[lot.symbol] = Decimal("0")
        holdings[lot.symbol] += lot.shares * lot.cost_basis

    total_holdings = sum(holdings.values())
    portfolio_value = float(total_holdings + state.cash)
    initial_value = float(runner.config.initial_cash)
    total_return = (portfolio_value - initial_value) / initial_value

    # Get recent trades (today's or most recent)
    recent_trades = [t for t in state.trades if t.timestamp.date() >= report_date]
    if not recent_trades:
        recent_trades = state.trades[-5:] if state.trades else []

    # Count trade types
    tlh_trades = [t for t in recent_trades if "TLH" in t.reason.value]
    rebalance_trades = [t for t in recent_trades if "REBALANCE" in t.reason.value]

    # Build the email HTML
    html = _generate_email_html(
        report_date=report_date,
        portfolio_id=portfolio_id,
        portfolio_value=portfolio_value,
        cash_balance=float(state.cash),
        positions_count=len(holdings),
        lots_count=len(state.lots),
        total_return=total_return,
        harvested_losses=float(state.harvested_losses),
        recent_trades=recent_trades,
        tlh_count=len(tlh_trades),
        rebalance_count=len(rebalance_trades),
    )

    return html


def _generate_email_html(
    report_date: date,
    portfolio_id: str,
    portfolio_value: float,
    cash_balance: float,
    positions_count: int,
    lots_count: int,
    total_return: float,
    harvested_losses: float,
    recent_trades: list,
    tlh_count: int,
    rebalance_count: int,
) -> str:
    """Generate the HTML email content."""

    # Format helpers
    def fmt_money(val):
        return f"${val:,.2f}"

    def fmt_pct(val):
        return f"{val*100:+.2f}%"

    # Determine return color
    return_color = "#3fb950" if total_return >= 0 else "#f85149"
    return_arrow = "&#9650;" if total_return >= 0 else "&#9660;"

    # Generate trades table rows
    trades_html = ""
    if recent_trades:
        for trade in recent_trades[:10]:
            side_color = "#3fb950" if trade.side.value == "BUY" else "#f85149"
            trades_html += f"""
            <tr>
                <td style="padding: 8px; border-bottom: 1px solid #30363d;">{trade.symbol}</td>
                <td style="padding: 8px; border-bottom: 1px solid #30363d; color: {side_color};">{trade.side.value}</td>
                <td style="padding: 8px; border-bottom: 1px solid #30363d; text-align: right;">{float(trade.shares):.2f}</td>
                <td style="padding: 8px; border-bottom: 1px solid #30363d; text-align: right;">{fmt_money(float(trade.price))}</td>
                <td style="padding: 8px; border-bottom: 1px solid #30363d;">{trade.reason.value}</td>
            </tr>
            """
    else:
        trades_html = '<tr><td colspan="5" style="padding: 16px; text-align: center; color: #8b949e;">No trades today</td></tr>'

    # What changed section
    changes_html = ""
    if tlh_count > 0:
        changes_html += f'<li style="margin-bottom: 8px;">Tax-Loss Harvesting: {tlh_count} trades executed</li>'
    if rebalance_count > 0:
        changes_html += f'<li style="margin-bottom: 8px;">Rebalancing: {rebalance_count} trades executed</li>'
    if not changes_html:
        changes_html = '<li style="color: #8b949e;">No trading activity today</li>'

    html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Direct Index Daily Update - {report_date}</title>
</head>
<body style="margin: 0; padding: 0; background-color: #0d1117; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
    <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
        <!-- Header -->
        <div style="background: linear-gradient(135deg, #161b22 0%, #0d1117 100%); border-radius: 12px; padding: 24px; margin-bottom: 20px; border: 1px solid #30363d;">
            <h1 style="color: #58a6ff; font-size: 20px; margin: 0 0 8px 0;">CFO Direct Index</h1>
            <p style="color: #8b949e; font-size: 14px; margin: 0;">Daily Update - {report_date.strftime("%B %d, %Y")}</p>
        </div>

        <!-- Portfolio Summary -->
        <div style="background: #161b22; border-radius: 12px; padding: 24px; margin-bottom: 20px; border: 1px solid #30363d;">
            <h2 style="color: #f0f6fc; font-size: 16px; margin: 0 0 16px 0;">Portfolio Summary</h2>
            
            <div style="display: flex; justify-content: space-between; margin-bottom: 16px;">
                <div>
                    <p style="color: #8b949e; font-size: 12px; margin: 0;">Total Value</p>
                    <p style="color: #f0f6fc; font-size: 28px; font-weight: 700; margin: 4px 0;">{fmt_money(portfolio_value)}</p>
                </div>
                <div style="text-align: right;">
                    <p style="color: #8b949e; font-size: 12px; margin: 0;">Total Return</p>
                    <p style="color: {return_color}; font-size: 28px; font-weight: 700; margin: 4px 0;">{return_arrow} {fmt_pct(total_return)}</p>
                </div>
            </div>

            <table style="width: 100%; border-collapse: collapse;">
                <tr>
                    <td style="padding: 8px 0; border-top: 1px solid #30363d;">
                        <span style="color: #8b949e;">Cash Balance</span>
                    </td>
                    <td style="padding: 8px 0; border-top: 1px solid #30363d; text-align: right;">
                        <span style="color: #f0f6fc;">{fmt_money(cash_balance)}</span>
                    </td>
                </tr>
                <tr>
                    <td style="padding: 8px 0; border-top: 1px solid #30363d;">
                        <span style="color: #8b949e;">Positions</span>
                    </td>
                    <td style="padding: 8px 0; border-top: 1px solid #30363d; text-align: right;">
                        <span style="color: #f0f6fc;">{positions_count} ({lots_count} lots)</span>
                    </td>
                </tr>
                <tr>
                    <td style="padding: 8px 0; border-top: 1px solid #30363d;">
                        <span style="color: #8b949e;">Tax Losses Harvested</span>
                    </td>
                    <td style="padding: 8px 0; border-top: 1px solid #30363d; text-align: right;">
                        <span style="color: #3fb950;">{fmt_money(harvested_losses)}</span>
                    </td>
                </tr>
            </table>
        </div>

        <!-- What Changed Today -->
        <div style="background: #161b22; border-radius: 12px; padding: 24px; margin-bottom: 20px; border: 1px solid #30363d;">
            <h2 style="color: #f0f6fc; font-size: 16px; margin: 0 0 16px 0;">What Changed Today</h2>
            <ul style="color: #f0f6fc; padding-left: 20px; margin: 0;">
                {changes_html}
            </ul>
        </div>

        <!-- Recent Trades -->
        <div style="background: #161b22; border-radius: 12px; padding: 24px; margin-bottom: 20px; border: 1px solid #30363d;">
            <h2 style="color: #f0f6fc; font-size: 16px; margin: 0 0 16px 0;">Recent Trades</h2>
            <table style="width: 100%; border-collapse: collapse; font-size: 13px;">
                <thead>
                    <tr>
                        <th style="padding: 8px; text-align: left; color: #8b949e; border-bottom: 1px solid #30363d;">Symbol</th>
                        <th style="padding: 8px; text-align: left; color: #8b949e; border-bottom: 1px solid #30363d;">Side</th>
                        <th style="padding: 8px; text-align: right; color: #8b949e; border-bottom: 1px solid #30363d;">Shares</th>
                        <th style="padding: 8px; text-align: right; color: #8b949e; border-bottom: 1px solid #30363d;">Price</th>
                        <th style="padding: 8px; text-align: left; color: #8b949e; border-bottom: 1px solid #30363d;">Reason</th>
                    </tr>
                </thead>
                <tbody style="color: #f0f6fc;">
                    {trades_html}
                </tbody>
            </table>
        </div>

        <!-- Footer -->
        <div style="text-align: center; padding: 20px;">
            <p style="color: #6e7681; font-size: 12px; margin: 0;">
                CFO Direct Index - Paper Trading Simulation<br>
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
<head>
    <meta charset="UTF-8">
    <title>Direct Index Error</title>
</head>
<body style="margin: 0; padding: 20px; background-color: #0d1117; font-family: sans-serif;">
    <div style="max-width: 600px; margin: 0 auto; background: #161b22; border-radius: 12px; padding: 24px; border: 1px solid #f85149;">
        <h1 style="color: #f85149; font-size: 18px; margin: 0 0 16px 0;">Error in Daily Report</h1>
        <p style="color: #f0f6fc;">{error_message}</p>
    </div>
</body>
</html>
"""


def save_email_to_file(html: str, output_path: Path) -> None:
    """Save email HTML to file for preview or manual sending."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)


if __name__ == "__main__":
    # Test generation
    html = generate_daily_email("forward_10mm")
    output = Path("outputs/emails/daily_report.html")
    save_email_to_file(html, output)
    print(f"Email saved to: {output}")
