"""
Daily runner script for Forward Test portfolio.

This script is designed to be run daily (via Task Scheduler or cron):
1. Runs the daily simulation for the forward test portfolio
2. Updates dashboard data
3. Generates and saves/sends the daily email report

Usage:
    python scripts/daily_runner.py

For automated scheduling on Windows:
    schtasks /create /tn "DirectIndexDailyRun" /tr "python path\\to\\daily_runner.py" /sc daily /st 18:00
"""

import sys
import logging
from pathlib import Path
from datetime import date, datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from di_pilot.simulation.forward import ForwardTestRunner
from di_pilot.data.providers.eodhd_provider import EODHDProvider
from di_pilot.config import load_api_keys


# Configure logging
log_dir = Path(__file__).parent.parent / "logs"
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_dir / "daily_runs.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def run_daily_update(portfolio_id: str = "forward_10mm"):
    """
    Run the daily portfolio update.
    
    Returns:
        Dict with run results
    """
    logger.info(f"Starting daily update for portfolio: {portfolio_id}")
    
    results = {
        "portfolio_id": portfolio_id,
        "run_date": date.today().isoformat(),
        "success": False,
        "error": None,
    }
    
    try:
        # Load API keys
        load_api_keys()
        
        # Initialize runner and provider
        runner = ForwardTestRunner()
        provider = EODHDProvider()
        
        if not runner.portfolio_exists(portfolio_id):
            raise ValueError(f"Portfolio {portfolio_id} does not exist")
        
        # Run the daily simulation
        logger.info("Executing daily simulation...")
        snapshot = runner.run_daily(
            portfolio_id=portfolio_id,
            provider=provider,
        )
        
        logger.info(f"Daily snapshot recorded:")
        logger.info(f"  Date: {snapshot.date}")
        logger.info(f"  Total Value: ${float(snapshot.total_value):,.2f}")
        logger.info(f"  Daily Return: {float(snapshot.daily_return)*100:.2f}%")
        
        results["snapshot"] = {
            "date": snapshot.date.isoformat(),
            "total_value": float(snapshot.total_value),
            "daily_return": float(snapshot.daily_return),
        }
        results["success"] = True
        
    except Exception as e:
        logger.error(f"Error in daily update: {e}")
        results["error"] = str(e)
        import traceback
        traceback.print_exc()
    
    return results


def update_dashboard_data(portfolio_id: str = "forward_10mm"):
    """Update the dashboard JSON data."""
    logger.info("Updating dashboard data...")
    
    try:
        # Import the generator
        from scripts.generate_dashboard_data import generate_dashboard_data
        import json
        
        data = generate_dashboard_data(portfolio_id)
        
        if data:
            output_path = Path(__file__).parent.parent / "dashboard" / "data" / "portfolio.json"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, "w") as f:
                json.dump(data, f, default=str, indent=2)
            
            logger.info(f"Dashboard data updated: {output_path}")
            return True
    except Exception as e:
        logger.error(f"Error updating dashboard: {e}")
    
    return False


def generate_and_save_email(portfolio_id: str = "forward_10mm"):
    """Generate the daily email and save to file."""
    logger.info("Generating daily email report...")
    
    try:
        from di_pilot.email_reports.generator import generate_daily_email, save_email_to_file
        
        html = generate_daily_email(portfolio_id)
        
        output_dir = Path(__file__).parent.parent / "outputs" / "emails"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"daily_report_{date.today().isoformat()}.html"
        output_path = output_dir / filename
        
        save_email_to_file(html, output_path)
        logger.info(f"Email saved: {output_path}")
        
        # Also save as latest
        latest_path = output_dir / "latest.html"
        save_email_to_file(html, latest_path)
        
        return output_path
    except Exception as e:
        logger.error(f"Error generating email: {e}")
        return None


def main():
    """Main daily runner entry point."""
    logger.info("=" * 60)
    logger.info(f"CFO Direct Index - Daily Runner")
    logger.info(f"Run Date: {datetime.now().isoformat()}")
    logger.info("=" * 60)
    
    portfolio_id = "forward_10mm"
    
    # Step 1: Run daily simulation
    results = run_daily_update(portfolio_id)
    
    if not results["success"]:
        logger.error(f"Daily update failed: {results.get('error')}")
        # Continue anyway to generate reports with current data
    
    # Step 2: Update dashboard data
    update_dashboard_data(portfolio_id)
    
    # Step 3: Generate email report
    email_path = generate_and_save_email(portfolio_id)
    
    # Summary
    logger.info("=" * 60)
    logger.info("Daily Run Complete")
    logger.info(f"  Portfolio: {portfolio_id}")
    logger.info(f"  Simulation: {'SUCCESS' if results['success'] else 'FAILED'}")
    if email_path:
        logger.info(f"  Email: {email_path}")
    logger.info("=" * 60)
    
    return 0 if results["success"] else 1


if __name__ == "__main__":
    sys.exit(main())
