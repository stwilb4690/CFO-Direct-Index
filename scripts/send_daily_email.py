"""
Automated email scheduling for CFO Direct Index.

This script can:
1. Run immediately and send an email
2. Be scheduled via Windows Task Scheduler
3. Be run as a daily cron job on Linux

Email configuration is loaded from environment variables or .env file.

Required environment variables:
    EMAIL_SMTP_SERVER: SMTP server (default: smtp.gmail.com)
    EMAIL_SMTP_PORT: SMTP port (default: 587)
    EMAIL_USERNAME: Your email username
    EMAIL_PASSWORD: Your email password or app password
    EMAIL_SENDER: Sender email address
    EMAIL_RECIPIENTS: Comma-separated list of recipient emails

Usage:
    python scripts/send_daily_email.py
    
For Gmail, you'll need an App Password (not your regular password):
https://support.google.com/accounts/answer/185833

To schedule on Windows (run at 6 PM daily):
    schtasks /create /tn "DirectIndexEmail" /tr "python C:\\path\\to\\send_daily_email.py" /sc daily /st 18:00
"""

import sys
import os
from pathlib import Path
from datetime import date

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Load environment variables from .env if it exists
env_file = Path(__file__).parent.parent / ".env"
if env_file.exists():
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ[key.strip()] = value.strip()

from di_pilot.email_reports.generator import generate_daily_email, save_email_to_file
from di_pilot.email_reports.sender import send_email, EmailConfig


def main():
    """Generate and send the daily email report."""
    print("=" * 60)
    print("CFO Direct Index - Daily Email Sender")
    print(f"Date: {date.today()}")
    print("=" * 60)
    
    # Generate the email HTML
    print("\nGenerating email report...")
    html = generate_daily_email("forward_10mm")
    
    # Save to file for reference
    output_dir = Path(__file__).parent.parent / "outputs" / "emails"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    filename = f"daily_report_{date.today().isoformat()}.html"
    save_email_to_file(html, output_dir / filename)
    save_email_to_file(html, output_dir / "latest.html")
    print(f"Email saved to: {output_dir / filename}")
    
    # Load email config from environment
    config = EmailConfig.from_env()
    
    # Check if email is configured
    if not config.username or not config.password:
        print("\n" + "=" * 60)
        print("EMAIL NOT CONFIGURED")
        print("=" * 60)
        print("To enable auto-sending, set these environment variables:")
        print("  EMAIL_USERNAME=your-email@gmail.com")
        print("  EMAIL_PASSWORD=your-app-password")
        print("  EMAIL_RECIPIENTS=recipient1@email.com,recipient2@email.com")
        print()
        print("Or create a .env file in the project root with these values.")
        print(f"\nEmail HTML saved to: {output_dir / filename}")
        return 1
    
    # Send the email
    print("\nSending email...")
    subject = f"CFO Direct Index - Daily Update ({date.today().strftime('%b %d, %Y')})"
    
    success = send_email(html, subject, config)
    
    if success:
        print("\n✓ Email sent successfully!")
        return 0
    else:
        print("\n✗ Failed to send email (see error above)")
        print(f"  Email saved to: {output_dir / filename}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
