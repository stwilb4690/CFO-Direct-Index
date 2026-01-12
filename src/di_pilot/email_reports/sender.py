"""
Email sender for Direct Index daily reports.

Supports:
- SMTP sending (Gmail, Outlook, custom)
- Saving HTML to file for manual sending
"""

import smtplib
import ssl
from dataclasses import dataclass
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path
from typing import Optional
import os


@dataclass
class EmailConfig:
    """Email configuration for SMTP sending."""
    smtp_server: str = "smtp.gmail.com"
    smtp_port: int = 587
    username: str = ""
    password: str = ""  # App password for Gmail
    sender_email: str = ""
    recipient_emails: list = None
    
    def __post_init__(self):
        if self.recipient_emails is None:
            self.recipient_emails = []
    
    @classmethod
    def from_env(cls) -> "EmailConfig":
        """Load configuration from environment variables."""
        return cls(
            smtp_server=os.getenv("EMAIL_SMTP_SERVER", "smtp.gmail.com"),
            smtp_port=int(os.getenv("EMAIL_SMTP_PORT", "587")),
            username=os.getenv("EMAIL_USERNAME", ""),
            password=os.getenv("EMAIL_PASSWORD", ""),
            sender_email=os.getenv("EMAIL_SENDER", ""),
            recipient_emails=os.getenv("EMAIL_RECIPIENTS", "").split(","),
        )


def send_email(
    html_content: str,
    subject: str,
    config: EmailConfig,
    recipient_override: Optional[str] = None,
) -> bool:
    """
    Send HTML email via SMTP.

    Args:
        html_content: HTML email body
        subject: Email subject line
        config: Email configuration
        recipient_override: Override recipient (for testing)

    Returns:
        True if sent successfully, False otherwise
    """
    if not config.username or not config.password:
        print("Error: Email credentials not configured")
        print("Set EMAIL_USERNAME and EMAIL_PASSWORD environment variables")
        return False

    recipients = [recipient_override] if recipient_override else config.recipient_emails
    
    if not recipients:
        print("Error: No recipients configured")
        return False

    try:
        # Create message
        message = MIMEMultipart("alternative")
        message["Subject"] = subject
        message["From"] = config.sender_email or config.username
        message["To"] = ", ".join(recipients)

        # Attach HTML content
        html_part = MIMEText(html_content, "html")
        message.attach(html_part)

        # Send via SMTP
        context = ssl.create_default_context()
        
        with smtplib.SMTP(config.smtp_server, config.smtp_port) as server:
            server.starttls(context=context)
            server.login(config.username, config.password)
            server.sendmail(
                config.sender_email or config.username,
                recipients,
                message.as_string()
            )

        print(f"Email sent successfully to: {', '.join(recipients)}")
        return True

    except smtplib.SMTPAuthenticationError:
        print("Error: SMTP authentication failed. Check credentials.")
        return False
    except smtplib.SMTPException as e:
        print(f"Error sending email: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False


def save_email_html(html_content: str, output_path: Path) -> None:
    """
    Save email HTML to file for preview or manual sending.

    Args:
        html_content: HTML email content
        output_path: Path to save the HTML file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"Email HTML saved to: {output_path}")


if __name__ == "__main__":
    # Test configuration loading
    config = EmailConfig.from_env()
    print(f"SMTP Server: {config.smtp_server}:{config.smtp_port}")
    print(f"Username: {config.username or '(not set)'}")
    print(f"Recipients: {config.recipient_emails or '(not set)'}")
