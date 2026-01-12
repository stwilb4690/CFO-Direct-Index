"""
Email Reports Module for Direct Index Forward Test.

Generates daily HTML email reports with portfolio updates,
trade activity, and TLH opportunities.
"""

from di_pilot.email_reports.generator import generate_daily_email
from di_pilot.email_reports.sender import send_email, EmailConfig

__all__ = ["generate_daily_email", "send_email", "EmailConfig"]
