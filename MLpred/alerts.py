# -*- coding: utf-8 -*-
"""
alerts.py

"""

import os
import traceback

import boto3
from botocore.exceptions import ClientError, BotoCoreError

_ENABLED = os.environ.get("ALERT_EMAIL_ENABLED", "1").strip().lower() not in ("0", "false", "no")
_FROM    = os.environ.get("noussair.lazrak@nyu.edu")
_TO      = [addr.strip() for addr in os.environ.get("noussair.lazrak@nyu.edu'", "").split(",") if addr.strip()]
_REGION  = os.environ.get("ALERT_SES_REGION", "us-east-1")

_ses_client = None
_warned_unconfigured = False


def _client():
    global _ses_client
    if _ses_client is None:
        _ses_client = boto3.client("ses", region_name=_REGION)
    return _ses_client


def send_alert(subject: str, body: str) -> bool:
    """
    Send an alert email via SES. Never raises — failures are logged and
    swallowed so a broken mailer can't interrupt the forecast pipeline.

    Returns True if the email was sent, False otherwise.
    """
    global _warned_unconfigured

    if not _ENABLED:
        return False

    if not _FROM or not _TO:
        if not _warned_unconfigured:
            print("WARNING: Email alerts not configured "
                  "(set ALERT_EMAIL_FROM and ALERT_EMAIL_TO) — skipping.")
            _warned_unconfigured = True
        return False

    try:
        _client().send_email(
            Source=_FROM,
            Destination={"ToAddresses": _TO},
            Message={
                "Subject": {"Data": subject, "Charset": "UTF-8"},
                "Body": {"Text": {"Data": body, "Charset": "UTF-8"}},
            },
        )
        print(f"INFO: Alert email sent — {subject}")
        return True
    except (ClientError, BotoCoreError) as exc:
        print(f"WARNING: Failed to send alert email ({subject}): {exc}")
        return False


def alert_read_failure(step: str, location: str, exc: Exception, **context) -> bool:
    """
    Alert that a data-read step (read_pandora / read_geos_cf / read_obs / …)
    raised an exception for a given location.
    """
    subject = f"[GEOS-CF Bias Correction] {step} failed for {location}"
    ctx_lines = "\n".join(f"  {k}: {v}" for k, v in context.items())
    body = (
        f"Step:     {step}\n"
        f"Location: {location}\n"
        f"{ctx_lines}\n\n"
        f"Error: {exc}\n\n"
        f"Traceback:\n{traceback.format_exc()}"
    )
    return send_alert(subject, body)


def alert_no_new_data(step: str, location: str, reason: str, **context) -> bool:
    """
    Alert that a data-read step completed without raising, but returned no
    usable / new data (e.g. empty observations, no GEOS-CF rows, forecast
    generation returned None).
    """
    subject = f"[GEOS-CF Bias Correction] No new data for {location} ({step})"
    ctx_lines = "\n".join(f"  {k}: {v}" for k, v in context.items())
    body = (
        f"Step:     {step}\n"
        f"Location: {location}\n"
        f"Reason:   {reason}\n"
        f"{ctx_lines}\n"
    )
    return send_alert(subject, body)
