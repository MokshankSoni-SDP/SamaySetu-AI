"""
services/calendar_provider.py
------------------------------
Per-tenant Google Calendar client factory for SamaySetu AI.

Replaces the old global service_account.json approach.
Every call to get_calendar_service() loads credentials from the DB
for the requesting tenant and builds an isolated Google API client.

Usage:
    from services.calendar_provider import get_calendar_service

    service, calendar_id = get_calendar_service(tenant_id)
    service.events().insert(calendarId=calendar_id, body=event).execute()
"""

import json
from functools import lru_cache
from google.oauth2 import service_account
from googleapiclient.discovery import build

# ── DB import (degrades gracefully if psycopg2 not installed) ─────────────────
try:
    from database.crud import get_calendar_token
    _DB_AVAILABLE = True
except ImportError:
    _DB_AVAILABLE = False


SCOPES = ['https://www.googleapis.com/auth/calendar']


class CalendarNotConnectedError(Exception):
    """Raised when a tenant has no calendar credentials stored in the DB."""
    pass


class InvalidCalendarCredentialsError(Exception):
    """Raised when stored credentials are malformed or rejected by Google."""
    pass


def get_calendar_service(tenant_id: str):
    """
    Load this tenant's Google Calendar credentials from the DB and return
    an authenticated (service, calendar_id) pair.

    Args:
        tenant_id: UUID string of the requesting tenant.

    Returns:
        (googleapiclient.discovery.Resource, str)
        A tuple of the Calendar API service object and the calendar_id string.

    Raises:
        CalendarNotConnectedError  — no credentials row found for this tenant.
        InvalidCalendarCredentialsError — credentials are present but invalid.
    """
    if not _DB_AVAILABLE:
        raise CalendarNotConnectedError(
            "Database module unavailable — cannot load calendar credentials."
        )

    token_row = get_calendar_token(tenant_id)
    if not token_row:
        raise CalendarNotConnectedError(
            f"No calendar connected for tenant {tenant_id}. "
            "Please connect a Google Calendar from the admin panel."
        )

    calendar_id = token_row.get("calendar_id")
    token_json_str = token_row.get("token_json")

    if not calendar_id or not token_json_str:
        raise CalendarNotConnectedError(
            f"Incomplete calendar record for tenant {tenant_id}."
        )

    try:
        creds_info = json.loads(token_json_str)
    except (json.JSONDecodeError, TypeError) as e:
        raise InvalidCalendarCredentialsError(
            f"Stored service account JSON is malformed for tenant {tenant_id}: {e}"
        )

    try:
        creds = service_account.Credentials.from_service_account_info(
            creds_info, scopes=SCOPES
        )
        service = build('calendar', 'v3', credentials=creds)
    except Exception as e:
        raise InvalidCalendarCredentialsError(
            f"Could not build Google Calendar client for tenant {tenant_id}: {e}"
        )

    return service, calendar_id


def verify_calendar_connection(tenant_id: str) -> dict:
    """
    Test whether the stored credentials actually work against the Google Calendar API.
    Used by the /admin/calendar/connect endpoint to give a real green-light signal.

    Returns:
        {"ok": True, "calendar_id": "..."} on success.
        {"ok": False, "error": "..."} on failure.
    """
    import datetime, pytz
    try:
        service, calendar_id = get_calendar_service(tenant_id)
        now = datetime.datetime.utcnow().isoformat() + "Z"
        one_hour_later = (
            datetime.datetime.utcnow() + datetime.timedelta(hours=1)
        ).isoformat() + "Z"

        service.freebusy().query(body={
            "timeMin": now,
            "timeMax": one_hour_later,
            "items": [{"id": calendar_id}]
        }).execute()

        return {"ok": True, "calendar_id": calendar_id}

    except CalendarNotConnectedError as e:
        return {"ok": False, "error": str(e)}
    except InvalidCalendarCredentialsError as e:
        return {"ok": False, "error": str(e)}
    except Exception as e:
        return {"ok": False, "error": f"Google Calendar API error: {e}"}