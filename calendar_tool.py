"""
calendar_tool.py
----------------
Multi-tenant Google Calendar tools for SamaySetu AI.

All Google API calls are now routed through services/calendar_provider.py,
which loads per-tenant credentials from the DB at call time.

The old module-level globals (SERVICE_ACCOUNT_FILE, CALENDAR_ID, `service`)
have been removed entirely. Every function now resolves the tenant from the
active WebSocket session context and fetches the correct calendar client.
"""

import datetime
import os
import pytz
import urllib.parse
from typing import Optional
from dotenv import load_dotenv
import config

load_dotenv()

# ── Per-tenant calendar client ────────────────────────────────────────────────
from services.calendar_provider import (
    get_calendar_service,
    CalendarNotConnectedError,
    InvalidCalendarCredentialsError,
)

# ── Tenant Context (injected by main.py before every tool call) ───────────────
# main.py sets `calendar_tool.tenant_context` before calling any tool so every
# function knows which tenant (and which phone number) is active — without
# needing those values as explicit tool arguments.
tenant_context = None


def get_session_context():
    """Return (tenant_id, phone_number) from the active WebSocket session.
    Returns (None, None) if context has not been injected yet."""
    if tenant_context is None:
        return None, None
    session = tenant_context.chat_sessions.get(tenant_context.session_id)
    if not session:
        return None, None
    return session.get("tenant_id"), session.get("phone_number")


def get_tenant_config():
    """Return bot_config from the active WebSocket session."""
    if tenant_context is None:
        return {}
    session = tenant_context.chat_sessions.get(tenant_context.session_id)
    if not session:
        return {}
    return session.get("bot_config", {})


def _get_service_and_calendar(tenant_id: str):
    """
    Thin wrapper so every tool function gets (service, calendar_id) in one line
    and surfaces a clear error when the calendar is not connected.
    """
    if not tenant_id:
        raise CalendarNotConnectedError(
            "No tenant context available — cannot identify which calendar to use."
        )
    return get_calendar_service(tenant_id)


# ── DB integration (optional — degrades gracefully if psycopg2 missing) ───────
try:
    from database.crud import (
        create_appointment,
        update_appointment_status,
        update_status_by_event_id,
        update_rescheduled_appointment,
        get_calendar_event_id,
        user_owns_appointment,
    )
    _DB_AVAILABLE = True
except ImportError:
    _DB_AVAILABLE = False


def _try_db(fn, *args, **kwargs):
    """Helper to call a DB function, catching all errors so calendar ops never crash."""
    if not _DB_AVAILABLE:
        return None
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        print(f"[DB] Non-fatal DB error in calendar_tool: {e}")
        return None


IST = pytz.timezone(config.CALENDAR_TIMEZONE)


def generate_google_calendar_link(
    summary: str,
    description: str,
    location: str,
    start_dt: datetime.datetime,
    end_dt: datetime.datetime,
) -> str:
    """
    Generates a Google Calendar "TEMPLATE" link from datetime objects.

    Note: Google expects UTC timestamps in YYYYMMDDTHHMMSSZ format.
    """
    if start_dt.tzinfo is None:
        start_dt = IST.localize(start_dt)
    if end_dt.tzinfo is None:
        end_dt = IST.localize(end_dt)

    start_utc = start_dt.astimezone(datetime.timezone.utc)
    end_utc = end_dt.astimezone(datetime.timezone.utc)

    start_str = start_utc.strftime("%Y%m%dT%H%M%SZ")
    end_str = end_utc.strftime("%Y%m%dT%H%M%SZ")

    base_url = "https://calendar.google.com/calendar/render?action=TEMPLATE"
    params = {
        "text": summary or "",
        "details": description or "",
        "location": location or "",
        "dates": f"{start_str}/{end_str}",
    }
    return f"{base_url}&{urllib.parse.urlencode(params)}"


def is_past_time(start_time_str: str) -> bool:
    naive_dt = datetime.datetime.fromisoformat(start_time_str)
    start_dt = IST.localize(naive_dt)
    now_dt = datetime.datetime.now(IST)
    return start_dt <= now_dt


def _ceil_dt_to_slot(dt: datetime.datetime, slot_minutes: int) -> datetime.datetime:
    """
    Round a datetime up to the next slot boundary.
    Example: 14:36 with 30-min slots -> 15:00
    """
    slot = max(1, int(slot_minutes or config.DEFAULT_APPOINTMENT_DURATION))
    base = dt.replace(second=0, microsecond=0)
    mins = base.hour * 60 + base.minute
    rounded = ((mins + slot - 1) // slot) * slot

    day_offset, mins_in_day = divmod(rounded, 24 * 60)
    hour, minute = divmod(mins_in_day, 60)
    aligned = base.replace(hour=hour, minute=minute)
    if day_offset:
        aligned += datetime.timedelta(days=day_offset)
    return aligned


def _time_to_minutes(t: str) -> Optional[int]:
    try:
        hh, mm = t.split(":")
        hh_i, mm_i = int(hh), int(mm)
    except Exception:
        return None
    if hh_i == 24 and mm_i == 0:
        return 24 * 60
    if 0 <= hh_i <= 23 and 0 <= mm_i <= 59:
        return hh_i * 60 + mm_i
    return None


def _normalize_business_periods(bot_cfg: dict):
    """
    Return availability periods as minute-ranges: [(start_min, end_min), ...]
    Supports:
      - new format: business_hours_periods=[{"start":"09:00","end":"13:00"}, ...]
      - legacy format: business_hours_start/business_hours_end
    """
    periods = []
    raw = bot_cfg.get("business_hours_periods") or []
    if isinstance(raw, list):
        for p in raw:
            if not isinstance(p, dict):
                continue
            start = _time_to_minutes(str(p.get("start", "")).strip())
            end = _time_to_minutes(str(p.get("end", "")).strip())
            if start is None or end is None or start >= end:
                continue
            periods.append((start, end))

    if not periods:
        start_hour = int(bot_cfg.get("business_hours_start", config.BUSINESS_START_HOUR))
        end_hour = int(bot_cfg.get("business_hours_end", config.BUSINESS_END_HOUR))
        if start_hour < end_hour:
            periods = [(start_hour * 60, end_hour * 60)]

    return sorted(periods, key=lambda x: x[0])


def _format_periods_for_msg(periods):
    if not periods:
        return "9:00 to 18:00"

    def fmt(total_mins: int) -> str:
        hh = total_mins // 60
        mm = total_mins % 60
        return f"{hh:02d}:{mm:02d}"

    return ", ".join([f"{fmt(s)} to {fmt(e)}" for s, e in periods])


def is_within_business_hours(start_time_str: str, duration_minutes: int = 0) -> bool:
    bot_cfg = get_tenant_config()
    periods = _normalize_business_periods(bot_cfg)
    if not periods:
        return False

    naive_dt = datetime.datetime.fromisoformat(start_time_str)
    start_mins = naive_dt.hour * 60 + naive_dt.minute
    end_mins = start_mins + max(0, int(duration_minutes or 0))

    # Slot must fit entirely inside one selected business period.
    for p_start, p_end in periods:
        if start_mins >= p_start and end_mins <= p_end:
            return True
    return False


# ── Tool: check_calendar_availability ────────────────────────────────────────

def check_calendar_availability(
    start_time_str: str,
    duration_minutes: Optional[int] = None,
    phone_number: Optional[str] = None,
):
    """Checks if a time slot is free in this tenant's Google Calendar."""
    tenant_id, _ = get_session_context()
    bot_cfg = get_tenant_config()
    if duration_minutes is None:
        duration_minutes = bot_cfg.get("slot_duration_mins", config.DEFAULT_APPOINTMENT_DURATION)

    if is_past_time(start_time_str):
        return "Error: Cannot book an appointment in the past."

    if not is_within_business_hours(start_time_str, duration_minutes=duration_minutes):
        periods = _normalize_business_periods(bot_cfg)
        return (
            f"Error: We only accept appointments during these business hours: "
            f"{_format_periods_for_msg(periods)}. "
            "Please choose a different time."
        )

    try:
        service, calendar_id = _get_service_and_calendar(tenant_id)
    except (CalendarNotConnectedError, InvalidCalendarCredentialsError) as e:
        return f"Error: Calendar not available — {e}"

    naive_dt = datetime.datetime.fromisoformat(start_time_str)
    start_dt = IST.localize(naive_dt)
    end_dt = start_dt + datetime.timedelta(minutes=duration_minutes)

    body = {
        "timeMin": start_dt.isoformat(),
        "timeMax": end_dt.isoformat(),
        "items": [{"id": calendar_id}],
    }

    query = service.freebusy().query(body=body).execute()
    busy_slots = query["calendars"][calendar_id]["busy"]

    if not busy_slots:
        return f"Slot {start_time_str} is FREE."
    else:
        return f"Slot {start_time_str} is BUSY."


# ── Tool: suggest_next_available_slot ────────────────────────────────────────

def suggest_next_available_slot(
    start_time_str: str,
    duration_minutes: Optional[int] = None,
    search_hours: int = 4,
    max_slots: int = 3,
    phone_number: Optional[str] = None,
):
    tenant_id, _ = get_session_context()
    bot_cfg = get_tenant_config()
    if duration_minutes is None:
        duration_minutes = bot_cfg.get("slot_duration_mins", config.DEFAULT_APPOINTMENT_DURATION)

    try:
        service, calendar_id = _get_service_and_calendar(tenant_id)
    except Exception as e:
        return f"Error: Calendar not available — {e}"

    # Parse time
    naive_dt = datetime.datetime.fromisoformat(start_time_str)
    start_dt = IST.localize(naive_dt)
    now_dt = datetime.datetime.now(IST)

    # Hard safety: never suggest slots in the past.
    # If LLM passes a past start, clamp to next valid slot from "now".
    search_start = start_dt
    if search_start < now_dt:
        search_start = _ceil_dt_to_slot(now_dt, duration_minutes)

    end_search = search_start + datetime.timedelta(hours=search_hours)

    # 🔥 SINGLE API CALL
    body = {
        "timeMin": search_start.isoformat(),
        "timeMax": end_search.isoformat(),
        "items": [{"id": calendar_id}],
    }

    query = service.freebusy().query(body=body).execute()
    busy_slots = query["calendars"][calendar_id]["busy"]

    # Convert busy slots to datetime
    busy_intervals = []
    for slot in busy_slots:
        busy_start = datetime.datetime.fromisoformat(slot["start"])
        busy_end = datetime.datetime.fromisoformat(slot["end"])
        busy_intervals.append((busy_start, busy_end))

    # Sort intervals (important)
    busy_intervals.sort()

    # 🧠 FIND FREE SLOTS LOCALLY
    available_slots = []
    current = search_start

    while current < end_search and len(available_slots) < max_slots:
        end_time = current + datetime.timedelta(minutes=duration_minutes)

        # Check overlap with any busy slot
        is_busy = False
        for busy_start, busy_end in busy_intervals:
            if not (end_time <= busy_start or current >= busy_end):
                is_busy = True
                break

        if not is_busy and is_within_business_hours(
            current.strftime("%Y-%m-%dT%H:%M:%S"),
            duration_minutes=duration_minutes
        ):
            available_slots.append(current.strftime("%Y-%m-%dT%H:%M:%S"))

        current += datetime.timedelta(minutes=duration_minutes)

    if available_slots:
        return f"AVAILABLE_SLOTS: {available_slots}"

    return "No available slots found in the next few hours."

# ── Tool: book_appointment ────────────────────────────────────────────────────

def book_appointment(
    start_time_str: str,
    summary: str = "AI Appointment",
    duration_minutes: Optional[int] = None,
    phone_number: Optional[str] = None,
):
    """Books a Google Calendar event for this tenant and stores the appointment in the DB."""
    ctx_tenant_id, ctx_phone = get_session_context()
    bot_cfg = get_tenant_config()
    if duration_minutes is None:
        duration_minutes = bot_cfg.get("slot_duration_mins", config.DEFAULT_APPOINTMENT_DURATION)

    if not phone_number:
        phone_number = ctx_phone
    tenant_id = ctx_tenant_id

    if is_past_time(start_time_str):
        return "Error: Cannot book an appointment in the past."

    availability = check_calendar_availability(start_time_str, duration_minutes=duration_minutes)
    if "BUSY" in availability:
        return "Error: Slot already occupied."
    if "Error" in availability:
        return availability   # propagate calendar-not-connected errors cleanly

    try:
        service, calendar_id = _get_service_and_calendar(tenant_id)
    except (CalendarNotConnectedError, InvalidCalendarCredentialsError) as e:
        return f"Error: Calendar not available — {e}"

    display_summary = f"{summary} - {phone_number}" if phone_number else summary

    naive_dt = datetime.datetime.fromisoformat(start_time_str)
    end_dt = naive_dt + datetime.timedelta(minutes=duration_minutes)

    # Build calendar-link datetimes (timezone-aware, in tenant timezone)
    start_local = IST.localize(naive_dt) if naive_dt.tzinfo is None else naive_dt.astimezone(IST)
    end_local = IST.localize(end_dt) if end_dt.tzinfo is None else end_dt.astimezone(IST)

    event = {
        "summary": display_summary,
        "description": "Booked via SamaySetu Gujarati AI Bot",
        "start": {
            "dateTime": naive_dt.strftime("%Y-%m-%dT%H:%M:%S"),
            "timeZone": "Asia/Kolkata",
        },
        "end": {
            "dateTime": end_dt.strftime("%Y-%m-%dT%H:%M:%S"),
            "timeZone": "Asia/Kolkata",
        },
    }

    created_event = service.events().insert(calendarId=calendar_id, body=event).execute()
    calendar_event_id = created_event.get("id")

    if phone_number and calendar_event_id:
        _try_db(
            create_appointment,
            phone_number,
            naive_dt.strftime("%Y-%m-%dT%H:%M:%S"),
            end_dt.strftime("%Y-%m-%dT%H:%M:%S"),
            calendar_event_id,
            tenant_id,
        )
        print(f"[TENANT] book_appointment: tenant_id={tenant_id} phone={phone_number}")

    # Generate "Add to Google Calendar" link for the end-user (dynamic dates)
    calendar_link = generate_google_calendar_link(
        summary=display_summary,
        description="Booked via SamaySetu Gujarati AI Bot",
        location=bot_cfg.get("location", "Clinic / Shop Visit"),
        start_dt=start_local,
        end_dt=end_local,
    )

    return {
        "status": "SUCCESS",
        "message": "Appointment booked successfully.",
        "calendar_link": calendar_link,
        "start_time": naive_dt.strftime("%Y-%m-%d %H:%M"),
        "end_time": end_dt.strftime("%Y-%m-%d %H:%M"),
    }


# ── Tool: cancel_appointment ──────────────────────────────────────────────────

def cancel_appointment(
    start_time_str: str,
    phone_number: Optional[str] = None,
):
    """
    Deletes a tenant-scoped appointment from Google Calendar and marks it CANCELLED in the DB.

    Ownership guard: the DB must confirm a BOOKED appointment exists for THIS
    user before any Google Calendar operation is attempted.
    """
    ctx_tenant_id, ctx_phone = get_session_context()
    if not phone_number:
        phone_number = ctx_phone
    tenant_id = ctx_tenant_id

    if is_past_time(start_time_str):
        return "Error: Cannot cancel a past appointment."

    # 1. Ownership check
    if phone_number:
        owns = _try_db(user_owns_appointment, phone_number, start_time_str, tenant_id)
        if not owns:
            return (
                f"Error: No appointment found for your account at {start_time_str}. "
                "Please check the time and try again."
            )

    # 2. Get Google Calendar event_id from DB
    event_id = None
    if phone_number:
        event_id = _try_db(get_calendar_event_id, phone_number, start_time_str, tenant_id)
    if not event_id:
        return f"Error: No calendar event found at {start_time_str} to cancel."

    # 3. Delete from Google Calendar
    try:
        service, calendar_id = _get_service_and_calendar(tenant_id)
    except (CalendarNotConnectedError, InvalidCalendarCredentialsError) as e:
        return f"Error: Calendar not available — {e}"

    service.events().delete(calendarId=calendar_id, eventId=event_id).execute()

    # 4. Mark CANCELLED in DB
    _try_db(update_status_by_event_id, event_id, "CANCELLED")
    print(f"[TENANT] cancel_appointment: tenant_id={tenant_id} phone={phone_number}")

    return f"SUCCESS: Appointment at {start_time_str} has been cancelled."


# ── Tool: reschedule_appointment ──────────────────────────────────────────────

def reschedule_appointment(
    old_start_time_str: str,
    new_start_time_str: str,
    duration_minutes: Optional[int] = None,
    phone_number: Optional[str] = None,
):
    """
    Moves a tenant-scoped appointment from an old time slot to a new one.

    Ownership guard: same as cancel_appointment.
    """
    ctx_tenant_id, ctx_phone = get_session_context()
    bot_cfg = get_tenant_config()
    if duration_minutes is None:
        duration_minutes = bot_cfg.get("slot_duration_mins", config.DEFAULT_APPOINTMENT_DURATION)

    if not phone_number:
        phone_number = ctx_phone
    tenant_id = ctx_tenant_id

    if is_past_time(new_start_time_str):
        return "Error: Cannot reschedule to a past time."

    # 1. Ownership check
    if phone_number:
        owns = _try_db(user_owns_appointment, phone_number, old_start_time_str, tenant_id)
        if not owns:
            return (
                f"Error: No appointment found for your account at {old_start_time_str}. "
                "Please check the time and try again."
            )

    # 2. Get event_id from DB
    event_id = None
    if phone_number:
        event_id = _try_db(get_calendar_event_id, phone_number, old_start_time_str, tenant_id)
    if not event_id:
        return f"Error: No appointment found at {old_start_time_str}."

    # 3. Check new slot is free
    availability = check_calendar_availability(new_start_time_str, duration_minutes=duration_minutes)
    if "BUSY" in availability:
        return f"Error: The new slot {new_start_time_str} is already occupied."
    if "Error" in availability:
        return availability

    # 4. Update Google Calendar event
    try:
        service, calendar_id = _get_service_and_calendar(tenant_id)
    except (CalendarNotConnectedError, InvalidCalendarCredentialsError) as e:
        return f"Error: Calendar not available — {e}"

    event = service.events().get(calendarId=calendar_id, eventId=event_id).execute()

    new_naive_dt = datetime.datetime.fromisoformat(new_start_time_str)
    new_end_dt = new_naive_dt + datetime.timedelta(minutes=duration_minutes)

    event["start"]["dateTime"] = new_naive_dt.strftime("%Y-%m-%dT%H:%M:%S")
    event["end"]["dateTime"] = new_end_dt.strftime("%Y-%m-%dT%H:%M:%S")

    service.events().update(calendarId=calendar_id, eventId=event_id, body=event).execute()

    # 5. Update DB
    if phone_number:
        _try_db(
            update_rescheduled_appointment,
            phone_number,
            old_start_time_str,
            new_start_time_str,
            new_end_dt.strftime("%Y-%m-%dT%H:%M:%S"),
            tenant_id,
        )
        print(f"[TENANT] reschedule_appointment: tenant_id={tenant_id} phone={phone_number}")

    return f"SUCCESS: Appointment moved from {old_start_time_str} to {new_start_time_str}."
