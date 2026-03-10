"""
database/crud.py
----------------
All database read/write operations for SamaySetu AI.
All functions use synchronous psycopg2 (called via asyncio.to_thread from async contexts).

Integration points:
  - book_appointment()      → create_appointment()
  - cancel_appointment()    → update_appointment_status()
  - reschedule_appointment()→ update_rescheduled_appointment()
  - POST /user/login        → create_user_if_not_exists()
  - GET /appointments/{ph}  → get_user_appointments()
"""

from datetime import datetime
from typing import Optional, List, Dict, Any

from database.db import get_db_connection


# ── User operations ─────────────────────────────────────────────────────────

def create_user_if_not_exists(phone_number: str, name: Optional[str] = None) -> Dict[str, Any]:
    """
    Inserts a new user row if the phone_number does not already exist.
    Uses ON CONFLICT DO NOTHING so repeated logins are safe.

    Returns the user row as a dict.
    """
    sql_insert = """
        INSERT INTO users (phone_number, name)
        VALUES (%s, %s)
        ON CONFLICT (phone_number) DO NOTHING;
    """
    sql_select = "SELECT * FROM users WHERE phone_number = %s;"

    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(sql_insert, (phone_number, name))
            conn.commit()
            cur.execute(sql_select, (phone_number,))
            row = cur.fetchone()
            return dict(row) if row else {}
    finally:
        conn.close()


# ── Appointment operations ───────────────────────────────────────────────────

def create_appointment(
    phone_number: str,
    start_time: str,          # ISO format string, e.g. "2025-03-15T10:00:00"
    end_time: str,
    calendar_event_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Inserts a new BOOKED appointment for the given phone_number.
    calendar_event_id is the Google Calendar event ID stored for fast retrieval.
    Returns the newly created appointment row.
    """
    sql = """
        INSERT INTO appointments (phone_number, start_time, end_time, calendar_event_id, status)
        VALUES (%s, %s, %s, %s, 'BOOKED')
        RETURNING *;
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (phone_number, start_time, end_time, calendar_event_id))
            row = cur.fetchone()
            conn.commit()
            return dict(row) if row else {}
    finally:
        conn.close()


def get_user_appointments(phone_number: str) -> List[Dict[str, Any]]:
    """
    Returns all appointments for a user ordered by most recent first.
    Converts datetime objects to ISO strings for JSON serialisation.
    """
    sql = """
        SELECT appointment_id, phone_number, start_time, end_time,
               calendar_event_id, status, created_at
        FROM appointments
        WHERE phone_number = %s
        ORDER BY start_time DESC;
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (phone_number,))
            rows = cur.fetchall()
            results = []
            for row in rows:
                r = dict(row)
                # Convert datetime → ISO string for JSON serialisation
                for key in ("start_time", "end_time", "created_at"):
                    if isinstance(r.get(key), datetime):
                        r[key] = r[key].isoformat()
                # appointment_id is a UUID — convert to str
                if r.get("appointment_id"):
                    r["appointment_id"] = str(r["appointment_id"])
                results.append(r)
            return results
    finally:
        conn.close()


def get_calendar_event_id(phone_number: str, start_time: str) -> Optional[str]:
    """
    Looks up the stored Google Calendar event ID for an appointment.
    Used by cancel/reschedule to avoid extra Calendar API calls.
    Returns None if not found (caller should fall back to find_event_id()).
    """
    sql = """
        SELECT calendar_event_id
        FROM appointments
        WHERE phone_number = %s
          AND start_time = %s
          AND status = 'BOOKED'
        LIMIT 1;
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (phone_number, start_time))
            row = cur.fetchone()
            return row["calendar_event_id"] if row else None
    finally:
        conn.close()


def update_appointment_status(
    phone_number: str,
    start_time: str,
    status: str,
) -> bool:
    """
    Updates the status of the most recent BOOKED appointment at start_time.
    status must be one of: 'BOOKED', 'CANCELLED', 'RESCHEDULED'.
    Returns True if a row was updated, False otherwise.
    """
    valid_statuses = {"BOOKED", "CANCELLED", "RESCHEDULED"}
    if status not in valid_statuses:
        raise ValueError(f"Invalid status '{status}'. Must be one of {valid_statuses}")

    sql = """
        UPDATE appointments
        SET status = %s
        WHERE phone_number = %s
          AND start_time = %s
          AND status = 'BOOKED';
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (status, phone_number, start_time))
            updated = cur.rowcount > 0
            conn.commit()
            return updated
    finally:
        conn.close()


def update_rescheduled_appointment(
    phone_number: str,
    old_start_time: str,
    new_start_time: str,
    new_end_time: Optional[str] = None,
) -> bool:
    """
    Updates an existing BOOKED appointment's time and marks it RESCHEDULED.
    Returns True if a row was updated, False if the original appointment wasn't found.
    """
    sql = """
        UPDATE appointments
        SET start_time = %s,
            end_time   = COALESCE(%s, end_time + (%s::timestamp - start_time)),
            status     = 'RESCHEDULED'
        WHERE phone_number = %s
          AND start_time   = %s
          AND status       = 'BOOKED';
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (new_start_time, new_end_time, new_start_time, phone_number, old_start_time))
            updated = cur.rowcount > 0
            conn.commit()
            return updated
    finally:
        conn.close()


def delete_user_appointments(phone_number: str) -> int:
    """
    Deletes all appointment records for a user (admin/cleanup utility).
    Returns the number of rows deleted.
    """
    sql = "DELETE FROM appointments WHERE phone_number = %s;"
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (phone_number,))
            deleted = cur.rowcount
            conn.commit()
            return deleted
    finally:
        conn.close()
