"""
database/crud.py  (multi-tenant edition + modules + knowledge_base)
-----------------------------------------
All DB read/write operations for SamaySetu AI.
Every customer-facing operation requires a tenant_id so data is
always isolated between different business owners.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from database.db import get_db_connection

# ── Known module names ────────────────────────────────────────────────────────
BOOKING_MODULE = "BOOKING_MODULE"
FACTS_MODULE   = "FACTS_MODULE"
ALL_MODULES    = [BOOKING_MODULE, FACTS_MODULE]


def _to_dt(s):
    if isinstance(s, datetime):
        return s
    return datetime.fromisoformat(s)


# ══════════════════════════════════════════════════════════
#  TENANT operations  (SamaySetu platform-level)
# ══════════════════════════════════════════════════════════

def create_tenant(business_name: str, business_type: str, owner_email: str) -> Dict[str, Any]:
    sql = """
        INSERT INTO tenants (business_name, business_type, owner_email)
        VALUES (%s, %s, %s)
        RETURNING *;
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (business_name, business_type, owner_email))
            row = cur.fetchone()
            conn.commit()
            tenant = _serialize(row)
        # Seed default modules: BOOKING enabled, FACTS disabled
        _seed_default_modules(tenant["tenant_id"])
        return tenant
    finally:
        conn.close()


def _seed_default_modules(tenant_id: str):
    """Insert default module rows for a new tenant (BOOKING on, FACTS off)."""
    defaults = {BOOKING_MODULE: True, FACTS_MODULE: False}
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            for module_name, enabled in defaults.items():
                cur.execute(
                    """
                    INSERT INTO module_configs (tenant_id, module_name, is_enabled)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (tenant_id, module_name) DO NOTHING;
                    """,
                    (tenant_id, module_name, enabled),
                )
        conn.commit()
    finally:
        conn.close()


def get_all_tenants() -> List[Dict[str, Any]]:
    sql = """
        SELECT t.*, bc.bot_name, bc.language_code,
               bc.business_hours_start, bc.business_hours_end
        FROM tenants t
        LEFT JOIN bot_configs bc ON bc.tenant_id = t.tenant_id
        ORDER BY t.created_at DESC;
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
            return [_serialize(r) for r in cur.fetchall()]
    finally:
        conn.close()


def get_tenant_by_id(tenant_id: str) -> Optional[Dict[str, Any]]:
    sql = "SELECT * FROM tenants WHERE tenant_id = %s;"
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (tenant_id,))
            row = cur.fetchone()
            return _serialize(row) if row else None
    finally:
        conn.close()


def update_tenant_status(tenant_id: str, is_active: bool) -> bool:
    sql = "UPDATE tenants SET is_active = %s WHERE tenant_id = %s;"
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (is_active, tenant_id))
            updated = cur.rowcount > 0
            conn.commit()
            return updated
    finally:
        conn.close()


def get_platform_stats() -> Dict[str, Any]:
    """High-level stats for the SamaySetu super-admin dashboard."""
    sql = """
        SELECT
            (SELECT COUNT(*) FROM tenants)                          AS total_tenants,
            (SELECT COUNT(*) FROM tenants WHERE is_active = TRUE)   AS active_tenants,
            (SELECT COUNT(*) FROM users)                            AS total_users,
            (SELECT COUNT(*) FROM appointments)                     AS total_appointments,
            (SELECT COUNT(*) FROM appointments WHERE status='BOOKED'
             AND start_time >= NOW())                               AS upcoming_appointments,
            (SELECT COUNT(*) FROM appointments
             WHERE created_at >= NOW() - INTERVAL '24 hours')       AS appointments_today;
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
            return _serialize(cur.fetchone())
    finally:
        conn.close()


# ══════════════════════════════════════════════════════════
#  ADMIN (tenant admin) operations
# ══════════════════════════════════════════════════════════

def create_tenant_admin(tenant_id: str, email: str, password_hash: str, role: str = "admin") -> Dict[str, Any]:
    sql = """
        INSERT INTO tenant_admins (tenant_id, email, password_hash, role)
        VALUES (%s, %s, %s, %s)
        RETURNING *;
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (tenant_id, email, password_hash, role))
            row = cur.fetchone()
            conn.commit()
            return _serialize(row)
    finally:
        conn.close()


def get_admin_by_email(email: str) -> Optional[Dict[str, Any]]:
    sql = """
        SELECT a.*, t.business_name, t.business_type, t.is_active AS tenant_active
        FROM tenant_admins a
        JOIN tenants t ON t.tenant_id = a.tenant_id
        WHERE a.email = %s;
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (email,))
            row = cur.fetchone()
            return _serialize(row) if row else None
    finally:
        conn.close()


# ══════════════════════════════════════════════════════════
#  BOT CONFIG operations
# ══════════════════════════════════════════════════════════

def get_bot_config(tenant_id: str) -> Optional[Dict[str, Any]]:
    sql = "SELECT * FROM bot_configs WHERE tenant_id = %s;"
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (tenant_id,))
            row = cur.fetchone()
            return _serialize(row) if row else None
    finally:
        conn.close()


def upsert_bot_config(tenant_id: str, **fields) -> Dict[str, Any]:
    """Insert or update bot configuration for a tenant."""
    allowed = {
        "bot_name", "receptionist_name", "language_code", "tts_speaker",
        "business_hours_start", "business_hours_end", "slot_duration_mins",
        "silence_timeout_ms", "greeting_message", "business_description",
        "extra_prompt_context", "calendar_id",
    }
    filtered = {k: v for k, v in fields.items() if k in allowed}
    if not filtered:
        return {}

    cols = ", ".join(filtered.keys())
    placeholders = ", ".join(["%s"] * len(filtered))
    updates = ", ".join(f"{k} = EXCLUDED.{k}" for k in filtered)

    sql = f"""
        INSERT INTO bot_configs (tenant_id, {cols})
        VALUES (%s, {placeholders})
        ON CONFLICT (tenant_id) DO UPDATE SET {updates}, updated_at = NOW()
        RETURNING *;
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (tenant_id, *filtered.values()))
            row = cur.fetchone()
            conn.commit()
            return _serialize(row)
    finally:
        conn.close()


# ══════════════════════════════════════════════════════════
#  MODULE CONFIG operations
# ══════════════════════════════════════════════════════════

def get_tenant_modules(tenant_id: str) -> Dict[str, bool]:
    """
    Returns a dict of {module_name: is_enabled} for the tenant.
    Used internally by brain.py / module_registry.
    Missing rows are backfilled with safe defaults (BOOKING=True, FACTS=False).
    """
    sql = "SELECT module_name, is_enabled FROM module_configs WHERE tenant_id = %s;"
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (tenant_id,))
            rows = cur.fetchall()
        result = {BOOKING_MODULE: True, FACTS_MODULE: False}   # safe defaults
        for row in rows:
            result[row["module_name"]] = row["is_enabled"]
        return result
    finally:
        conn.close()


def get_module_configs_list(tenant_id: str) -> List[Dict[str, Any]]:
    """
    Returns module configs as a LIST for the admin panel API.
    Shape: [{"module_name": "BOOKING_MODULE", "is_enabled": true}, ...]
    Always returns both known modules (seeds defaults if rows missing).
    This is what fixes the 'modules.map is not a function' frontend error.
    """
    # Ensure rows exist for all known modules (idempotent, safe to call repeatedly)
    _seed_module_defaults_for_tenant(tenant_id)
    sql = """
        SELECT module_name, is_enabled, updated_at
        FROM module_configs
        WHERE tenant_id = %s
        ORDER BY module_name ASC;
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (tenant_id,))
            rows = cur.fetchall()
        return [_serialize(r) for r in rows]
    finally:
        conn.close()


def _seed_module_defaults_for_tenant(tenant_id: str):
    """Idempotently ensure BOOKING_MODULE and FACTS_MODULE rows exist for this tenant."""
    sql = """
        INSERT INTO module_configs (tenant_id, module_name, is_enabled)
        VALUES (%s, 'BOOKING_MODULE', TRUE), (%s, 'FACTS_MODULE', FALSE)
        ON CONFLICT (tenant_id, module_name) DO NOTHING;
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (tenant_id, tenant_id))
        conn.commit()
    finally:
        conn.close()


def set_module_enabled(tenant_id: str, module_name: str, is_enabled: bool) -> Dict[str, Any]:
    """Enable or disable a module for a tenant (upsert)."""
    sql = """
        INSERT INTO module_configs (tenant_id, module_name, is_enabled)
        VALUES (%s, %s, %s)
        ON CONFLICT (tenant_id, module_name)
        DO UPDATE SET is_enabled = EXCLUDED.is_enabled, updated_at = NOW()
        RETURNING *;
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (tenant_id, module_name, is_enabled))
            row = cur.fetchone()
            conn.commit()
            return _serialize(row)
    finally:
        conn.close()


# ══════════════════════════════════════════════════════════
#  KNOWLEDGE BASE operations  (FACTS_MODULE)
# ══════════════════════════════════════════════════════════

def add_knowledge_chunks(tenant_id: str, chunks: List[str]) -> int:
    """Insert text chunks into knowledge_base. Returns count inserted."""
    sql = """
        INSERT INTO knowledge_base (tenant_id, content)
        VALUES (%s, %s);
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            for chunk in chunks:
                cur.execute(sql, (tenant_id, chunk.strip()))
        conn.commit()
        return len(chunks)
    finally:
        conn.close()


def get_knowledge_chunks(tenant_id: str) -> List[Dict[str, Any]]:
    """Return all knowledge chunks for a tenant (used for admin preview)."""
    sql = "SELECT * FROM knowledge_base WHERE tenant_id = %s ORDER BY created_at ASC;"
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (tenant_id,))
            return [_serialize(r) for r in cur.fetchall()]
    finally:
        conn.close()


def delete_all_knowledge(tenant_id: str) -> int:
    """Delete all knowledge chunks for a tenant. Returns rows deleted."""
    sql = "DELETE FROM knowledge_base WHERE tenant_id = %s;"
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (tenant_id,))
            count = cur.rowcount
            conn.commit()
            return count
    finally:
        conn.close()


# ══════════════════════════════════════════════════════════
#  USER operations  (tenant-scoped)
# ══════════════════════════════════════════════════════════

def create_user_if_not_exists(phone_number: str, name: Optional[str] = None,
                               tenant_id: Optional[str] = None) -> Dict[str, Any]:
    if tenant_id:
        sql_insert = """
            INSERT INTO users (tenant_id, phone_number, name)
            VALUES (%s, %s, %s)
            ON CONFLICT (tenant_id, phone_number) DO NOTHING;
        """
        sql_select = "SELECT * FROM users WHERE tenant_id = %s AND phone_number = %s;"
        params_insert = (tenant_id, phone_number, name)
        params_select = (tenant_id, phone_number)
    else:
        sql_insert = """
            INSERT INTO users (tenant_id, phone_number, name)
            VALUES ((SELECT tenant_id FROM tenants LIMIT 1), %s, %s)
            ON CONFLICT DO NOTHING;
        """
        sql_select = "SELECT * FROM users WHERE phone_number = %s LIMIT 1;"
        params_insert = (phone_number, name)
        params_select = (phone_number,)

    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(sql_insert, params_insert)
            conn.commit()
            cur.execute(sql_select, params_select)
            row = cur.fetchone()
            return _serialize(row) if row else {}
    finally:
        conn.close()


def get_user_appointments(phone_number: str, tenant_id: Optional[str] = None) -> List[Dict[str, Any]]:
    if tenant_id:
        sql = """
            SELECT * FROM appointments
            WHERE tenant_id = %s AND phone_number = %s
            ORDER BY start_time DESC;
        """
        params = (tenant_id, phone_number)
    else:
        sql = "SELECT * FROM appointments WHERE phone_number = %s ORDER BY start_time DESC;"
        params = (phone_number,)

    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return [_serialize(r) for r in cur.fetchall()]
    finally:
        conn.close()


def get_tenant_users(tenant_id: str) -> List[Dict[str, Any]]:
    sql = """
        SELECT u.*, COUNT(a.appointment_id) AS appointment_count
        FROM users u
        LEFT JOIN appointments a ON a.tenant_id = u.tenant_id AND a.phone_number = u.phone_number
        WHERE u.tenant_id = %s
        GROUP BY u.user_id
        ORDER BY u.created_at DESC;
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (tenant_id,))
            return [_serialize(r) for r in cur.fetchall()]
    finally:
        conn.close()


# ══════════════════════════════════════════════════════════
#  APPOINTMENT operations  (tenant-scoped)
# ══════════════════════════════════════════════════════════

def create_appointment(phone_number: str, start_time: str, end_time: str,
                        calendar_event_id: str, tenant_id: Optional[str] = None) -> Dict[str, Any]:
    sql = """
        INSERT INTO appointments
            (tenant_id, phone_number, start_time, end_time, calendar_event_id)
        VALUES (%s, %s, %s, %s, %s)
        RETURNING *;
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (tenant_id, phone_number,
                              _to_dt(start_time), _to_dt(end_time), calendar_event_id))
            row = cur.fetchone()
            conn.commit()
            return _serialize(row)
    finally:
        conn.close()


def get_tenant_appointments_for_date(tenant_id: str, date_str: str) -> List[Dict[str, Any]]:
    sql = """
        SELECT a.*, u.name AS customer_name
        FROM appointments a
        LEFT JOIN users u ON u.tenant_id = a.tenant_id AND u.phone_number = a.phone_number
        WHERE a.tenant_id = %s AND a.start_time::date = %s::date
        ORDER BY a.start_time ASC;
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (tenant_id, date_str))
            return [_serialize(r) for r in cur.fetchall()]
    finally:
        conn.close()


def get_tenant_appointments_range(tenant_id: str, from_date: str, to_date: str) -> List[Dict[str, Any]]:
    sql = """
        SELECT a.*, u.name AS customer_name
        FROM appointments a
        LEFT JOIN users u ON u.tenant_id = a.tenant_id AND u.phone_number = a.phone_number
        WHERE a.tenant_id = %s
          AND a.start_time BETWEEN %s AND %s
        ORDER BY a.start_time ASC;
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (tenant_id, from_date, to_date))
            return [_serialize(r) for r in cur.fetchall()]
    finally:
        conn.close()


def get_tenant_stats(tenant_id: str) -> Dict[str, Any]:
    sql = """
        SELECT
            COUNT(*) FILTER (WHERE status = 'BOOKED' AND start_time >= NOW())   AS upcoming,
            COUNT(*) FILTER (WHERE start_time::date = CURRENT_DATE)             AS today_total,
            COUNT(*) FILTER (WHERE status = 'CANCELLED'
                             AND created_at >= NOW() - INTERVAL '30 days')      AS cancelled_30d,
            COUNT(*) FILTER (WHERE created_at >= NOW() - INTERVAL '30 days')    AS booked_30d,
            COUNT(DISTINCT phone_number)                                         AS unique_customers
        FROM appointments
        WHERE tenant_id = %s;
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (tenant_id,))
            return _serialize(cur.fetchone())
    finally:
        conn.close()


def get_calendar_event_id(phone_number: str, start_time: str,
                           tenant_id: Optional[str] = None) -> Optional[str]:
    if tenant_id:
        sql = """
            SELECT calendar_event_id FROM appointments
            WHERE tenant_id = %s AND phone_number = %s AND start_time = %s AND status = 'BOOKED'
            LIMIT 1;
        """
        params = (tenant_id, phone_number, _to_dt(start_time))
    else:
        sql = """
            SELECT calendar_event_id FROM appointments
            WHERE phone_number = %s AND start_time = %s AND status = 'BOOKED'
            LIMIT 1;
        """
        params = (phone_number, _to_dt(start_time))

    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            row = cur.fetchone()
            return row["calendar_event_id"] if row else None
    finally:
        conn.close()


def update_appointment_status(phone_number: str, start_time: str, status: str,
                               tenant_id: Optional[str] = None) -> bool:
    _validate_status(status)
    if tenant_id:
        sql = """
            UPDATE appointments SET status = %s
            WHERE tenant_id = %s AND phone_number = %s AND start_time = %s AND status = 'BOOKED';
        """
        params = (status, tenant_id, phone_number, _to_dt(start_time))
    else:
        sql = """
            UPDATE appointments SET status = %s
            WHERE phone_number = %s AND start_time = %s AND status = 'BOOKED';
        """
        params = (status, phone_number, _to_dt(start_time))

    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            updated = cur.rowcount > 0
            conn.commit()
            return updated
    finally:
        conn.close()


def update_status_by_event_id(calendar_event_id: str, status: str) -> bool:
    _validate_status(status)
    sql = """
        UPDATE appointments SET status = %s
        WHERE calendar_event_id = %s AND status = 'BOOKED';
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (status, calendar_event_id))
            updated = cur.rowcount > 0
            conn.commit()
            return updated
    finally:
        conn.close()


def update_rescheduled_appointment(phone_number: str, old_start_time: str,
                                    new_start_time: str, new_end_time: Optional[str] = None,
                                    tenant_id: Optional[str] = None) -> bool:
    base = """
        UPDATE appointments
        SET start_time = %s,
            end_time   = COALESCE(%s, end_time + (%s::timestamp - start_time)),
            status     = 'RESCHEDULED'
        WHERE phone_number = %s AND start_time = %s AND status = 'BOOKED'
    """
    params = [_to_dt(new_start_time),
              _to_dt(new_end_time) if new_end_time else None,
              _to_dt(new_start_time),
              phone_number,
              _to_dt(old_start_time)]

    if tenant_id:
        base += " AND tenant_id = %s"
        params.append(tenant_id)

    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(base + ";", params)
            updated = cur.rowcount > 0
            conn.commit()
            return updated
    finally:
        conn.close()


def user_owns_appointment(phone_number: str, start_time: str,
                           tenant_id: Optional[str] = None) -> bool:
    base = """
        SELECT 1 FROM appointments
        WHERE phone_number = %s
          AND start_time BETWEEN %s::timestamp - INTERVAL '1 minute'
                            AND %s::timestamp + INTERVAL '1 minute'
          AND status = 'BOOKED'
    """
    params = [phone_number, _to_dt(start_time), _to_dt(start_time)]
    if tenant_id:
        base += " AND tenant_id = %s"
        params.append(tenant_id)

    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(base + " LIMIT 1;", params)
            return cur.fetchone() is not None
    finally:
        conn.close()


# ══════════════════════════════════════════════════════════
#  CALENDAR TOKEN operations
# ══════════════════════════════════════════════════════════

def save_calendar_token(tenant_id: str, calendar_id: str, token_json: str) -> Dict[str, Any]:
    sql = """
        INSERT INTO calendar_tokens (tenant_id, calendar_id, token_json)
        VALUES (%s, %s, %s)
        ON CONFLICT (tenant_id) DO UPDATE
            SET calendar_id = EXCLUDED.calendar_id,
                token_json  = EXCLUDED.token_json,
                connected_at = NOW()
        RETURNING *;
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (tenant_id, calendar_id, token_json))
            row = cur.fetchone()
            conn.commit()
            return _serialize(row)
    finally:
        conn.close()


def get_calendar_token(tenant_id: str) -> Optional[Dict[str, Any]]:
    sql = "SELECT * FROM calendar_tokens WHERE tenant_id = %s;"
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (tenant_id,))
            row = cur.fetchone()
            return _serialize(row) if row else None
    finally:
        conn.close()


# ══════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════

def _validate_status(status: str):
    valid = {"BOOKED", "CANCELLED", "RESCHEDULED"}
    if status not in valid:
        raise ValueError(f"Invalid status '{status}'. Must be one of {valid}")


def _serialize(row) -> Dict[str, Any]:
    """Convert a psycopg2 RealDictRow to a plain dict, stringifying UUIDs & datetimes."""
    if row is None:
        return {}
    r = dict(row)
    for k, v in r.items():
        if hasattr(v, 'isoformat'):   # datetime / date
            r[k] = v.isoformat()
        elif hasattr(v, 'hex'):       # UUID
            r[k] = str(v)
    return r


# ══════════════════════════════════════════════════════════
#  SINGLE-ENTRY knowledge helpers (used by admin endpoints)
# ══════════════════════════════════════════════════════════

def add_knowledge(tenant_id: str, chunks_or_text) -> int:
    """
    Flexible helper: accepts either a list[str] (chunks) or a raw str (single entry).
    Returns count of rows inserted.
    """
    if isinstance(chunks_or_text, str):
        chunks = [chunks_or_text]
    else:
        chunks = list(chunks_or_text)
    return add_knowledge_chunks(tenant_id, chunks)


def get_all_knowledge(tenant_id: str):
    """Alias for get_knowledge_chunks."""
    return get_knowledge_chunks(tenant_id)


def delete_knowledge(knowledge_id: str, tenant_id: str) -> bool:
    """Delete a single knowledge entry by id."""
    sql = "DELETE FROM knowledge_base WHERE id = %s AND tenant_id = %s;"
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (knowledge_id, tenant_id))
            deleted = cur.rowcount > 0
            conn.commit()
            return deleted
    finally:
        conn.close()


def get_all_module_configs(tenant_id: str) -> list:
    """
    Returns module configs as a LIST of dicts for the admin panel API.
    Always returns a list — fixes the 'modules.map is not a function' JS error.
    Delegates to get_module_configs_list() which reads directly from module_configs table.
    """
    return get_module_configs_list(tenant_id)


def get_enabled_modules(tenant_id: str) -> list:
    """Returns list of enabled module name strings."""
    modules_dict = get_tenant_modules(tenant_id)
    return [name for name, on in modules_dict.items() if on]