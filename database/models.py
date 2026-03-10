"""
database/models.py
------------------
Table definitions for SamaySetu AI.
Exposes create_tables() which is called on FastAPI startup.
Uses CREATE TABLE IF NOT EXISTS so it is safe to call repeatedly.
"""

from database.db import get_db_connection


def create_tables():
    """
    Creates the 'users' and 'appointments' tables if they do not already exist.
    Called once on application startup via FastAPI lifespan event.
    """
    create_users_sql = """
        CREATE TABLE IF NOT EXISTS users (
            phone_number VARCHAR(15)  PRIMARY KEY,
            name         VARCHAR(100),
            created_at   TIMESTAMP    DEFAULT CURRENT_TIMESTAMP
        );
    """

    create_appointments_sql = """
        CREATE TABLE IF NOT EXISTS appointments (
            appointment_id    UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
            phone_number      VARCHAR(15) NOT NULL REFERENCES users(phone_number) ON DELETE CASCADE,
            start_time        TIMESTAMP   NOT NULL,
            end_time          TIMESTAMP,
            calendar_event_id TEXT,
            status            VARCHAR(20) NOT NULL DEFAULT 'BOOKED'
                                CHECK (status IN ('BOOKED', 'CANCELLED', 'RESCHEDULED')),
            created_at        TIMESTAMP   DEFAULT CURRENT_TIMESTAMP
        );
    """

    # Index for fast per-user lookups
    create_index_sql = """
        CREATE INDEX IF NOT EXISTS idx_appointments_phone
            ON appointments (phone_number, start_time DESC);
    """

    try:
        conn = get_db_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(create_users_sql)
                cur.execute(create_appointments_sql)
                cur.execute(create_index_sql)
            conn.commit()
            print("[DB] Tables created (or already exist) successfully.")
        finally:
            conn.close()
    except ConnectionError as e:
        # Server can still start; DB features will degrade gracefully
        print(f"[DB] Warning: could not create tables. Database features disabled.\n{e}")
    except Exception as e:
        print(f"[DB] Unexpected error during table creation: {e}")
