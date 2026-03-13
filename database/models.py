"""
database/models.py
------------------
Multi-tenant table definitions for SamaySetu AI.
Supports multiple business owners (tenants), each with their own customers,
appointments, calendar integration, and bot configuration.

Tables:
  tenants          — one row per business (doctor, salon, etc.)
  tenant_admins    — admin login accounts linked to a tenant
  users            — end-customers, scoped to a tenant
  appointments     — appointments scoped to tenant + user
  bot_configs      — per-tenant bot personality/prompt config
  calendar_tokens  — per-tenant Google Calendar OAuth tokens
"""

from database.db import get_db_connection


def create_tables():
    stmts = [

        # ── Tenants: one row = one business owner using SamaySetu ────────────
        """
        CREATE TABLE IF NOT EXISTS tenants (
            tenant_id       UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
            business_name   VARCHAR(150) NOT NULL,
            business_type   VARCHAR(50)  NOT NULL DEFAULT 'general',
            owner_email     VARCHAR(200) UNIQUE NOT NULL,
            plan            VARCHAR(20)  NOT NULL DEFAULT 'free'
                                CHECK (plan IN ('free','starter','pro','enterprise')),
            is_active       BOOLEAN     NOT NULL DEFAULT TRUE,
            created_at      TIMESTAMP   DEFAULT CURRENT_TIMESTAMP
        );
        """,

        # ── Tenant admins: login accounts for shop owners ─────────────────────
        """
        CREATE TABLE IF NOT EXISTS tenant_admins (
            admin_id        UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
            tenant_id       UUID        NOT NULL REFERENCES tenants(tenant_id) ON DELETE CASCADE,
            email           VARCHAR(200) UNIQUE NOT NULL,
            password_hash   TEXT        NOT NULL,
            role            VARCHAR(20) NOT NULL DEFAULT 'admin'
                                CHECK (role IN ('owner','admin','staff')),
            created_at      TIMESTAMP   DEFAULT CURRENT_TIMESTAMP
        );
        """,

        # ── Bot configs: per-tenant LLM personality and booking settings ──────
        """
        CREATE TABLE IF NOT EXISTS bot_configs (
            config_id           UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
            tenant_id           UUID        UNIQUE NOT NULL REFERENCES tenants(tenant_id) ON DELETE CASCADE,
            bot_name            VARCHAR(100) NOT NULL DEFAULT 'SamaySetu AI',
            receptionist_name   VARCHAR(100) NOT NULL DEFAULT 'Priya',
            language_code       VARCHAR(10)  NOT NULL DEFAULT 'gu-IN',
            tts_speaker         VARCHAR(50)  NOT NULL DEFAULT 'simran',
            business_hours_start INTEGER    NOT NULL DEFAULT 9,
            business_hours_end   INTEGER    NOT NULL DEFAULT 18,
            slot_duration_mins   INTEGER    NOT NULL DEFAULT 30,
            silence_timeout_ms   INTEGER    NOT NULL DEFAULT 1500,
            greeting_message    TEXT,
            business_description TEXT,
            extra_prompt_context TEXT,
            calendar_id         TEXT,
            updated_at          TIMESTAMP   DEFAULT CURRENT_TIMESTAMP
        );
        """,

        # ── Users: end-customers, always scoped to a tenant ──────────────────
        """
        CREATE TABLE IF NOT EXISTS users (
            user_id         UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
            tenant_id       UUID        NOT NULL REFERENCES tenants(tenant_id) ON DELETE CASCADE,
            phone_number    VARCHAR(15) NOT NULL,
            name            VARCHAR(100),
            created_at      TIMESTAMP   DEFAULT CURRENT_TIMESTAMP,
            UNIQUE (tenant_id, phone_number)
        );
        """,

        # ── Appointments: fully scoped to tenant + user ───────────────────────
        """
        CREATE TABLE IF NOT EXISTS appointments (
            appointment_id    UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
            tenant_id         UUID        NOT NULL REFERENCES tenants(tenant_id) ON DELETE CASCADE,
            phone_number      VARCHAR(15) NOT NULL,
            start_time        TIMESTAMP   NOT NULL,
            end_time          TIMESTAMP,
            calendar_event_id TEXT,
            notes             TEXT,
            status            VARCHAR(20) NOT NULL DEFAULT 'BOOKED'
                                CHECK (status IN ('BOOKED', 'CANCELLED', 'RESCHEDULED')),
            created_at        TIMESTAMP   DEFAULT CURRENT_TIMESTAMP
        );
        """,

        # ── Calendar tokens: per-tenant OAuth credentials ─────────────────────
        """
        CREATE TABLE IF NOT EXISTS calendar_tokens (
            token_id        UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
            tenant_id       UUID        UNIQUE NOT NULL REFERENCES tenants(tenant_id) ON DELETE CASCADE,
            calendar_id     TEXT        NOT NULL,
            token_json      TEXT        NOT NULL,
            connected_at    TIMESTAMP   DEFAULT CURRENT_TIMESTAMP,
            expires_at      TIMESTAMP
        );
        """,

        # ── Indexes ────────────────────────────────────────────────────────────
        "CREATE INDEX IF NOT EXISTS idx_appt_tenant_phone   ON appointments (tenant_id, phone_number, start_time DESC);",
        "CREATE INDEX IF NOT EXISTS idx_appt_tenant_date    ON appointments (tenant_id, start_time);",
        "CREATE INDEX IF NOT EXISTS idx_users_tenant_phone  ON users (tenant_id, phone_number);",
    ]

    try:
        conn = get_db_connection()
        try:
            with conn.cursor() as cur:
                for stmt in stmts:
                    cur.execute(stmt)
            conn.commit()
            print("[DB] All tables created / verified successfully.")
        finally:
            conn.close()
    except ConnectionError as e:
        print(f"[DB] Warning: could not create tables. Database features disabled.\n{e}")
    except Exception as e:
        print(f"[DB] Unexpected error during table creation: {e}")