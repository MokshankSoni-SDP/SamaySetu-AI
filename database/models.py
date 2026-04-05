"""
database/models.py
------------------
Multi-tenant table definitions for SamaySetu AI.
Now includes module_configs and knowledge_base tables.
"""

from database.db import get_db_connection


def create_tables():
    stmts = [

        # ── Tenants ────────────────────────────────────────────────────────────
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

        # ── Tenant admins ─────────────────────────────────────────────────────
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

        # ── Bot configs ────────────────────────────────────────────────────────
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

        # ── Users ──────────────────────────────────────────────────────────────
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

        # ── Appointments ───────────────────────────────────────────────────────
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

        # ── Calendar tokens ────────────────────────────────────────────────────
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

        # ── NEW: Module configs — controls which modules are active per tenant ─
        """
        CREATE TABLE IF NOT EXISTS module_configs (
            id              UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
            tenant_id       UUID        NOT NULL REFERENCES tenants(tenant_id) ON DELETE CASCADE,
            module_name     VARCHAR(50) NOT NULL,
            is_enabled      BOOLEAN     NOT NULL DEFAULT FALSE,
            created_at      TIMESTAMP   DEFAULT CURRENT_TIMESTAMP,
            updated_at      TIMESTAMP   DEFAULT CURRENT_TIMESTAMP,
            UNIQUE (tenant_id, module_name)
        );
        """,

        # ── NEW: Knowledge base — raw content for FACTS_MODULE (RAG) ──────────
        """
        CREATE TABLE IF NOT EXISTS knowledge_base (
            id              UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
            tenant_id       UUID        NOT NULL REFERENCES tenants(tenant_id) ON DELETE CASCADE,
            content         TEXT        NOT NULL,
            created_at      TIMESTAMP   DEFAULT CURRENT_TIMESTAMP
        );
        """,

        # ── Schema migrations ─────────────────────────────────────────────────
        "ALTER TABLE bot_configs ADD COLUMN IF NOT EXISTS business_hours_periods TEXT;",
        """
        UPDATE bot_configs
        SET business_hours_periods = json_build_array(
            json_build_object('start', business_hours_start, 'end', business_hours_end)
        )::text
        WHERE business_hours_periods IS NULL OR business_hours_periods = '';
        """,

        # ── Indexes ────────────────────────────────────────────────────────────
        "CREATE INDEX IF NOT EXISTS idx_appt_tenant_phone   ON appointments (tenant_id, phone_number, start_time DESC);",
        "CREATE INDEX IF NOT EXISTS idx_appt_tenant_date    ON appointments (tenant_id, start_time);",
        "CREATE INDEX IF NOT EXISTS idx_users_tenant_phone  ON users (tenant_id, phone_number);",
        "CREATE INDEX IF NOT EXISTS idx_module_configs_tenant ON module_configs (tenant_id, module_name);",
        "CREATE INDEX IF NOT EXISTS idx_knowledge_tenant    ON knowledge_base (tenant_id);",
    ]

    try:
        conn = get_db_connection()
        try:
            with conn.cursor() as cur:
                for stmt in stmts:
                    cur.execute(stmt)
            conn.commit()
            # Seed BOOKING_MODULE as enabled by default for all existing tenants
            _seed_default_modules(conn)
            print("[DB] All tables created / verified successfully.")
        finally:
            conn.close()
    except ConnectionError as e:
        print(f"[DB] Warning: could not create tables. Database features disabled.\n{e}")
    except Exception as e:
        print(f"[DB] Unexpected error during table creation: {e}")


def _seed_default_modules(conn):
    """
    Ensures every existing tenant has BOOKING_MODULE enabled by default.
    FACTS_MODULE defaults to disabled. Safe to run multiple times (ON CONFLICT DO NOTHING).
    """
    sql = """
        INSERT INTO module_configs (tenant_id, module_name, is_enabled)
        SELECT tenant_id, 'BOOKING_MODULE', TRUE FROM tenants
        ON CONFLICT (tenant_id, module_name) DO NOTHING;

        INSERT INTO module_configs (tenant_id, module_name, is_enabled)
        SELECT tenant_id, 'FACTS_MODULE', FALSE FROM tenants
        ON CONFLICT (tenant_id, module_name) DO NOTHING;
    """
    with conn.cursor() as cur:
        cur.execute(sql)
    conn.commit()
    print("[DB] Default module configs seeded.")
