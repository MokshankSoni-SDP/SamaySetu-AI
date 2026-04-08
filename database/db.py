"""
database/db.py
--------------
PostgreSQL connection management for SamaySetu AI.

DEVELOPER ACTION REQUIRED:
  Replace the placeholder credentials in DATABASE_URL before first run.
  Example real value:
    postgresql://samaysetu:secretpassword@localhost:5432/samaysetu_db

To create the database in psql:
    CREATE USER samaysetu WITH PASSWORD 'secretpassword';
    CREATE DATABASE samaysetu_db OWNER samaysetu;
"""

import os
import psycopg2
from psycopg2.extras import RealDictCursor

# ── Database URL ─────────────────────────────────────────────────────────────
# Replace with your actual PostgreSQL credentials, or set DATABASE_URL env var.
DATABASE_URL = os.getenv("DATABASE_URL")
CA_CERT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ca.pem")

def get_db_connection():
    """
    Returns a new psycopg2 connection using DATABASE_URL.
    Caller is responsible for closing the connection.

    Usage:
        conn = get_db_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT 1")
        finally:
            conn.close()
    """
    try:
        connect_kwargs = {"cursor_factory": RealDictCursor}
        
        # Configure SSL for Aiven PostgreSQL if ca.pem exists
        if os.path.exists(CA_CERT_PATH) and DATABASE_URL and "aivencloud" in DATABASE_URL:
            connect_kwargs["sslmode"] = "verify-ca"
            connect_kwargs["sslrootcert"] = CA_CERT_PATH
            
        conn = psycopg2.connect(DATABASE_URL, **connect_kwargs)
        return conn
    except psycopg2.OperationalError as e:
        raise ConnectionError(
            f"[DB] Could not connect to PostgreSQL.\n"
            f"Check your DATABASE_URL: {DATABASE_URL}\n"
            f"Original error: {e}"
        ) from e
