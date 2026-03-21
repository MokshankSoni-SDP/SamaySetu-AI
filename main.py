import os
import re
import re as _re
import json
import base64
import asyncio
import struct
import traceback
import hashlib
import secrets
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends, Header
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sarvamai import AsyncSarvamAI, AudioOutput
from dotenv import load_dotenv
load_dotenv()

from typing import Dict, List, Optional
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from services.calendar_provider import verify_calendar_connection

import prompts
import calendar_tool
from calendar_tool import *

# ── Tenant Context ───────────────────────────────────────────────────────────
class TenantContext:
    """Lightweight carrier that tells calendar_tool which session is active."""
    session_id: Optional[str] = None
    chat_sessions: Optional[dict] = None

tenant_context = TenantContext()


def get_current_tenant() -> Optional[str]:
    """Return the tenant_id for the active session, or None."""
    if not tenant_context.chat_sessions or not tenant_context.session_id:
        return None
    session = tenant_context.chat_sessions.get(tenant_context.session_id)
    return session.get("tenant_id") if session else None

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from datetime import datetime, date

GROQ_SMALL_LLM_KEY = os.getenv("GROQ_SMALL_LLM")

small_llm = ChatGroq(
    api_key=GROQ_SMALL_LLM_KEY,
    model="llama-3.1-8b-instant",   # or any 8B Groq model
    temperature=0
)

# ── Database imports ────────────────────────────────────────────────────────
try:
    from database.models import create_tables
    from database.crud import (
        create_user_if_not_exists,
        get_user_appointments,
        get_tenant_by_id,
        get_bot_config,
        upsert_bot_config,
        get_tenant_users,
        get_tenant_stats,
        get_tenant_appointments_for_date,
        get_tenant_appointments_range,
        get_all_tenants,
        create_tenant,
        get_platform_stats,
        update_tenant_status,
        create_tenant_admin,
        get_admin_by_email,
        save_calendar_token,
        get_calendar_token,
    )
    DB_AVAILABLE = True
except ImportError as _db_import_err:
    DB_AVAILABLE = False
    print(f"[DB] Database module missing: {_db_import_err}")


def log(step: str, msg: str):
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{ts}] {step}: {msg}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    if DB_AVAILABLE:
        await asyncio.to_thread(create_tables)
    else:
        print("[DB] Skipping table creation — psycopg2 unavailable.")
    yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Pydantic models ─────────────────────────────────────────────────────────

class UserLoginRequest(BaseModel):
    phone_number: str
    name: Optional[str] = None
    tenant_id: Optional[str] = None   # Optional — resolved from host/header if not sent

class AdminLoginRequest(BaseModel):
    email: str
    password: str

class AdminRegisterRequest(BaseModel):
    email: str
    password: str
    tenant_id: str
    role: str = "admin"

class TenantCreateRequest(BaseModel):
    business_name: str
    business_type: str = "general"
    owner_email: str
    admin_password: str

class BotConfigRequest(BaseModel):
    bot_name: Optional[str] = None
    receptionist_name: Optional[str] = None
    language_code: Optional[str] = None
    tts_speaker: Optional[str] = None
    business_hours_start: Optional[int] = None
    business_hours_end: Optional[int] = None
    slot_duration_mins: Optional[int] = None
    silence_timeout_ms: Optional[int] = None
    greeting_message: Optional[str] = None
    business_description: Optional[str] = None
    extra_prompt_context: Optional[str] = None
    calendar_id: Optional[str] = None

class CalendarConnectRequest(BaseModel):
    calendar_id: str
    service_account_json: str   # JSON string of the service account credentials


SARVAM_API_KEY = os.getenv('SARVAM_API_KEY')
client_stt = AsyncSarvamAI(api_subscription_key=SARVAM_API_KEY)
client_tts = AsyncSarvamAI(api_subscription_key=SARVAM_API_KEY)

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1)
tools = [check_calendar_availability, book_appointment, cancel_appointment,
         reschedule_appointment, suggest_next_available_slot]
llm_with_tools = llm.bind_tools(tools)

# ── HTML page routes (must come BEFORE the /static mount) ───────────────────
@app.get("/")
async def serve_customer():
    return FileResponse("static/index.html")

@app.get("/admin")
async def serve_admin():
    return FileResponse("static/admin.html")

@app.get("/superadmin")
async def serve_superadmin():
    return FileResponse("static/superadmin.html")

# ── Public endpoint: bot config for customer UI (no auth needed) ─────────────
@app.get("/public/bot-config")
async def public_bot_config(tenant_id: Optional[str] = None):
    """
    Returns the safe subset of bot_config that the customer UI needs
    (name, description, greeting, hours, language, speaker).
    No secrets, no calendar credentials.
    """
    if not tenant_id or not DB_AVAILABLE:
        return {}
    try:
        cfg = await asyncio.to_thread(get_bot_config, tenant_id)
        if not cfg:
            return {}
        # Only expose safe fields to the public
        return {k: cfg[k] for k in (
            "bot_name", "receptionist_name", "language_code", "tts_speaker",
            "business_hours_start", "business_hours_end", "slot_duration_mins",
            "silence_timeout_ms", "greeting_message", "business_description",
        ) if k in cfg}
    except Exception:
        return {}

app.mount("/static", StaticFiles(directory="static"), name="static")

# ── In-memory session store ──────────────────────────────────────────────────
# chat_sessions maps session_id → {history, phone_number, tenant_id, bot_config}
chat_sessions: Dict[str, dict] = {}
# admin_sessions maps token → {tenant_id, email, role}
admin_sessions: Dict[str, dict] = {}
# super_admin_sessions maps token → {email}
superadmin_sessions: Dict[str, dict] = {}

MAX_HISTORY = 5
MAX_TOOL_ITERATIONS = 2

# Gujarati number words → digits
# Includes Gujarati Unicode + phonetic romanized STT variants (Fix 3)
GUJ_NUMBERS = {
    # Authentic Gujarati script
    "એક": 1, "બે": 2, "ત્રણ": 3, "ચાર": 4,
    "પાંચ": 5, "છ": 6, "સાત": 7, "આઠ": 8,
    "નવ": 9, "દસ": 10, "અગિયાર": 11, "બાર": 12,
    # Phonetic romanized variants (common Sarvam STT outputs)
    "ek": 1, "eak": 1, "aek": 1, "1": 1,
    "be": 2, "bay": 2, "2": 2,
    "tran": 3, "teen": 3, "tin": 3, "3": 3,
    "char": 4, "chaar": 4, "4": 4,
    "panch": 5, "paanch": 5, "5": 5,
    "chha": 6, "chhah": 6, "cha": 6, "6": 6,
    "saat": 7, "sat": 7, "7": 7,
    "aath": 8, "ath": 8, "8": 8,
    "nav": 9, "9": 9,
    "das": 10, "10": 10,
    "agiyar": 11, "eleven": 11, "11": 11,
    "bar": 12, "bara": 12, "baara": 12, "twelve": 12, "12": 12,
}
SPECIAL_PHRASES = {"દોઢ": (1, 30), "અઢી": (2, 30)}
COMMON_STT_VARIANTS = {
    "સ્વા": "સવા", "સવા": "સવા",
    "સાડા": "સાડા", "પોના": "પોણા", "પોણા": "પોણા"
}
# Phonetic romanized prefix variants → standard Gujarati (Fix 3)
PHONETIC_PREFIX_MAP = {
    "sava ": "સવા ", "saava ": "સવા ",
    "sada ": "સાડા ", "saada ": "સાડા ", "sade ": "સાડા ",
    "pona ": "પોણા ", "pauna ": "પોણા ", "paune ": "પોણા ",
}

def is_noisy_transcript(text: str) -> bool:
    stripped = _re.sub(r'[\s.,!?।\u0964]+', ' ', text).strip()
    if len(stripped) <= 3:
        return True
    tokens = stripped.split()
    from collections import Counter
    counts = Counter(tokens)
    if any(v >= 3 for v in counts.values()):
        return True
    if len(tokens) >= 4:
        unique = len(set(tokens))
        if unique / len(tokens) < 0.5:
            return True
    return False

def normalize_variants(text):
    for wrong, correct in COMMON_STT_VARIANTS.items():
        text = text.replace(wrong, correct)
    return text

def normalize_phonetic_numbers(text: str) -> str:
    """Pre-pass: map phonetic romanized time-prefixes to Gujarati before regex (Fix 3)."""
    t = text.lower()
    for roman, gujarati in PHONETIC_PREFIX_MAP.items():
        t = t.replace(roman, gujarati)
    return t

def normalize_gujarati_time(text: str) -> str:
    text = text.lower().strip()
    # Fix 3 pre-pass — convert phonetic romanized prefixes to Gujarati
    text = normalize_phonetic_numbers(text)
    for phrase, (h, m) in SPECIAL_PHRASES.items():
        if phrase in text:
            text = text.replace(phrase, f"{h}:{m:02d}")
    text = normalize_variants(text)
    for prefix, minutes, offset in [("સવા", 15, 0), ("સાડા", 30, 0), ("પોણા", 45, -1)]:
        match = re.search(rf"{prefix}\s*(\w+)", text)
        if match:
            hour_word = match.group(1)
            # Fix 3: look up both Gujarati & romanized variants
            hour = GUJ_NUMBERS.get(hour_word) or GUJ_NUMBERS.get(hour_word.lower())
            if hour:
                h = hour + offset
                text = text.replace(match.group(0), f"{h}:{minutes}")
    return text

def _get_fallback_message(exc: Exception) -> str:
    err_str = str(exc).lower()
    if "rate_limit" in err_str or "429" in err_str:
        return "માફ કરશો, અત્યારે સર્વર ખૂબ વ્યસ્ત છે. થોડી વાર પછી ફરી પ્રયત્ન કરો."
    if "timeout" in err_str or "timed out" in err_str:
        return "માફ કરશો, જવાબ આવવામાં વધુ સમય લાગ્યો. કૃપા કરી ફરી પ્રયત્ન કરો."
    if "connection" in err_str or "network" in err_str:
        return "નેટવર્ક સમસ્યા આવી. કૃપા કરી ઇન્ટરનેટ તપાસો અને ફરી બોલો."
    return "માફ કરશો, કંઈક ભૂલ થઈ. થોડી વાર પછી ફરી પ્રયત્ન કરો."


# ── Auth helpers ──────────────────────────────────────────────────────────────

def _hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def _check_admin_token(x_admin_token: Optional[str] = Header(None)):
    if not x_admin_token or x_admin_token not in admin_sessions:
        raise HTTPException(status_code=401, detail="Invalid or missing admin token")
    return admin_sessions[x_admin_token]

def _check_superadmin_token(x_superadmin_token: Optional[str] = Header(None)):
    key = os.getenv("SUPERADMIN_SECRET", "changeme-superadmin")
    if x_superadmin_token != key:
        raise HTTPException(status_code=401, detail="Superadmin access denied")


# ── REST endpoints ────────────────────────────────────────────────────────────

# ── Customer endpoints ────────────────────────────────────────────────────────

@app.post("/user/login")
async def user_login(req: UserLoginRequest):
    if not req.phone_number or not req.phone_number.strip():
        raise HTTPException(status_code=400, detail="phone_number is required")
    if not DB_AVAILABLE:
        return {"status": "success", "phone_number": req.phone_number, "db": False}
    try:
        await asyncio.to_thread(
            create_user_if_not_exists,
            req.phone_number.strip(), req.name, req.tenant_id
        )
        return {"status": "success", "phone_number": req.phone_number.strip()}
    except Exception as e:
        print(f"[DB] /user/login error: {e}")
        raise HTTPException(status_code=500, detail="Database error during login")


@app.get("/appointments/{phone_number}")
async def get_appointments(phone_number: str, tenant_id: Optional[str] = None):
    if not DB_AVAILABLE:
        return []
    try:
        appts = await asyncio.to_thread(get_user_appointments, phone_number, tenant_id)
        return appts
    except Exception as e:
        raise HTTPException(status_code=500, detail="Could not fetch appointments")


# ── Admin auth endpoints ──────────────────────────────────────────────────────

@app.post("/admin/login")
async def admin_login(req: AdminLoginRequest):
    if not DB_AVAILABLE:
        raise HTTPException(status_code=503, detail="Database unavailable")
    admin = await asyncio.to_thread(get_admin_by_email, req.email)
    if not admin:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    if admin["password_hash"] != _hash_password(req.password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    if not admin.get("tenant_active", True):
        raise HTTPException(status_code=403, detail="This account is suspended")
    token = secrets.token_hex(32)
    admin_sessions[token] = {
        "tenant_id": admin["tenant_id"],
        "email": admin["email"],
        "role": admin["role"],
        "business_name": admin.get("business_name", ""),
    }
    return {"token": token, "tenant_id": admin["tenant_id"],
            "role": admin["role"], "business_name": admin.get("business_name")}


@app.post("/admin/logout")
async def admin_logout(session=Depends(_check_admin_token),
                       x_admin_token: Optional[str] = Header(None)):
    if x_admin_token in admin_sessions:
        del admin_sessions[x_admin_token]
    return {"status": "logged out"}


# ── Admin dashboard endpoints ─────────────────────────────────────────────────

@app.get("/admin/stats")
async def admin_stats(session=Depends(_check_admin_token)):
    tenant_id = session["tenant_id"]
    if not DB_AVAILABLE:
        return {}
    stats = await asyncio.to_thread(get_tenant_stats, tenant_id)
    return stats


@app.get("/admin/appointments/today")
async def admin_today(session=Depends(_check_admin_token)):
    tenant_id = session["tenant_id"]
    today = date.today().isoformat()
    if not DB_AVAILABLE:
        return []
    appts = await asyncio.to_thread(get_tenant_appointments_for_date, tenant_id, today)
    return appts


@app.get("/admin/appointments")
async def admin_appointments(session=Depends(_check_admin_token),
                              from_date: Optional[str] = None,
                              to_date: Optional[str] = None,
                              appt_date: Optional[str] = None):
    tenant_id = session["tenant_id"]
    if not DB_AVAILABLE:
        return []
    if appt_date:
        return await asyncio.to_thread(get_tenant_appointments_for_date, tenant_id, appt_date)
    if from_date and to_date:
        return await asyncio.to_thread(get_tenant_appointments_range, tenant_id, from_date, to_date)
    # Default: today
    today = date.today().isoformat()
    return await asyncio.to_thread(get_tenant_appointments_for_date, tenant_id, today)


@app.get("/admin/users")
async def admin_users(session=Depends(_check_admin_token)):
    tenant_id = session["tenant_id"]
    if not DB_AVAILABLE:
        return []
    return await asyncio.to_thread(get_tenant_users, tenant_id)


@app.get("/admin/bot-config")
async def admin_get_config(session=Depends(_check_admin_token)):
    tenant_id = session["tenant_id"]
    if not DB_AVAILABLE:
        return {}
    cfg = await asyncio.to_thread(get_bot_config, tenant_id)
    return cfg or {}


@app.post("/admin/bot-config")
async def admin_save_config(req: BotConfigRequest, session=Depends(_check_admin_token)):
    tenant_id = session["tenant_id"]
    if not DB_AVAILABLE:
        raise HTTPException(status_code=503, detail="Database unavailable")
    fields = {k: v for k, v in req.dict().items() if v is not None}
    cfg = await asyncio.to_thread(upsert_bot_config, tenant_id, **fields)
    return cfg


@app.post("/admin/calendar/connect")
async def admin_connect_calendar(
    req: CalendarConnectRequest,
    session=Depends(_check_admin_token),
):
    """
    Store Google Calendar credentials for this tenant AND verify they work.
 
    Steps:
      1. Validate the submitted JSON is parseable.
      2. Save credentials to the DB.
      3. Call the Google Calendar freebusy API to confirm the service account
         can actually access the calendar.
      4. Return connected=True only when step 3 succeeds; otherwise return an
         actionable error so the admin knows what to fix.
    """
    tenant_id = session["tenant_id"]
    if not DB_AVAILABLE:
        raise HTTPException(status_code=503, detail="Database unavailable")
 
    # Step 1 — validate JSON shape
    try:
        json.loads(req.service_account_json)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid service account JSON")
 
    # Step 2 — persist to DB
    await asyncio.to_thread(
        save_calendar_token, tenant_id, req.calendar_id, req.service_account_json
    )
    await asyncio.to_thread(upsert_bot_config, tenant_id, calendar_id=req.calendar_id)
 
    # Step 3 — live verification against Google Calendar API
    result = await asyncio.to_thread(verify_calendar_connection, tenant_id)
    if not result["ok"]:
        # Credentials saved but verification failed — tell the admin why
        raise HTTPException(
            status_code=422,
            detail=(
                f"Credentials saved but Google Calendar verification failed: "
                f"{result['error']}. "
                "Check that the service account has been shared with this calendar."
            ),
        )
 
    # Step 4 — all good
    return {
        "status": "connected",
        "verified": True,
        "calendar_id": req.calendar_id,
    }


@app.get("/admin/calendar/status")
async def admin_calendar_status(session=Depends(_check_admin_token)):
    """
    Returns real-time calendar connection status by actually calling Google.
    'connected: true' means credentials exist AND Google accepted them just now.
    """
    tenant_id = session["tenant_id"]
    if not DB_AVAILABLE:
        return {"connected": False}
 
    token = await asyncio.to_thread(get_calendar_token, tenant_id)
    if not token:
        return {"connected": False}
 
    # Run a live check instead of just returning "row exists"
    result = await asyncio.to_thread(verify_calendar_connection, tenant_id)
    if result["ok"]:
        return {
            "connected": True,
            "verified": True,
            "calendar_id": token["calendar_id"],
            "connected_at": token.get("connected_at"),
        }
    else:
        return {
            "connected": False,
            "verified": False,
            "calendar_id": token["calendar_id"],
            "error": result["error"],
        }


# ── Superadmin (SamaySetu platform owner) endpoints ──────────────────────────

@app.get("/superadmin/tenants", dependencies=[Depends(_check_superadmin_token)])
async def superadmin_tenants():
    if not DB_AVAILABLE:
        return []
    return await asyncio.to_thread(get_all_tenants)


@app.post("/superadmin/tenants", dependencies=[Depends(_check_superadmin_token)])
async def superadmin_create_tenant(req: TenantCreateRequest):
    if not DB_AVAILABLE:
        raise HTTPException(status_code=503, detail="Database unavailable")
    tenant = await asyncio.to_thread(
        create_tenant, req.business_name, req.business_type, req.owner_email
    )
    # Create default bot config
    await asyncio.to_thread(upsert_bot_config, tenant["tenant_id"],
                             bot_name=req.business_name)
    # Create owner admin account
    await asyncio.to_thread(
        create_tenant_admin,
        tenant["tenant_id"], req.owner_email,
        _hash_password(req.admin_password), "owner"
    )
    return tenant


@app.patch("/superadmin/tenants/{tenant_id}/status",
           dependencies=[Depends(_check_superadmin_token)])
async def superadmin_set_status(tenant_id: str, is_active: bool):
    if not DB_AVAILABLE:
        raise HTTPException(status_code=503, detail="Database unavailable")
    ok = await asyncio.to_thread(update_tenant_status, tenant_id, is_active)
    return {"updated": ok}


@app.get("/superadmin/stats", dependencies=[Depends(_check_superadmin_token)])
async def superadmin_stats():
    if not DB_AVAILABLE:
        return {}
    return await asyncio.to_thread(get_platform_stats)


# ── Token usage logger ────────────────────────────────────────────────────────
def print_token_usage(msg, step_name):
    usage = msg.response_metadata.get("token_usage", {})
    if usage:
        print(f"--- 📊 {step_name} | prompt={usage.get('prompt_tokens')} "
              f"completion={usage.get('completion_tokens')} "
              f"total={usage.get('total_tokens')} ---")

async def run_tool_async(tool_name: str, args: dict):
    func = globals()[tool_name]
    return await asyncio.to_thread(func, **args)

MIN_CHUNK_CHARS = 20

# ── Fix 2: TTS output filter — block tool calls / JSON from being spoken ─────
_TOOL_BLOCK_RE = re.compile(
    r'<function=[^>]*>.*?(?:</function>|$)'  # <function=name>...</function>
    r'|\{[^{}]{0,800}"(?:tool_name|function_name|arguments|tool_use_id)"[^{}]{0,800}\}',
    re.DOTALL,
)

def is_tool_output(text: str) -> bool:
    """Return True if the whole reply is a raw tool/JSON artifact (Fix 2)."""
    t = text.strip()
    return (
        t.startswith("{") or
        t.startswith("[{") or
        "<function=" in t or
        '"tool_name"' in t or
        '"function_name"' in t
    )

def clean_for_tts(text: str) -> str:
    """Strip tool call fragments from mixed LLM output before TTS (Fix 2)."""
    text = _TOOL_BLOCK_RE.sub("", text)
    # Remove any remaining bare JSON-looking blocks (conservative: must have quotes)
    text = re.sub(r'\{[^}]{0,500}\}', "", text)
    return text.strip()

# ── Fix 4: AI-echo similarity filter ─────────────────────────────────────────
def is_echo_of_ai(user_text: str, ai_text: str, threshold: float = 0.75) -> bool:
    """Jaccard token overlap — catches mic capturing AI's own speech (Fix 4)."""
    if not ai_text or not user_text:
        return False
    u_tokens = set(user_text.lower().split())
    a_tokens = set(ai_text.lower().split())
    if not u_tokens or len(u_tokens) < 3:  # too short to judge
        return False
    intersection = u_tokens & a_tokens
    union = u_tokens | a_tokens
    return len(intersection) / len(union) >= threshold

def split_into_sentences(text: str) -> List[str]:
    raw = re.split(r'(?<=[.!?।\u0964])\s+', text.strip())
    chunks, buffer = [], ""
    for part in raw:
        part = part.strip()
        if not part:
            continue
        buffer = (buffer + " " + part).strip() if buffer else part
        if len(buffer) >= MIN_CHUNK_CHARS:
            chunks.append(buffer)
            buffer = ""
    if buffer:
        if chunks and len(buffer) < MIN_CHUNK_CHARS:
            chunks[-1] += " " + buffer
        else:
            chunks.append(buffer)
    return chunks or [text]

# ── Fallback: non-streaming TTS (used only for error/fallback messages) ────────
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=4),
    retry=retry_if_exception_type(Exception),
    before_sleep=lambda rs: log("[TTS_RETRY]", f"Attempt {rs.attempt_number} failed…")
)
async def tts_convert(text: str, speaker: str = "simran", lang: str = "gu-IN") -> str:
    """Blocking HTTP TTS — only used for fallback/error messages."""
    res = await client_tts.text_to_speech.convert(
        text=text,
        target_language_code=lang,
        model="bulbul:v3",
        speaker=speaker
    )
    return res.audios[0]


# ── Streaming TTS session — one per voice WebSocket connection ────────────────
class StreamingTTSSession:
    """
    Streaming TTS via Bulbul v3 WebSocket.

    Architecture:
    - Opens a FRESH WS per response (clean lifecycle, no "iterator never exits" problem).
    - Runs two concurrent tasks inside each WS:
        sender:   sends each sentence + flush() one by one
        receiver: async for message in tts_ws → forwards chunks to browser immediately
    - The sender signals an asyncio.Event when all text is sent.
    - The receiver runs until the WS context manager exits (which happens after
      the sender finishes and a short idle window confirms no more chunks arrive).
    - Each chunk is forwarded to the browser THE MOMENT it arrives — true streaming.
    - Frontend receives many small audio_chunk messages and plays them via
      MediaSource API (gapless streaming), so chunk count doesn't matter.
    """

    _SENTINEL = object()

    def __init__(self, browser_ws: WebSocket):
        self._browser_ws = browser_ws
        self._queue: asyncio.Queue = asyncio.Queue()
        self._task: Optional[asyncio.Task] = None

    def start(self):
        self._task = asyncio.create_task(self._run_forever())

    async def speak(
        self,
        sentences: List[str],
        speaker: str,
        lang: str,
        response_id: int,
        done_event: asyncio.Event,
    ):
        """Enqueue a speak request. Returns immediately; done_event is set when audio is fully sent."""
        await self._queue.put((sentences, speaker, lang, response_id, done_event))

    async def close(self):
        await self._queue.put(self._SENTINEL)
        if self._task:
            try:
                await asyncio.wait_for(self._task, timeout=8.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self._task.cancel()

    async def _run_forever(self):
        log("[TTS_STREAM]", "Worker started")
        while True:
            try:
                item = await self._queue.get()
                if item is self._SENTINEL:
                    log("[TTS_STREAM]", "Sentinel — shutting down")
                    return

                sentences, speaker, lang, response_id, done_event = item
                log("[TTS_STREAM]", f"resp_id={response_id} | {len(sentences)} sentence(s) | "
                    f"speaker={speaker} lang={lang}")
                try:
                    await self._do_speak(sentences, speaker, lang, response_id)
                except Exception as e:
                    log("[TTS_STREAM]", f"resp_id={response_id} streaming failed ({e}) — HTTP fallback")
                    await self._fallback_http(sentences, speaker, lang, response_id)
                finally:
                    done_event.set()

            except asyncio.CancelledError:
                log("[TTS_STREAM]", "Worker cancelled")
                return
            except Exception as e:
                log("[TTS_STREAM]", f"Unexpected worker error: {e}")

    async def _do_speak(
        self,
        sentences: List[str],
        speaker: str,
        lang: str,
        response_id: int,
    ):
        """
        TRUE streaming path.
        - Sender task:   configure → per-sentence convert() + flush()
        - Receiver task: async for message in tts_ws → forward chunk to browser immediately
        - Drain logic:   after sender finishes, keep reading until no new chunk arrives
          for IDLE_AFTER_LAST_CHUNK_S seconds (rolling reset on every chunk).
          This is robust regardless of how long synthesis takes.
        """
        chunk_count = 0
        send_done   = asyncio.Event()
        last_chunk_event = asyncio.Event()   # pulsed each time a chunk arrives

        async with client_tts.text_to_speech_streaming.connect(
            model="bulbul:v3"
        ) as tts_ws:
            log("[TTS_STREAM]", f"resp_id={response_id} WS open")

            # ── Sender ────────────────────────────────────────────────────────
            async def sender():
                try:
                    await tts_ws.configure(target_language_code=lang, speaker=speaker)
                    for sentence in sentences:
                        log("[TTS_STREAM]", f"resp_id={response_id} → '{sentence[:60]}'")
                        await tts_ws.convert(sentence)
                        await tts_ws.flush()
                    log("[TTS_STREAM]", f"resp_id={response_id} sender done")
                except Exception as e:
                    log("[TTS_STREAM]", f"resp_id={response_id} sender error: {e}")
                finally:
                    send_done.set()

            # ── Receiver ──────────────────────────────────────────────────────
            async def receiver():
                nonlocal chunk_count
                try:
                    async for message in tts_ws:
                        if not isinstance(message, AudioOutput):
                            continue
                        audio_b64 = message.data.audio
                        chunk_count += 1
                        # log("[TTS_STREAM]", f"resp_id={response_id} chunk#{chunk_count} "
                        #     f"({len(audio_b64)} b64 chars) → browser")
                        try:
                            await self._browser_ws.send_json({
                                "type": "audio_chunk",
                                "index": chunk_count - 1,
                                "total": -1,
                                "audio": audio_b64,
                                "audio_format": "mp3",
                                "is_last": False,
                                "response_id": response_id,
                            })
                        except Exception:
                            log("[TTS_STREAM]", f"resp_id={response_id} browser send failed")
                            return
                        # Signal the drain watcher that a chunk just arrived
                        last_chunk_event.set()
                        last_chunk_event.clear()
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    log("[TTS_STREAM]", f"resp_id={response_id} receiver error: {e}")
                    raise

            sender_task   = asyncio.create_task(sender())
            receiver_task = asyncio.create_task(receiver())

            # ── Drain watcher ─────────────────────────────────────────────────
            # Wait for sender to finish, then wait until no new chunk has arrived
            # for IDLE_S seconds. This window resets on every chunk, so we never
            # cut off mid-stream no matter how slow Bulbul is.
            IDLE_S = 1.2   # seconds of silence after last chunk before we stop
            await send_done.wait()
            log("[TTS_STREAM]", f"resp_id={response_id} sender done — draining (idle={IDLE_S}s)")

            while True:
                try:
                    # Wait for a new chunk to arrive within IDLE_S
                    await asyncio.wait_for(last_chunk_event.wait(), timeout=IDLE_S)
                    # A chunk arrived — reset and wait again
                except asyncio.TimeoutError:
                    # No chunk for IDLE_S seconds after the last one → stream is done
                    log("[TTS_STREAM]", f"resp_id={response_id} idle timeout — stopping receiver")
                    break

            receiver_task.cancel()
            try:
                await receiver_task
            except asyncio.CancelledError:
                pass
            sender_task.cancel()
            try:
                await sender_task
            except asyncio.CancelledError:
                pass

        if chunk_count == 0:
            raise RuntimeError("No audio chunks received from Bulbul")

        log("[TTS_STREAM]", f"resp_id={response_id} ✓ {chunk_count} chunks streamed to browser")
        try:
            await self._browser_ws.send_json({
                "type": "tts_done",
                "response_id": response_id,
                "total_chunks": chunk_count,
            })
        except Exception:
            pass

    async def _fallback_http(
        self,
        sentences: List[str],
        speaker: str,
        lang: str,
        response_id: int,
    ):
        """HTTP fallback — used only if WS fails. Single call for full text."""
        log("[TTS_STREAM]", f"resp_id={response_id} HTTP fallback")
        try:
            full_text = " ".join(sentences)
            audio_b64 = await tts_convert(full_text, speaker, lang)
            await self._browser_ws.send_json({
                "type": "audio_chunk",
                "index": 0, "total": 1,
                "audio": audio_b64,
                "audio_format": "wav",
                "is_last": True,
                "response_id": response_id,
            })
            await self._browser_ws.send_json({
                "type": "tts_done",
                "response_id": response_id,
            })
        except Exception as e:
            log("[TTS_STREAM]", f"resp_id={response_id} HTTP fallback also failed: {e}")

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(min=1, max=5),
    retry=retry_if_exception_type(Exception),
    before_sleep=lambda rs: log("[LLM_RETRY]", f"Attempt {rs.attempt_number} failed…")
)
async def safe_llm_call(messages):
    return await llm_with_tools.ainvoke(messages)

async def extract_memory(user_text: str, memory: dict):
    try:
        prompt = prompts.get_memory_extraction_prompt()

        today = datetime.now().strftime("%Y-%m-%d")

        messages = [
            SystemMessage(content=prompt),
            HumanMessage(content=f"""
                    Today date: {today}

                    Current memory:
                    {json.dumps(memory)}
                    
                    User input:
                    {user_text}
                    """)
        ]

        response = await small_llm.ainvoke(messages)

        extracted = safe_json_parse(response.content)

        return extracted

    except Exception as e:
        print("[MEMORY] Extraction failed:", e)
        return {}

async def run_brain(session_id: str, user_text: str, websocket: WebSocket,
                    tts_session: Optional["StreamingTTSSession"] = None):
    t0    = datetime.now()
    today = t0.strftime("%Y-%m-%d")
    day   = t0.strftime("%A")

    log("[NORMALIZER]", f"Before: {user_text}")
    user_text = normalize_gujarati_time(user_text)
    log("[NORMALIZER]", f"After: {user_text}")

    log("[BRAIN]", f"START | session='{session_id}' | text='{user_text}' | phone='{chat_sessions.get(session_id, {}).get('phone_number')}' | tenant='{chat_sessions.get(session_id, {}).get('tenant_id')}'")

    if user_text == "__UNCLEAR__":
        clarification = "માફ કરશો, મને સ્પષ્ટ સંભળાયું નહીં. શું તમે ફરીથી કહી શકો?"
        log("[BRAIN]", "Noisy input — sending clarification")
        sentences = split_into_sentences(clarification)
        await websocket.send_json({"type": "ai_text", "text": clarification, "chunk_count": len(sentences)})
        await websocket.send_json({"type": "ai_speaking_start"})
        if tts_session:
            done_evt = asyncio.Event()
            await tts_session.speak(sentences, "simran", "gu-IN", 0, done_evt)
            await done_evt.wait()
        else:
            for idx, sentence in enumerate(sentences):
                audio_b64 = await tts_convert(sentence)
                await websocket.send_json({
                    "type": "audio_chunk", "index": idx, "total": len(sentences),
                    "text": sentence, "audio": audio_b64, "is_last": idx == len(sentences) - 1
                })
            await websocket.send_json({"type": "tts_done"})
        return

    if session_id not in chat_sessions:
        chat_sessions[session_id] = {"history": [],
                                     "phone_number": None,
                                     "tenant_id": None,
                                     "bot_config": None,
                                     "memory":{
                                        "intent": None,
                                        "pending_action": "waiting_for_confirmation",
                                        "appointment": {
                                            "date": None,
                                            "time": None,
                                            "duration": None
                                        },
                                        "reschedule": {
                                            "old_time": None,
                                            "new_time": None
                                        }
                                     }}

    session_data = chat_sessions[session_id]
    history      = session_data["history"]
    phone_number = session_data.get("phone_number")
    bot_config   = session_data.get("bot_config") or {}
    tts_speaker  = bot_config.get("tts_speaker", "simran")
    tts_lang     = bot_config.get("language_code", "gu-IN")

    memory = session_data.get("memory", {})
    
    # Extract memory using small LLM
    new_memory = await extract_memory(user_text, memory)
    
    # Merge memory
    updated_memory = merge_memory(memory, new_memory)
    
    print(updated_memory)
    # Save back
    session_data["memory"] = updated_memory

    memory_context = f"\n\n=== MEMORY STATE ===\n{json.dumps(session_data.get('memory', {}))}"

    system_prompt = prompts.get_system_prompt(today, day, bot_config) + memory_context

    if history and isinstance(history[0], SystemMessage):
        history[0] = SystemMessage(content=system_prompt)
    else:
        history.insert(0, SystemMessage(content=system_prompt))

    history.append(HumanMessage(content=user_text))
    recent_history = [history[0]] + history[-MAX_HISTORY:]

    log("[MEMORY]", f"Updated: {updated_memory}")

    t_llm = datetime.now()
    preview = str(recent_history[0].content)[:80] if recent_history else 'NONE'
    log("[LLM]", f"ainvoke() | {len(recent_history)} messages | sys_prompt_preview='{preview}'")
    try:
        ai_msg = await safe_llm_call(recent_history)
        log("[LLM]", f"Done in {(datetime.now()-t_llm).total_seconds():.2f}s | "
            f"tool_calls={len(ai_msg.tool_calls)}")
        print_token_usage(ai_msg, "Initial LLM")
    except Exception as e:
        log("[LLM]", f"FAILED: {e}\n{traceback.format_exc()}")
        raise

    tool_iteration = 0
    while ai_msg.tool_calls:
        tool_iteration += 1
        if tool_iteration > MAX_TOOL_ITERATIONS:
            log("[TOOLS]", f"Max tool iterations ({MAX_TOOL_ITERATIONS}). Forcing AI response.")
            break
        log("[TOOLS]", f"Iteration #{tool_iteration} — {len(ai_msg.tool_calls)} tool(s)")
        history.append(ai_msg)

        async def execute_tool(tool_call):
            tname = tool_call["name"]
            targs = tool_call["args"]
            if phone_number:
                targs = {**targs, "phone_number": phone_number}
            # ── Inject tenant context so calendar_tool uses the correct tenant_id ─
            tenant_context.session_id    = session_id
            tenant_context.chat_sessions = chat_sessions
            calendar_tool.tenant_context = tenant_context
            log("[TOOL]", f"Executing '{tname}' | args={targs} | tenant={get_current_tenant()}")
            await websocket.send_json({"type": "tool_call", "name": tname, "args": targs, "status": "running"})
            t_tool = datetime.now()
            try:
                obs    = await run_tool_async(tname, targs)
                if "success" in str(obs).lower():
                    print("$$$state memory deleted after success$$$")
                    session_data["memory"] = {
                            "intent": "none",
                            "pending_action": "waiting_for_confirmation",
                            "appointment": {
                                "date": None,
                                "time": None,
                                "duration": None
                            },
                            "reschedule": {
                                "old_time": None,
                                "new_time": None
                            },
                            "date_context": {
                                "resolved_date": None,
                                "source": "none"
                            }
                        }

                status = "ok"
                log("[TOOL]", f"'{tname}' OK in {(datetime.now()-t_tool).total_seconds():.2f}s | result='{str(obs)[:100]}'")
            except Exception as e:
                obs, status = f"Error: {e}", "error"
                log("[TOOL]", f"'{tname}' FAILED: {e}")
            await websocket.send_json({"type": "tool_call", "name": tname, "args": targs, "status": status, "result": str(obs)})
            return ToolMessage(content=str(obs), tool_call_id=tool_call["id"])

        tool_results = await asyncio.gather(*[execute_tool(tc) for tc in ai_msg.tool_calls])
        history.extend(tool_results)
        recent_history = [history[0]] + history[-MAX_HISTORY:]
        t_llm2 = datetime.now()
        log("[LLM]", f"Post-tool ainvoke()")
        try:
            ai_msg = await safe_llm_call(recent_history)
            log("[LLM]", f"Done in {(datetime.now()-t_llm2).total_seconds():.2f}s")
            print_token_usage(ai_msg, "Post-tool LLM")
        except Exception as e:
            log("[LLM]", f"Post-tool FAILED: {e}")
            raise

    reply_text = ai_msg.content if isinstance(ai_msg.content, str) else str(ai_msg.content)
    history.append(ai_msg)

    if len(history) > MAX_HISTORY * 2 + 2:
        history[:] = [history[0]] + history[-(MAX_HISTORY * 2):]

    # Fix 4: store last AI response for echo detection
    session_data["last_ai_text"] = reply_text

    # Fix 2: filter tool/JSON output — do NOT speak raw tool artifacts
    if is_tool_output(reply_text):
        log("[TTS_FILTER]", f"Blocked tool output from TTS: '{reply_text[:80]}'")
        log("[BRAIN]", f"DONE (tool-only, no TTS) in {(datetime.now()-t0).total_seconds():.2f}s")
        return

    # Fix 2: strip any stray tool fragments from mixed text before TTS
    tts_text = clean_for_tts(reply_text)
    if not tts_text:
        log("[TTS_FILTER]", "Reply cleaned to empty — skipping TTS")
        log("[BRAIN]", f"DONE (empty after clean) in {(datetime.now()-t0).total_seconds():.2f}s")
        return

    sentences = split_into_sentences(tts_text)
    log("[BRAIN]", f"Reply: '{tts_text[:80]}' | {len(sentences)} sentences")

    await websocket.send_json({"type": "ai_text", "text": reply_text, "chunk_count": len(sentences)})
    await websocket.send_json({"type": "ai_speaking_start"})

    if tts_session:
        # ── Streaming TTS path ────────────────────────────────────────────────
        # We enqueue all sentences at once; the streaming session sends audio
        # chunks to the browser as they arrive from Bulbul — no per-sentence wait.
        done_evt = asyncio.Event()
        import random
        resp_id = random.randint(1, 999999)
        log("[TTS_STREAM]", f"Enqueuing {len(sentences)} sentence(s) | resp_id={resp_id}")
        await tts_session.speak(sentences, tts_speaker, tts_lang, resp_id, done_evt)
        await done_evt.wait()
        log("[TTS_STREAM]", f"resp_id={resp_id} finished")
    else:
        # ── Fallback: sequential HTTP TTS ─────────────────────────────────────
        for idx, sentence in enumerate(sentences):
            log("[TTS]", f"Chunk {idx+1}/{len(sentences)} | speaker={tts_speaker} lang={tts_lang} | '{sentence[:60]}'")
            audio_b64 = await tts_convert(sentence, tts_speaker, tts_lang)
            await websocket.send_json({
                "type": "audio_chunk", "index": idx, "total": len(sentences),
                "text": sentence, "audio": audio_b64, "is_last": idx == len(sentences) - 1
            })
        await websocket.send_json({"type": "tts_done"})

    log("[BRAIN]", f"DONE in {(datetime.now()-t0).total_seconds():.2f}s")


# ── Silence warm-up for Sarvam STT ────────────────────────────────────────────
_SILENCE_FRAME_B64 = base64.b64encode(bytes(4096 * 2)).decode("utf-8")
_WARMUP_FRAMES = 10

async def _warmup_sarvam(sarvam_ws, label: str):
    log("[STT_WARMUP]", f"{label} — priming Sarvam with {_WARMUP_FRAMES} silence frames")
    for i in range(_WARMUP_FRAMES):
        try:
            await sarvam_ws.transcribe(audio=_SILENCE_FRAME_B64)
            await asyncio.sleep(0.04)
        except Exception as e:
            log("[STT_WARMUP]", f"Frame {i+1} failed: {e}")
            return False
    log("[STT_WARMUP]", f"{label} — warm-up done")
    return True


# ── WebSocket voice handler ────────────────────────────────────────────────────
@app.websocket("/ws/voice")
async def voice_ws(websocket: WebSocket, phone_number: Optional[str] = None):
    await websocket.accept()
    session_id = f"web_{datetime.now().strftime('%H%M%S%f')}"
    log("[WS]", f"Connection ACCEPTED | session='{session_id}' | phone='{phone_number}'")

    # Session is pre-created with phone from query param; set_user / init can update it
    chat_sessions[session_id] = {
        "history":      [],
        "phone_number": phone_number,
        "tenant_id":    None,
        "bot_config":   {},
    }

    brain_lock   = asyncio.Lock()
    # Fix 1: Replace asyncio.Event with a float epoch-seconds timestamp.
    # Mic is blocked while time.monotonic() < ai_speaking_until.
    # This lets us add a server-side post-TTS buffer without any async sleep.
    import time as _time
    ai_speaking_until: list = [0.0]   # mutable float in a list so nested funcs can write it
    AI_POST_TTS_BUFFER = 0.90          # seconds of mic-silence after browser sends ai_speaking_end

    def _ai_is_speaking() -> bool:
        return _time.monotonic() < ai_speaking_until[0]

    commit_queue: asyncio.Queue = asyncio.Queue()
    audio_buf:    asyncio.Queue = asyncio.Queue(maxsize=300)

    # ── Streaming TTS session (separate Bulbul v3 WebSocket) ─────────────────
    tts_session = StreamingTTSSession(websocket)
    tts_session.start()
    log("[WS]", "StreamingTTSSession started")

    _recv = _drop_ai = _sent = 0

    # ── Task 1: Single browser receiver — handles both binary PCM and JSON ctrl ─
    # IMPORTANT: Only ONE coroutine may call websocket.receive() at a time.
    # The old split into browser_listener + raw_audio_listener broke this rule.
    async def browser_to_sarvam():
        nonlocal _recv, _drop_ai, _sent, phone_number
        log("[AUDIO_TASK]", "browser_to_sarvam() started")
        try:
            while True:
                try:
                    msg = await websocket.receive()
                except (WebSocketDisconnect, RuntimeError) as e:
                    log("[AUDIO_TASK]", f"Browser disconnected: {e}")
                    break

                # ── JSON control messages ─────────────────────────────────────
                if "text" in msg:
                    try:
                        ctrl      = json.loads(msg["text"])
                        ctrl_type = ctrl.get("type", "unknown")

                        if ctrl_type == "init":
                            # New customer UI sends 'init' with session_id, phone, tenant
                            new_phone   = ctrl.get("phone_number")
                            tenant_id   = ctrl.get("tenant_id")
                            new_sess_id = ctrl.get("session_id", session_id)
                            if new_phone:
                                phone_number = new_phone
                                chat_sessions[session_id]["phone_number"] = new_phone
                            if tenant_id:
                                chat_sessions[session_id]["tenant_id"] = tenant_id
                                # Load tenant bot config so the brain uses the right persona
                                if DB_AVAILABLE:
                                    try:
                                        bot_cfg = await asyncio.to_thread(get_bot_config, tenant_id)
                                        chat_sessions[session_id]["bot_config"] = bot_cfg or {}
                                        log("[WS]", f"Bot config loaded for tenant={tenant_id}")
                                        
                                        # Trigger greeting if configured
                                        if bot_cfg and bot_cfg.get("greeting_message"):
                                            greeting_text = bot_cfg["greeting_message"]
                                            tts_speaker = bot_cfg.get("tts_speaker", "simran")
                                            tts_lang = bot_cfg.get("language_code", "gu-IN")
                                            
                                            async def play_initial_greeting():
                                                try:
                                                    log("[GREETING]", f"Preparing initial greeting: {greeting_text[:30]}...")
                                                    sentences = split_into_sentences(greeting_text)
                                                    
                                                    # Lock the brain during greeting
                                                    async with brain_lock:
                                                        await websocket.send_json({
                                                            "type": "ai_text",
                                                            "text": greeting_text,
                                                            "chunk_count": len(sentences)
                                                        })
                                                        await websocket.send_json({"type": "ai_speaking_start"})
                                                        
                                                        done_evt = asyncio.Event()
                                                        import random as _rand
                                                        resp_id = _rand.randint(1, 999999)
                                                        await tts_session.speak(
                                                            sentences, tts_speaker, tts_lang,
                                                            resp_id, done_evt
                                                        )
                                                        await done_evt.wait()
                                                        log("[GREETING]", "Greeting playback finished")
                                                except Exception as e:
                                                    log("[GREETING]", f"Error playing greeting: {e}")
                                            
                                            asyncio.create_task(play_initial_greeting())
                                            
                                    except Exception as cfg_err:
                                        log("[WS]", f"Bot config load failed: {cfg_err}")
                            log("[WS]", f"init | session={session_id} phone={phone_number} tenant={tenant_id}")

                        elif ctrl_type == "set_user":
                            # Old-style message — still supported
                            phone = ctrl.get("phone_number")
                            if phone and session_id in chat_sessions:
                                phone_number = phone
                                chat_sessions[session_id]["phone_number"] = phone
                                log("[WS]", f"set_user phone={phone}")

                        elif ctrl_type == "ai_speaking_start":
                            # Fix 1: set timestamp far ahead (30s max guard)
                            ai_speaking_until[0] = _time.monotonic() + 30.0
                            log("[CTRL]", "ai_speaking_start → mic MUTED")

                        elif ctrl_type == "ai_speaking_end":
                            # Fix 1: add post-TTS buffer — mic stays blocked for AI_POST_TTS_BUFFER
                            # seconds after browser fires onended, preventing feedback from audio tail
                            ai_speaking_until[0] = _time.monotonic() + AI_POST_TTS_BUFFER
                            log("[CTRL]", f"ai_speaking_end → mic blocked until +{AI_POST_TTS_BUFFER}s | recv={_recv} sent={_sent} drop_ai={_drop_ai}")

                        elif ctrl_type == "commit_transcript":
                            text = ctrl.get("text", "").strip()
                            speaking_now = _ai_is_speaking()
                            log("[COMMIT]", f"Received text='{text}' | ai_speaking={speaking_now} | queue_size={commit_queue.qsize()}")
                            if not text:
                                log("[COMMIT]", "IGNORED — empty text")
                            elif speaking_now:
                                log("[COMMIT]", f"DROPPED (AI speaking/buffer): '{text}'")
                            elif is_noisy_transcript(text):
                                log("[COMMIT]", f"DROPPED (noisy): '{text}' → sending __UNCLEAR__")
                                await commit_queue.put("__UNCLEAR__")
                            else:
                                # Fix 4: echo detection — drop if too similar to last AI response
                                last_ai = chat_sessions.get(session_id, {}).get("last_ai_text", "")
                                if is_echo_of_ai(text, last_ai):
                                    log("[COMMIT]", f"DROPPED (echo of AI): '{text}'")
                                else:
                                    log("[COMMIT]", f"QUEUED: '{text}'")
                                    await commit_queue.put(text)
                        else:
                            log("[CTRL]", f"Unknown ctrl type='{ctrl_type}'")
                    except Exception as e:
                        log("[CTRL]", f"JSON parse error: {e}")
                    continue

                # ── Raw PCM audio bytes ───────────────────────────────────────
                if "bytes" in msg:
                    _recv += 1
                    raw = msg["bytes"]
                    # Fix 1: gate on timestamp instead of asyncio.Event
                    if _ai_is_speaking():
                        _drop_ai += 1
                        continue
                    try:
                        audio_buf.put_nowait(raw)
                    except asyncio.QueueFull:
                        try:
                            audio_buf.get_nowait()
                        except asyncio.QueueEmpty:
                            pass
                        audio_buf.put_nowait(raw)

        except Exception as e:
            log("[AUDIO_TASK]", f"Crashed: {e}\n{traceback.format_exc()}")

    # ── Task 2: audio_buf → Sarvam STT (with automatic reconnect) ────────────
    async def sarvam_sender():
        nonlocal _sent
        reconnect_num = 0
        MAX_RECONNECTS = 15

        while reconnect_num <= MAX_RECONNECTS:
            attempt_label = f"conn#{reconnect_num + 1}"
            log("[STT_SEND]", f"Opening Sarvam STT connection ({attempt_label})")
            try:
                async with client_stt.speech_to_text_streaming.connect(
                    model="saaras:v3", language_code="gu-IN", sample_rate=16000
                ) as sarvam_ws:
                    log("[STT_SEND]", f"Sarvam STT ESTABLISHED ({attempt_label})")

                    ok = await _warmup_sarvam(sarvam_ws, attempt_label)
                    if not ok:
                        log("[STT_SEND]", "Warm-up failed — will reconnect")
                        reconnect_num += 1
                        await asyncio.sleep(1)
                        continue

                    try:
                        await websocket.send_json({"type": "stt_ready", "reconnect_num": reconnect_num})
                    except Exception:
                        pass

                    recv_task = asyncio.create_task(
                        _sarvam_receiver(sarvam_ws, commit_queue, websocket)
                    )

                    try:
                        KEEPALIVE_INTERVAL = 20.0
                        POLL_INTERVAL      = 0.1
                        last_real_pkt_time = asyncio.get_event_loop().time()

                        while True:
                            try:
                                raw = audio_buf.get_nowait()
                            except asyncio.QueueEmpty:
                                now = asyncio.get_event_loop().time()
                                idle_s = now - last_real_pkt_time
                                if idle_s >= KEEPALIVE_INTERVAL and not _ai_is_speaking():
                                    log("[STT_SEND]", f"Keep-alive after {idle_s:.0f}s idle")
                                    await sarvam_ws.transcribe(audio=_SILENCE_FRAME_B64)
                                    last_real_pkt_time = now
                                await asyncio.sleep(POLL_INTERVAL)
                                continue

                            last_real_pkt_time = asyncio.get_event_loop().time()
                            await sarvam_ws.transcribe(
                                audio=base64.b64encode(raw).decode("utf-8")
                            )
                            _sent += 1
                            if _sent % 50 == 1:
                                rms = compute_rms(raw)
                                log("[STT_SEND]", f"pkt#{_sent} | rms={rms:.0f} "
                                    f"({'speech' if rms > 300 else 'silence'})")

                    except Exception as e:
                        log("[STT_SEND]", f"Send loop ended ({attempt_label}): {e}")
                    finally:
                        recv_task.cancel()
                        try:
                            await recv_task
                        except asyncio.CancelledError:
                            pass

            except Exception as e:
                log("[STT_SEND]", f"Sarvam connect FAILED ({attempt_label}): {e}")

            reconnect_num += 1
            if reconnect_num <= MAX_RECONNECTS:
                wait_s = min(2 ** reconnect_num, 16)
                log("[STT_SEND]", f"Reconnecting in {wait_s}s (attempt {reconnect_num + 1}/{MAX_RECONNECTS})")
                try:
                    await websocket.send_json({
                        "type": "stt_reconnecting",
                        "attempt": reconnect_num,
                        "wait_s": wait_s
                    })
                except Exception:
                    pass
                await asyncio.sleep(wait_s)
            else:
                log("[STT_SEND]", "Max reconnects reached — giving up")
                try:
                    await websocket.send_json({"type": "stt_failed"})
                except Exception:
                    pass

    # ── Task 2b: Sarvam transcript receiver (one per Sarvam connection) ───────
    async def _sarvam_receiver(sarvam_ws, commit_queue, websocket):
        log("[STT_RECV]", "Receiver started")
        try:
            p_count = f_count = 0
            async for response in sarvam_ws:
                if not (hasattr(response, "type") and response.type == "data"):
                    continue
                transcript = getattr(response.data, 'transcript', "").strip()
                is_final   = getattr(response.data, 'is_final', False)
                if not transcript:
                    continue

                speaking_now = _ai_is_speaking()
                speaking_tag = "AI_SPEAKING" if speaking_now else "user_turn"
                log("[DIAG:STT]", f"{'FINAL' if is_final else 'partial'} [{speaking_tag}]: '{transcript}' | noisy={is_noisy_transcript(transcript)}")

                if is_final:
                    f_count += 1
                    log("[STT_RECV]", f"FINAL #{f_count}: '{transcript}'")
                    if speaking_now:
                        # Fix 1: blocked by timestamp guard (includes post-TTS buffer)
                        log("[STT_RECV]", f"FINAL DROPPED (AI speaking/buffer): '{transcript}'")
                    elif is_noisy_transcript(transcript):
                        log("[STT_RECV]", f"FINAL DROPPED (noisy): '{transcript}'")
                        await commit_queue.put("__UNCLEAR__")
                    else:
                        # Fix 4: echo detection — secondary guard against mic capturing AI voice tail
                        last_ai = chat_sessions.get(session_id, {}).get("last_ai_text", "")
                        if is_echo_of_ai(transcript, last_ai):
                            log("[STT_RECV]", f"FINAL DROPPED (echo of AI): '{transcript}'")
                        else:
                            await commit_queue.put(transcript)
                else:
                    p_count += 1
                    if p_count % 5 == 1:
                        log("[STT_RECV]", f"Partial #{p_count}: '{transcript}'")

                try:
                    await websocket.send_json({
                        "type": "transcript",
                        "text": transcript,
                        "is_final": is_final
                    })
                except Exception:
                    pass

        except asyncio.CancelledError:
            log("[STT_RECV]", "Receiver cancelled (Sarvam reconnect in progress)")
            raise
        except Exception as e:
            log("[STT_RECV]", f"Receiver ended: {e}")

    async def brain_consumer():
        log("[BRAIN_CONSUMER]", "Started — waiting for commits")
        while True:
            try:
                sentence = await commit_queue.get()
                log("[BRAIN_CONSUMER]", f"Dequeued from queue: '{sentence}' | brain_locked={brain_lock.locked()}")
                if brain_lock.locked():
                    log("[BRAIN_CONSUMER]", "Brain BUSY — dropping sentence")
                    continue
                async with brain_lock:
                    log("[BRAIN_CONSUMER]", "Lock ACQUIRED")
                    await websocket.send_json({"type": "processing_start"})
                    try:
                        await run_brain(session_id, sentence, websocket, tts_session)
                    except Exception as e:
                        log("[BRAIN_CONSUMER]", f"run_brain() FAILED: {e}")
                        fallback_text = _get_fallback_message(e)
                        log("[BRAIN_CONSUMER]", f"Sending fallback TTS: '{fallback_text}'")
                        try:
                            await websocket.send_json({
                                "type": "ai_text", "text": fallback_text, "chunk_count": 1
                            })
                            await websocket.send_json({"type": "ai_speaking_start"})
                            # Use streaming TTS for fallback too
                            done_evt = asyncio.Event()
                            import random as _rand
                            resp_id = _rand.randint(1, 999999)
                            await tts_session.speak(
                                [fallback_text], "simran", "gu-IN", resp_id, done_evt
                            )
                            await done_evt.wait()
                        except Exception as tts_err:
                            log("[BRAIN_CONSUMER]", f"Fallback TTS also failed: {tts_err}")
                        continue
                    log("[BRAIN_CONSUMER]", "Lock RELEASED")
            except asyncio.CancelledError:
                log("[BRAIN_CONSUMER]", "Cancelled — exiting")
                break
            except Exception as e:
                log("[BRAIN_CONSUMER]", f"Error: {e}\n{traceback.format_exc()}")

    log("[WS]", "Launching tasks")
    try:
        await asyncio.gather(
            browser_to_sarvam(),
            sarvam_sender(),
            brain_consumer(),
        )
    except Exception as e:
        log("[WS]", f"Handler CRASHED: {e}\n{traceback.format_exc()}")
    finally:
        log("[WS]", "Closing StreamingTTSSession")
        await tts_session.close()
    log("[WS]", "All tasks exited")



def compute_rms(raw_bytes: bytes) -> float:
    if len(raw_bytes) < 2:
        return 0.0
    samples = struct.unpack(f"<{len(raw_bytes)//2}h", raw_bytes)
    return (sum(s * s for s in samples) / len(samples)) ** 0.5

def merge_memory(old, new):
    for key, value in new.items():
        if isinstance(value, dict):
            old[key] = merge_memory(old.get(key, {}), value)
        else:
            if value is not None:
                old[key] = value
    return old

def safe_json_parse(text):
    try:
        return json.loads(text)
    except:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise