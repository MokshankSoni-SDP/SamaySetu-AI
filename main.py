import os
import re
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
from sarvamai import AsyncSarvamAI
from dotenv import load_dotenv
from typing import Dict, List, Optional
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

import prompts
from calendar_tool import *
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from datetime import datetime, date

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


load_dotenv()


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
MAX_TOOL_ITERATIONS = 4

# Gujarati number words → digits
GUJ_NUMBERS = {
    "એક": 1, "બે": 2, "ત્રણ": 3, "ચાર": 4,
    "પાંચ": 5, "છ": 6, "સાત": 7, "આઠ": 8,
    "નવ": 9, "દસ": 10, "અગિયાર": 11, "બાર": 12
}
SPECIAL_PHRASES = {"દોઢ": (1, 30), "અઢી": (2, 30)}
COMMON_STT_VARIANTS = {
    "સ્વા": "સવા", "સવા": "સવા",
    "સાડા": "સાડા", "પોના": "પોણા", "પોણા": "પોણા"
}

import re as _re

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

def normalize_gujarati_time(text: str) -> str:
    original_text = text
    text = text.lower().strip()
    for phrase, (h, m) in SPECIAL_PHRASES.items():
        if phrase in text:
            text = text.replace(phrase, f"{h}:{m:02d}")
    text = normalize_variants(text)
    for prefix, minutes, offset in [("સવા", 15, 0), ("સાડા", 30, 0), ("પોણા", 45, -1)]:
        match = re.search(rf"{prefix}\s*(\w+)", text)
        if match:
            hour_word = match.group(1)
            hour = GUJ_NUMBERS.get(hour_word)
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
async def admin_connect_calendar(req: CalendarConnectRequest,
                                  session=Depends(_check_admin_token)):
    """
    Store Google Calendar credentials for this tenant.
    The frontend submits a service_account JSON string + the calendar ID.
    The backend stores it and uses it for subsequent calendar operations.
    """
    tenant_id = session["tenant_id"]
    if not DB_AVAILABLE:
        raise HTTPException(status_code=503, detail="Database unavailable")
    # Validate the JSON is parseable
    try:
        json.loads(req.service_account_json)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid service account JSON")
    token = await asyncio.to_thread(
        save_calendar_token, tenant_id, req.calendar_id, req.service_account_json
    )
    # Also update bot_config with calendar_id for quick access
    await asyncio.to_thread(upsert_bot_config, tenant_id, calendar_id=req.calendar_id)
    return {"status": "connected", "calendar_id": req.calendar_id}


@app.get("/admin/calendar/status")
async def admin_calendar_status(session=Depends(_check_admin_token)):
    tenant_id = session["tenant_id"]
    if not DB_AVAILABLE:
        return {"connected": False}
    token = await asyncio.to_thread(get_calendar_token, tenant_id)
    if token:
        return {"connected": True, "calendar_id": token["calendar_id"],
                "connected_at": token["connected_at"]}
    return {"connected": False}


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

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=4),
    retry=retry_if_exception_type(Exception),
    before_sleep=lambda rs: log("[TTS_RETRY]", f"Attempt {rs.attempt_number} failed…")
)
async def tts_convert(text: str, speaker: str = "simran", lang: str = "gu-IN") -> str:
    res = await client_tts.text_to_speech.convert(
        text=text,
        target_language_code=lang,
        model="bulbul:v3",
        speaker=speaker
    )
    return res.audios[0]

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(min=1, max=5),
    retry=retry_if_exception_type(Exception),
    before_sleep=lambda rs: log("[LLM_RETRY]", f"Attempt {rs.attempt_number} failed…")
)
async def safe_llm_call(messages):
    return await llm_with_tools.ainvoke(messages)

async def run_brain(session_id: str, user_text: str, websocket: WebSocket):
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
        for idx, sentence in enumerate(sentences):
            audio_b64 = await tts_convert(sentence)
            await websocket.send_json({
                "type": "audio_chunk", "index": idx, "total": len(sentences),
                "text": sentence, "audio": audio_b64, "is_last": idx == len(sentences) - 1
            })
        await websocket.send_json({"type": "tts_done"})
        return

    if session_id not in chat_sessions:
        chat_sessions[session_id] = {"history": [], "phone_number": None,
                                      "tenant_id": None, "bot_config": None}

    session_data = chat_sessions[session_id]
    history      = session_data["history"]
    phone_number = session_data.get("phone_number")
    bot_config   = session_data.get("bot_config") or {}
    tts_speaker  = bot_config.get("tts_speaker", "simran")
    tts_lang     = bot_config.get("language_code", "gu-IN")

    if not history or not isinstance(history[0], SystemMessage):
        log("[BRAIN]", f"Injecting system prompt | today={today} day={day}")
        history.insert(0, SystemMessage(content=prompts.get_system_prompt(today, day, bot_config)))

    history.append(HumanMessage(content=user_text))
    recent_history = [history[0]] + history[-MAX_HISTORY:]

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
            log("[TOOL]", f"Executing '{tname}' | args={targs}")
            await websocket.send_json({"type": "tool_call", "name": tname, "args": targs, "status": "running"})
            t_tool = datetime.now()
            try:
                obs    = await run_tool_async(tname, targs)
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

    sentences = split_into_sentences(reply_text)
    log("[BRAIN]", f"Reply: '{reply_text[:80]}' | {len(sentences)} sentences")

    await websocket.send_json({"type": "ai_text", "text": reply_text, "chunk_count": len(sentences)})
    await websocket.send_json({"type": "ai_speaking_start"})

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
    ai_speaking  = asyncio.Event()
    commit_queue: asyncio.Queue = asyncio.Queue()
    audio_buf:    asyncio.Queue = asyncio.Queue(maxsize=300)

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
                            ai_speaking.set()
                            log("[CTRL]", "ai_speaking_start → mic MUTED")

                        elif ctrl_type == "ai_speaking_end":
                            ai_speaking.clear()
                            log("[CTRL]", f"ai_speaking_end → mic UN-MUTED | recv={_recv} sent={_sent} drop_ai={_drop_ai}")

                        elif ctrl_type == "commit_transcript":
                            text = ctrl.get("text", "").strip()
                            log("[COMMIT]", f"Received text='{text}' | ai_speaking={ai_speaking.is_set()} | queue_size={commit_queue.qsize()}")
                            if not text:
                                log("[COMMIT]", "IGNORED — empty text")
                            elif ai_speaking.is_set():
                                log("[COMMIT]", f"DROPPED (AI speaking): '{text}'")
                            elif is_noisy_transcript(text):
                                log("[COMMIT]", f"DROPPED (noisy): '{text}' → sending __UNCLEAR__")
                                await commit_queue.put("__UNCLEAR__")
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
                    if ai_speaking.is_set():
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
                        _sarvam_receiver(sarvam_ws, commit_queue, websocket, ai_speaking)
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
                                if idle_s >= KEEPALIVE_INTERVAL and not ai_speaking.is_set():
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
    async def _sarvam_receiver(sarvam_ws, commit_queue, websocket, ai_speaking):
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

                speaking_tag = "AI_SPEAKING" if ai_speaking.is_set() else "user_turn"
                log("[DIAG:STT]", f"{'FINAL' if is_final else 'partial'} [{speaking_tag}]: '{transcript}' | noisy={is_noisy_transcript(transcript)}")

                if is_final:
                    f_count += 1
                    log("[STT_RECV]", f"FINAL #{f_count}: '{transcript}'")
                    if ai_speaking.is_set():
                        log("[STT_RECV]", f"FINAL DROPPED (AI speaking): '{transcript}'")
                    elif is_noisy_transcript(transcript):
                        log("[STT_RECV]", f"FINAL DROPPED (noisy): '{transcript}'")
                        await commit_queue.put("__UNCLEAR__")
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

    # ── Task 3: Brain consumer ────────────────────────────────────────────────
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
                        await run_brain(session_id, sentence, websocket)
                    except Exception as e:
                        log("[BRAIN_CONSUMER]", f"run_brain() FAILED: {e}")
                        fallback_text = _get_fallback_message(e)
                        log("[BRAIN_CONSUMER]", f"Sending fallback TTS: '{fallback_text}'")
                        try:
                            await websocket.send_json({
                                "type": "ai_text", "text": fallback_text, "chunk_count": 1
                            })
                            await websocket.send_json({"type": "ai_speaking_start"})
                            audio_b64 = await tts_convert(fallback_text)
                            await websocket.send_json({
                                "type": "audio_chunk", "index": 0, "total": 1,
                                "text": fallback_text, "audio": audio_b64, "is_last": True
                            })
                            await websocket.send_json({"type": "tts_done"})
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
    log("[WS]", "All tasks exited")


def compute_rms(raw_bytes: bytes) -> float:
    if len(raw_bytes) < 2:
        return 0.0
    samples = struct.unpack(f"<{len(raw_bytes)//2}h", raw_bytes)
    return (sum(s * s for s in samples) / len(samples)) ** 0.5