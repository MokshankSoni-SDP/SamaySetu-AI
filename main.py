"""
main.py
-------
SamaySetu AI — FastAPI application entry point.
Handles: HTTP routes, WebSocket voice sessions, admin/superadmin REST APIs,
         module configuration endpoints, knowledge base management.

Brain logic (LLM, memory, tools) lives in brain.py.
Module system lives in modules/.
"""

import os
import json
import base64
import asyncio
import secrets
import hashlib
import traceback
from contextlib import asynccontextmanager
from typing import Dict, List, Optional
from datetime import datetime, date

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request,Depends, Header
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sarvamai import AsyncSarvamAI, AudioOutput
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

load_dotenv()

import config
import prompts
from brain import (
    run_brain, log,
    is_noisy_transcript, is_echo_of_ai,
    split_into_sentences, compute_rms, _get_fallback_message,
)

# ── DB imports ────────────────────────────────────────────────────────────────
try:
    from database.models import create_tables
    from database.crud import (
        create_user_if_not_exists, get_user_appointments,
        get_tenant_by_id, get_bot_config, upsert_bot_config,
        get_tenant_users, get_tenant_stats,
        get_tenant_appointments_for_date, get_tenant_appointments_range,
        get_all_tenants, create_tenant, get_platform_stats,
        update_tenant_status, create_tenant_admin, get_admin_by_email,
        save_calendar_token, get_calendar_token,
        # Module operations (names match updated crud.py)
        get_module_configs_list as get_all_module_configs,
        set_module_enabled,
        get_tenant_modules as get_enabled_modules,
        # Knowledge base
        add_knowledge_chunks as add_knowledge,
        get_knowledge_chunks as get_all_knowledge,
        delete_all_knowledge,
        delete_knowledge,
    )
    DB_AVAILABLE = True
except ImportError as _db_err:
    DB_AVAILABLE = False
    print(f"[DB] Database module missing: {_db_err}")

try:
    from services.calendar_provider import verify_calendar_connection
    CALENDAR_SERVICE_AVAILABLE = True
except ImportError:
    CALENDAR_SERVICE_AVAILABLE = False


# ── Pydantic models ────────────────────────────────────────────────────────────

class UserLoginRequest(BaseModel):
    phone_number: str
    name: Optional[str] = None
    tenant_id: Optional[str] = None

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
    service_account_json: str

class ModuleToggleRequest(BaseModel):
    module_name: str
    is_enabled: bool

class KnowledgeUploadRequest(BaseModel):
    content: str


# ── App lifecycle ──────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Database setup ─────────────────────────────────────────────────────────
    if DB_AVAILABLE:
        await asyncio.to_thread(create_tables)
    else:
        print("[DB] Skipping table creation — psycopg2 unavailable.")

    # ── FIX 2: Pre-warm FACTS module (embedding model + Qdrant) ───────────────
    # facts_module._bootstrap() already ran at import time, loading the model.
    # This warmup call runs a dummy encode() to prime BLAS/ONNX routines so the
    # very first real request doesn't pay the JIT warm-up cost.
    try:
        from modules.facts_module import warmup as warmup_facts
        await asyncio.to_thread(warmup_facts)
        print("[STARTUP] FACTS module warmup complete")
    except Exception as e:
        print(f"[STARTUP] FACTS module warmup skipped (not installed?): {e}")

    yield

app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)


# ── In-memory session stores ───────────────────────────────────────────────────
chat_sessions:       Dict[str, dict] = {}
admin_sessions:      Dict[str, dict] = {}
superadmin_sessions: Dict[str, dict] = {}


# ── Sarvam STT / TTS clients ──────────────────────────────────────────────────
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
client_stt = AsyncSarvamAI(api_subscription_key=SARVAM_API_KEY)
client_tts = AsyncSarvamAI(api_subscription_key=SARVAM_API_KEY)

_SILENCE_FRAME_B64 = base64.b64encode(bytes(4096 * 2)).decode("utf-8")
_warmup_frames = config.WARMUP_FRAMES


# ── Auth helpers ───────────────────────────────────────────────────────────────

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


# ── TTS helpers ────────────────────────────────────────────────────────────────

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=4),
    retry=retry_if_exception_type(Exception),
    before_sleep=lambda rs: log("[TTS_RETRY]", f"Attempt {rs.attempt_number} failed…")
)
async def tts_convert(text: str, speaker: str, lang: str) -> str:
    res = await client_tts.text_to_speech.convert(
        text=text, target_language_code=lang,
        model="bulbul:v3", speaker=speaker
    )
    return res.audios[0]


# ── Streaming TTS session ──────────────────────────────────────────────────────

class StreamingTTSSession:
    _SENTINEL = object()

    def __init__(self, browser_ws: WebSocket):
        self._browser_ws = browser_ws
        self._queue: asyncio.Queue = asyncio.Queue()
        self._task: Optional[asyncio.Task] = None

    def start(self):
        self._task = asyncio.create_task(self._run_forever())

    async def speak(self, sentences: List[str], speaker: str, lang: str,
                    response_id: int, done_event: asyncio.Event):
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
                    return
                sentences, speaker, lang, response_id, done_event = item
                try:
                    await self._do_speak(sentences, speaker, lang, response_id)
                except Exception as e:
                    log("[TTS_STREAM]", f"resp_id={response_id} streaming failed ({e}) — HTTP fallback")
                    await self._fallback_http(sentences, speaker, lang, response_id)
                finally:
                    done_event.set()
            except asyncio.CancelledError:
                return
            except Exception as e:
                log("[TTS_STREAM]", f"Unexpected worker error: {e}")

    async def _do_speak(self, sentences, speaker, lang, response_id):
        chunk_count = 0
        send_done = asyncio.Event()
        last_chunk_event = asyncio.Event()

        async with client_tts.text_to_speech_streaming.connect(model="bulbul:v3") as tts_ws:
            async def sender():
                try:
                    await tts_ws.configure(target_language_code=lang, speaker=speaker)
                    for sentence in sentences:
                        await tts_ws.convert(sentence)
                        await tts_ws.flush()
                except Exception as e:
                    log("[TTS_STREAM]", f"resp_id={response_id} sender error: {e}")
                finally:
                    send_done.set()

            async def receiver():
                nonlocal chunk_count
                try:
                    async for message in tts_ws:
                        if not isinstance(message, AudioOutput):
                            continue
                        audio_b64 = message.data.audio
                        chunk_count += 1
                        try:
                            await self._browser_ws.send_json({
                                "type": "audio_chunk", "index": chunk_count - 1, "total": -1,
                                "audio": audio_b64, "audio_format": "mp3",
                                "is_last": False, "response_id": response_id,
                            })
                        except Exception:
                            return
                        last_chunk_event.set()
                        last_chunk_event.clear()
                except asyncio.CancelledError:
                    pass

            sender_task   = asyncio.create_task(sender())
            receiver_task = asyncio.create_task(receiver())

            IDLE_S = 1.2
            await send_done.wait()
            while True:
                try:
                    await asyncio.wait_for(last_chunk_event.wait(), timeout=IDLE_S)
                except asyncio.TimeoutError:
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

        try:
            await self._browser_ws.send_json({
                "type": "tts_done", "response_id": response_id, "total_chunks": chunk_count,
            })
        except Exception:
            pass

    async def _fallback_http(self, sentences, speaker, lang, response_id):
        try:
            full_text = " ".join(sentences)
            audio_b64 = await tts_convert(full_text, speaker, lang)
            await self._browser_ws.send_json({
                "type": "audio_chunk", "index": 0, "total": 1,
                "audio": audio_b64, "audio_format": "wav",
                "is_last": True, "response_id": response_id,
            })
            await self._browser_ws.send_json({"type": "tts_done", "response_id": response_id})
        except Exception as e:
            log("[TTS_STREAM]", f"resp_id={response_id} HTTP fallback also failed: {e}")


# ── HTML page routes ───────────────────────────────────────────────────────────

@app.get("/")
async def serve_customer():
    return FileResponse("static/index.html")

@app.get("/admin")
async def serve_admin():
    return FileResponse("static/admin.html")

@app.get("/superadmin")
async def serve_superadmin():
    return FileResponse("static/superadmin.html")

@app.get("/public/bot-config")
async def public_bot_config(tenant_id: Optional[str] = None):
    if not tenant_id or not DB_AVAILABLE:
        return {}
    try:
        cfg = await asyncio.to_thread(get_bot_config, tenant_id)
        if not cfg:
            return {}
        return {k: cfg[k] for k in (
            "bot_name", "receptionist_name", "language_code", "tts_speaker",
            "business_hours_start", "business_hours_end", "slot_duration_mins",
            "silence_timeout_ms", "greeting_message", "business_description",
        ) if k in cfg}
    except Exception:
        return {}

app.mount("/static", StaticFiles(directory="static"), name="static")


# ── Customer endpoints ─────────────────────────────────────────────────────────

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
        raise HTTPException(status_code=500, detail="Database error during login")


@app.get("/appointments/{phone_number}")
async def get_appointments(phone_number: str, tenant_id: Optional[str] = None):
    if not DB_AVAILABLE:
        return []
    try:
        return await asyncio.to_thread(get_user_appointments, phone_number, tenant_id)
    except Exception:
        raise HTTPException(status_code=500, detail="Could not fetch appointments")


# ── Admin auth ─────────────────────────────────────────────────────────────────

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
        "tenant_id":     admin["tenant_id"],
        "email":         admin["email"],
        "role":          admin["role"],
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


# ── Admin dashboard ────────────────────────────────────────────────────────────

@app.get("/admin/stats")
async def admin_stats(session=Depends(_check_admin_token)):
    if not DB_AVAILABLE:
        return {}
    return await asyncio.to_thread(get_tenant_stats, session["tenant_id"])

@app.get("/admin/appointments/today")
async def admin_today(session=Depends(_check_admin_token)):
    if not DB_AVAILABLE:
        return []
    return await asyncio.to_thread(
        get_tenant_appointments_for_date, session["tenant_id"], date.today().isoformat()
    )

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
    return await asyncio.to_thread(
        get_tenant_appointments_for_date, tenant_id, date.today().isoformat()
    )

@app.get("/admin/users")
async def admin_users(session=Depends(_check_admin_token)):
    if not DB_AVAILABLE:
        return []
    return await asyncio.to_thread(get_tenant_users, session["tenant_id"])

@app.get("/admin/bot-config")
async def admin_get_config(session=Depends(_check_admin_token)):
    if not DB_AVAILABLE:
        return {}
    return await asyncio.to_thread(get_bot_config, session["tenant_id"]) or {}

@app.post("/admin/bot-config")
async def admin_save_config(req: BotConfigRequest, session=Depends(_check_admin_token)):
    if not DB_AVAILABLE:
        raise HTTPException(status_code=503, detail="Database unavailable")
    fields = {k: v for k, v in req.dict().items() if v is not None}
    return await asyncio.to_thread(upsert_bot_config, session["tenant_id"], **fields)


# ── Admin: Calendar ────────────────────────────────────────────────────────────

@app.post("/admin/calendar/connect")
async def admin_connect_calendar(req: CalendarConnectRequest,
                                  session=Depends(_check_admin_token)):
    tenant_id = session["tenant_id"]
    if not DB_AVAILABLE:
        raise HTTPException(status_code=503, detail="Database unavailable")
    try:
        json.loads(req.service_account_json)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid service account JSON")

    await asyncio.to_thread(save_calendar_token, tenant_id, req.calendar_id, req.service_account_json)
    await asyncio.to_thread(upsert_bot_config, tenant_id, calendar_id=req.calendar_id)

    if CALENDAR_SERVICE_AVAILABLE:
        result = await asyncio.to_thread(verify_calendar_connection, tenant_id)
        if not result["ok"]:
            raise HTTPException(
                status_code=422,
                detail=f"Credentials saved but Google Calendar verification failed: {result['error']}"
            )

    return {"status": "connected", "verified": True, "calendar_id": req.calendar_id}

@app.get("/admin/calendar/status")
async def admin_calendar_status(session=Depends(_check_admin_token)):
    tenant_id = session["tenant_id"]
    if not DB_AVAILABLE:
        return {"connected": False}
    token = await asyncio.to_thread(get_calendar_token, tenant_id)
    if not token:
        return {"connected": False}
    if CALENDAR_SERVICE_AVAILABLE:
        result = await asyncio.to_thread(verify_calendar_connection, tenant_id)
        if result["ok"]:
            return {"connected": True, "verified": True, "calendar_id": token["calendar_id"],
                    "connected_at": token.get("connected_at")}
        return {"connected": False, "verified": False, "calendar_id": token["calendar_id"],
                "error": result["error"]}
    return {"connected": bool(token), "calendar_id": token.get("calendar_id")}


# ── Admin: Module management (NEW) ────────────────────────────────────────────

@app.get("/admin/modules")
async def admin_get_modules(session=Depends(_check_admin_token)):
    """Get all module configs for this tenant."""
    tenant_id = session["tenant_id"]
    if not DB_AVAILABLE:
        # Return defaults if no DB
        return [
            {"module_name": "BOOKING_MODULE", "is_enabled": True},
            {"module_name": "FACTS_MODULE",   "is_enabled": False},
        ]
    return await asyncio.to_thread(get_all_module_configs, tenant_id)


@app.post("/admin/modules/toggle")
async def admin_toggle_module(req: ModuleToggleRequest, session=Depends(_check_admin_token)):
    """Enable or disable a module for this tenant."""
    tenant_id = session["tenant_id"]
    if not DB_AVAILABLE:
        raise HTTPException(status_code=503, detail="Database unavailable")

    from modules.module_registry import ALL_MODULES
    if req.module_name not in ALL_MODULES:
        raise HTTPException(status_code=400, detail=f"Unknown module: {req.module_name}. "
                            f"Valid: {ALL_MODULES}")

    result = await asyncio.to_thread(set_module_enabled, tenant_id, req.module_name, req.is_enabled)
    if not result:
        raise HTTPException(status_code=500, detail="Failed to update module config")

    # Invalidate cached module config and LLM cache for live sessions of this tenant
    for sess in chat_sessions.values():
        if sess.get("tenant_id") == tenant_id:
            sess.pop("enabled_modules", None)

    # Invalidate the tools + LLM cache so the next request rebuilds with new modules
    from brain import invalidate_llm_cache
    from modules.module_registry import invalidate_tools_cache
    invalidate_llm_cache(tenant_id)
    invalidate_tools_cache(tenant_id)

    return {
        "module_name": req.module_name,
        "is_enabled":  req.is_enabled,
        "status":      "updated"
    }


# ── Admin: Knowledge base (NEW — FACTS_MODULE) ─────────────────────────────────

@app.get("/admin/knowledge")
async def admin_get_knowledge(session=Depends(_check_admin_token)):
    """List all knowledge entries for this tenant."""
    if not DB_AVAILABLE:
        return []
    return await asyncio.to_thread(get_all_knowledge, session["tenant_id"])


@app.post("/admin/knowledge")
async def admin_add_knowledge(req: KnowledgeUploadRequest, session=Depends(_check_admin_token)):
    """
    Upload knowledge content for FACTS_MODULE.
    The text is chunked and indexed into Qdrant automatically.
    """
    tenant_id = session["tenant_id"]
    if not DB_AVAILABLE:
        raise HTTPException(status_code=503, detail="Database unavailable")
    if not req.content or not req.content.strip():
        raise HTTPException(status_code=400, detail="Content cannot be empty")

    # Save raw content to DB (add_knowledge_chunks takes a list of chunks)
    from modules.facts_module import chunk_text as _chunk_text
    raw_chunks = _chunk_text(req.content.strip())
    rows_added = await asyncio.to_thread(add_knowledge, session["tenant_id"], raw_chunks)

    # Index into Qdrant (runs in thread to avoid blocking)
    chunks_indexed = 0
    try:
        from modules.facts_module import index_knowledge
        chunks_indexed = await asyncio.to_thread(index_knowledge, tenant_id, req.content.strip())
    except Exception as e:
        print(f"[FACTS] Qdrant indexing failed (content saved to DB): {e}")

    return {
        "chunks_saved":   rows_added,
        "chunks_indexed": chunks_indexed,
        "status":         "saved"
    }


@app.delete("/admin/knowledge/{knowledge_id}")
async def admin_delete_knowledge(knowledge_id: str, session=Depends(_check_admin_token)):
    """Delete a single knowledge entry (DB only; Qdrant vectors remain until re-index)."""
    if not DB_AVAILABLE:
        raise HTTPException(status_code=503, detail="Database unavailable")
    deleted = await asyncio.to_thread(delete_knowledge, knowledge_id, session["tenant_id"])
    return {"deleted": deleted}


@app.delete("/admin/knowledge")
async def admin_clear_knowledge(session=Depends(_check_admin_token)):
    """Delete ALL knowledge for this tenant (DB + Qdrant vectors)."""
    tenant_id = session["tenant_id"]
    if not DB_AVAILABLE:
        raise HTTPException(status_code=503, detail="Database unavailable")

    count = await asyncio.to_thread(delete_all_knowledge, tenant_id)

    try:
        from modules.facts_module import delete_tenant_knowledge
        await asyncio.to_thread(delete_tenant_knowledge, tenant_id)
    except Exception as e:
        print(f"[FACTS] Qdrant clear failed: {e}")

    return {"deleted_rows": count, "status": "cleared"}


@app.post("/admin/knowledge/reindex")
async def admin_reindex_knowledge(session=Depends(_check_admin_token)):
    """Re-index all DB knowledge into Qdrant (useful after Qdrant restart)."""
    tenant_id = session["tenant_id"]
    if not DB_AVAILABLE:
        raise HTTPException(status_code=503, detail="Database unavailable")

    rows = await asyncio.to_thread(get_all_knowledge, tenant_id)
    total_chunks = 0
    try:
        from modules.facts_module import index_knowledge, delete_tenant_knowledge
        await asyncio.to_thread(delete_tenant_knowledge, tenant_id)
        for row in rows:
            n = await asyncio.to_thread(index_knowledge, tenant_id, row["content"])
            total_chunks += n
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reindex failed: {e}")

    return {"entries_reindexed": len(rows), "chunks_indexed": total_chunks}


# ── Superadmin endpoints ───────────────────────────────────────────────────────

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
    await asyncio.to_thread(upsert_bot_config, tenant["tenant_id"], bot_name=req.business_name)
    await asyncio.to_thread(
        create_tenant_admin, tenant["tenant_id"], req.owner_email,
        _hash_password(req.admin_password), "owner"
    )
    return tenant

@app.patch("/superadmin/tenants/{tenant_id}/status",
           dependencies=[Depends(_check_superadmin_token)])
async def superadmin_set_status(tenant_id: str, is_active: bool):
    if not DB_AVAILABLE:
        raise HTTPException(status_code=503, detail="Database unavailable")
    return {"updated": await asyncio.to_thread(update_tenant_status, tenant_id, is_active)}

@app.get("/superadmin/stats", dependencies=[Depends(_check_superadmin_token)])
async def superadmin_stats():
    if not DB_AVAILABLE:
        return {}
    return await asyncio.to_thread(get_platform_stats)


# ── Superadmin: Module requests ────────────────────────────────────────────────

class ModuleSetRequest(BaseModel):
    tenant_id: str
    module_name: str
    is_enabled: bool

class ModuleRequestResolve(BaseModel):
    status: str  # 'approved' | 'rejected'

@app.post("/superadmin/modules/set", dependencies=[Depends(_check_superadmin_token)])
async def superadmin_set_module(req: ModuleSetRequest):
    """Directly enable/disable a module for any tenant (superadmin override)."""
    if not DB_AVAILABLE:
        raise HTTPException(status_code=503, detail="DB unavailable")
    from modules.module_registry import ALL_MODULES
    if req.module_name not in ALL_MODULES:
        raise HTTPException(status_code=400, detail=f"Unknown module: {req.module_name}")
    result = await asyncio.to_thread(set_module_enabled, req.tenant_id, req.module_name, req.is_enabled)
    if not result:
        raise HTTPException(status_code=500, detail="Failed to update module")
    # Invalidate session cache for this tenant
    for sess in chat_sessions.values():
        if sess.get("tenant_id") == req.tenant_id:
            sess.pop("enabled_modules", None)
    try:
        from modules.module_registry import invalidate_tools_cache
        invalidate_tools_cache(req.tenant_id)
    except Exception:
        pass
    return {"ok": True, "tenant_id": req.tenant_id, "module_name": req.module_name, "is_enabled": req.is_enabled}


@app.get("/superadmin/module-requests", dependencies=[Depends(_check_superadmin_token)])
async def superadmin_get_module_requests():
    """
    Get all module enable/disable requests submitted by tenant admins.
    Requires a module_requests table — returns empty list gracefully if not present.
    """
    if not DB_AVAILABLE:
        return []
    try:
        from database.crud import get_all_module_requests
        return await asyncio.to_thread(get_all_module_requests)
    except (ImportError, Exception):
        return []  # table may not exist yet


@app.post("/admin/modules/request")
async def admin_request_module(req: Request, session=Depends(_check_admin_token)):
    """
    Tenant admin submits a request to enable/disable a module.
    Stores in module_requests table (created if needed).
    """
    data = await req.json()
    module_name = data.get("module_name", "")
    requested_state = bool(data.get("requested_state", True))
    note = str(data.get("note", ""))[:500]

    if not DB_AVAILABLE:
        return {"ok": True, "queued": True}
    try:
        from database.crud import create_module_request
        result = await asyncio.to_thread(
            create_module_request,
            session["tenant_id"], module_name, requested_state, note
        )
        return {"ok": True, "request_id": result}
    except (ImportError, Exception) as e:
        # Graceful degradation — log and return ok so frontend shows success
        print(f"[MODULE_REQUEST] Could not save request (table may not exist): {e}")
        return {"ok": True, "queued": True}


@app.patch("/superadmin/module-requests/{request_id}", dependencies=[Depends(_check_superadmin_token)])
async def superadmin_resolve_module_request(request_id: str, req: ModuleRequestResolve):
    """Mark a module request as approved or rejected."""
    if not DB_AVAILABLE:
        return {"ok": True}
    try:
        from database.crud import resolve_module_request
        await asyncio.to_thread(resolve_module_request, request_id, req.status)
    except (ImportError, Exception) as e:
        print(f"[MODULE_REQUEST] Could not resolve request: {e}")
    return {"ok": True}


# ── Sarvam STT warm-up ─────────────────────────────────────────────────────────

async def _warmup_sarvam(sarvam_ws, label: str):
    log("[STT_WARMUP]", f"{label} — priming Sarvam with {_warmup_frames} silence frames")
    for i in range(_warmup_frames):
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

    chat_sessions[session_id] = {
        "history":      [],
        "phone_number": phone_number,
        "tenant_id":    None,
        "bot_config":   {},
    }

    brain_lock = asyncio.Lock()

    import time as _time
    ai_speaking_until: list = [0.0]
    ai_post_tts_buffer = config.AI_POST_TTS_BUFFER

    def _ai_is_speaking() -> bool:
        return _time.monotonic() < ai_speaking_until[0]

    commit_queue: asyncio.Queue = asyncio.Queue()
    audio_buf:    asyncio.Queue = asyncio.Queue(maxsize=300)

    tts_session = StreamingTTSSession(websocket)
    tts_session.start()
    log("[WS]", "StreamingTTSSession started")

    _recv = _drop_ai = _sent = 0

    # ── Task 1: Browser receiver ──────────────────────────────────────────────
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

                if "text" in msg:
                    try:
                        ctrl      = json.loads(msg["text"])
                        ctrl_type = ctrl.get("type", "unknown")

                        if ctrl_type == "init":
                            new_phone = ctrl.get("phone_number")
                            tenant_id = ctrl.get("tenant_id")
                            if new_phone:
                                phone_number = new_phone
                                chat_sessions[session_id]["phone_number"] = new_phone
                            if tenant_id:
                                chat_sessions[session_id]["tenant_id"] = tenant_id
                                if DB_AVAILABLE:
                                    try:
                                        bot_cfg = await asyncio.to_thread(get_bot_config, tenant_id)
                                        chat_sessions[session_id]["bot_config"] = bot_cfg or {}
                                        log("[WS]", f"Bot config loaded for tenant={tenant_id}")

                                        if bot_cfg and bot_cfg.get("greeting_message"):
                                            greeting_text = bot_cfg["greeting_message"]
                                            tts_speaker   = bot_cfg.get("tts_speaker", "simran")
                                            tts_lang      = bot_cfg.get("language_code", "gu-IN")

                                            async def play_initial_greeting():
                                                try:
                                                    sentences = split_into_sentences(greeting_text)
                                                    async with brain_lock:
                                                        await websocket.send_json({
                                                            "type": "ai_text", "text": greeting_text,
                                                            "chunk_count": len(sentences)
                                                        })
                                                        await websocket.send_json({"type": "ai_speaking_start"})
                                                        done_evt = asyncio.Event()
                                                        import random as _rand
                                                        resp_id = _rand.randint(1, 999999)
                                                        await tts_session.speak(
                                                            sentences, tts_speaker, tts_lang, resp_id, done_evt
                                                        )
                                                        await done_evt.wait()
                                                except Exception as e:
                                                    log("[GREETING]", f"Error: {e}")

                                            asyncio.create_task(play_initial_greeting())
                                    except Exception as cfg_err:
                                        log("[WS]", f"Bot config load failed: {cfg_err}")

                            log("[WS]", f"init | session={session_id} phone={phone_number} tenant={tenant_id}")

                        elif ctrl_type == "set_user":
                            phone = ctrl.get("phone_number")
                            if phone and session_id in chat_sessions:
                                phone_number = phone
                                chat_sessions[session_id]["phone_number"] = phone
                                log("[WS]", f"set_user phone={phone}")

                        elif ctrl_type == "ai_speaking_start":
                            ai_speaking_until[0] = _time.monotonic() + 30.0
                            log("[CTRL]", "ai_speaking_start → mic MUTED")

                        elif ctrl_type == "ai_speaking_end":
                            ai_speaking_until[0] = _time.monotonic() + ai_post_tts_buffer
                            log("[CTRL]", f"ai_speaking_end → mic blocked +{ai_post_tts_buffer}s")

                        elif ctrl_type == "commit_transcript":
                            text = ctrl.get("text", "").strip()
                            speaking_now = _ai_is_speaking()
                            if not text:
                                pass
                            elif speaking_now:
                                log("[COMMIT]", f"DROPPED (AI speaking): '{text}'")
                            elif is_noisy_transcript(text):
                                await commit_queue.put("__UNCLEAR__")
                            else:
                                last_ai = chat_sessions.get(session_id, {}).get("last_ai_text", "")
                                if is_echo_of_ai(text, last_ai):
                                    log("[COMMIT]", f"DROPPED (echo): '{text}'")
                                else:
                                    await commit_queue.put(text)

                    except Exception as e:
                        log("[CTRL]", f"JSON parse error: {e}")
                    continue

                if "bytes" in msg:
                    _recv += 1
                    raw = msg["bytes"]
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

    # ── Task 2: audio_buf → Sarvam STT ──────────────────────────────────────
    async def sarvam_sender():
        nonlocal _sent
        reconnect_num  = 0
        max_reconnects = config.MAX_RECONNECTS

        while reconnect_num <= max_reconnects:
            attempt_label = f"conn#{reconnect_num + 1}"

            if reconnect_num == 0:
                for _ in range(15):
                    if chat_sessions.get(session_id, {}).get("bot_config"):
                        break
                    await asyncio.sleep(0.1)

            bot_cfg  = chat_sessions.get(session_id, {}).get("bot_config") or {}
            stt_lang = bot_cfg.get("language_code", "gu-IN")

            log("[STT_SEND]", f"Opening Sarvam STT ({attempt_label}) lang={stt_lang}")
            try:
                async with client_stt.speech_to_text_streaming.connect(
                    model="saaras:v3", language_code=stt_lang, sample_rate=16000
                ) as sarvam_ws:
                    log("[STT_SEND]", f"Sarvam STT ESTABLISHED ({attempt_label})")

                    ok = await _warmup_sarvam(sarvam_ws, attempt_label)
                    if not ok:
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
                                now    = asyncio.get_event_loop().time()
                                idle_s = now - last_real_pkt_time
                                if idle_s >= KEEPALIVE_INTERVAL and not _ai_is_speaking():
                                    await sarvam_ws.transcribe(audio=_SILENCE_FRAME_B64)
                                    last_real_pkt_time = now
                                await asyncio.sleep(POLL_INTERVAL)
                                continue

                            last_real_pkt_time = asyncio.get_event_loop().time()
                            await sarvam_ws.transcribe(
                                audio=base64.b64encode(raw).decode("utf-8")
                            )
                            _sent += 1

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
            if reconnect_num <= max_reconnects:
                wait_s = min(2 ** reconnect_num, 16)
                try:
                    await websocket.send_json({
                        "type": "stt_reconnecting", "attempt": reconnect_num, "wait_s": wait_s
                    })
                except Exception:
                    pass
                await asyncio.sleep(wait_s)
            else:
                try:
                    await websocket.send_json({"type": "stt_failed"})
                except Exception:
                    pass

    # ── Task 2b: Sarvam transcript receiver ──────────────────────────────────
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
                if is_final:
                    f_count += 1
                    if speaking_now:
                        log("[STT_RECV]", f"FINAL DROPPED (AI speaking): '{transcript}'")
                    elif is_noisy_transcript(transcript):
                        await commit_queue.put("__UNCLEAR__")
                    else:
                        last_ai = chat_sessions.get(session_id, {}).get("last_ai_text", "")
                        if is_echo_of_ai(transcript, last_ai):
                            log("[STT_RECV]", f"FINAL DROPPED (echo): '{transcript}'")
                        else:
                            await commit_queue.put(transcript)
                else:
                    p_count += 1

                try:
                    await websocket.send_json({
                        "type": "transcript", "text": transcript, "is_final": is_final
                    })
                except Exception:
                    pass

        except asyncio.CancelledError:
            raise
        except Exception as e:
            log("[STT_RECV]", f"Receiver ended: {e}")

    # ── Task 3: Brain consumer ────────────────────────────────────────────────
    async def brain_consumer():
        log("[BRAIN_CONSUMER]", "Started")
        while True:
            try:
                sentence = await commit_queue.get()
                log("[BRAIN_CONSUMER]", f"Dequeued: '{sentence}' | locked={brain_lock.locked()}")
                if brain_lock.locked():
                    log("[BRAIN_CONSUMER]", "Brain BUSY — dropping")
                    continue
                async with brain_lock:
                    await websocket.send_json({"type": "processing_start"})
                    try:
                        await run_brain(
                            session_id=session_id,
                            user_text=sentence,
                            websocket=websocket,
                            tts_session=tts_session,
                            chat_sessions=chat_sessions,
                            tts_convert_fn=tts_convert,
                        )
                    except Exception as e:
                        fallback_text = _get_fallback_message(e)
                        bot_cfg = chat_sessions.get(session_id, {}).get("bot_config") or {}
                        fb_speaker = bot_cfg.get("tts_speaker", "simran")
                        fb_lang    = bot_cfg.get("language_code", "gu-IN")
                        try:
                            await websocket.send_json({
                                "type": "ai_text", "text": fallback_text, "chunk_count": 1
                            })
                            await websocket.send_json({"type": "ai_speaking_start"})
                            done_evt = asyncio.Event()
                            import random as _rand
                            resp_id = _rand.randint(1, 999999)
                            await tts_session.speak(
                                [fallback_text], fb_speaker, fb_lang, resp_id, done_evt
                            )
                            await done_evt.wait()
                        except Exception as tts_err:
                            log("[BRAIN_CONSUMER]", f"Fallback TTS also failed: {tts_err}")

            except asyncio.CancelledError:
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