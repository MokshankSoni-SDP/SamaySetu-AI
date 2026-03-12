import os
import re
import json
import base64
import asyncio
import struct
import traceback
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sarvamai import AsyncSarvamAI
from dotenv import load_dotenv
from typing import Dict, List, Optional
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

import prompts
from calendar_tool import *
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from datetime import datetime

# ── Database imports ────────────────────────────────────────────────────────
try:
    from database.models import create_tables
    from database.crud import (
        create_user_if_not_exists,
        get_user_appointments,
    )
    DB_AVAILABLE = True
except ImportError as _db_import_err:
    DB_AVAILABLE = False
    print(f"[DB] psycopg2 not installed or database module missing: {_db_import_err}")

# ── Logging helper ─────────────────────────────────────────────────────────────
def log(step: str, msg: str):
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{ts}] {step}: {msg}")

# ── Setup & Environment ────────────────────────────────────────────────────────
load_dotenv()


# ── FastAPI lifespan: initialise DB tables on startup ──────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Run startup tasks (DB table creation) then yield to serve requests."""
    if DB_AVAILABLE:
        await asyncio.to_thread(create_tables)
    else:
        print("[DB] Skipping table creation — psycopg2 unavailable.")
    yield   # application runs
    # (shutdown cleanup goes here if needed)


app = FastAPI(lifespan=lifespan)


# ── Pydantic models ─────────────────────────────────────────────────────────
class UserLoginRequest(BaseModel):
    phone_number: str
    name: Optional[str] = None

SARVAM_API_KEY = os.getenv('SARVAM_API_KEY')
client_stt = AsyncSarvamAI(api_subscription_key=SARVAM_API_KEY)
client_tts = AsyncSarvamAI(api_subscription_key=SARVAM_API_KEY)

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1)
tools = [check_calendar_availability, book_appointment, cancel_appointment,
         reschedule_appointment, suggest_next_available_slot]
llm_with_tools = llm.bind_tools(tools)

app.mount("/static", StaticFiles(directory="static"), name="static")

# chat_sessions maps session_id → {"history": [...], "phone_number": str|None}
chat_sessions: Dict[str, dict] = {}
MAX_HISTORY = 5  # increased from 5 — avoids context amnesia on longer conversations

# maximum number of iteration llm can make 
MAX_TOOL_ITERATIONS = 4

# Gujarati number words → digits
GUJ_NUMBERS = {
    "એક": 1, "બે": 2, "ત્રણ": 3, "ચાર": 4,
    "પાંચ": 5, "છ": 6, "સાત": 7, "આઠ": 8,
    "નવ": 9, "દસ": 10, "અગિયાર": 11, "બાર": 12
}

# Handle half / quarter / special phrases
SPECIAL_PHRASES = {
    "દોઢ": (1, 30),
    "અઢી": (2, 30),
}

COMMON_STT_VARIANTS = {
    "સ્વા": "સવા",
    "સવા": "સવા",
    "સાડા": "સાડા",
    "સાડા": "સાડા",
    "પોના": "પોણા",
    "પોણા": "પોણા"
}



# ── Noisy STT filter (Fix #2) ──────────────────────────────────────────────────
import re as _re

def is_noisy_transcript(text: str) -> bool:
    """
    Heuristic to detect garbled / low-confidence STT output before it reaches
    the LLM. Returns True (= noisy, should not be processed as intent) if ANY
    of the following are true:

      1. Very short after stripping punctuation/spaces (≤ 3 meaningful chars)
      2. Word repetition: same word appears ≥ 3 times in a short string
      3. High repetition ratio: > 50 % of tokens are duplicates
      4. Incomplete / truncated fragments that contain no verb or noun signal

    These patterns appeared directly in the logs (e.g. "ડાયલેક્ટ ડાયલેક્ટ
    કરી બપોરે બપોરે 3:00 વાગ્યા વાગ્યાની અપ").
    """
    stripped = _re.sub(r'[\s.,!?।\u0964]+', ' ', text).strip()

    # Rule 1 — too short to be meaningful
    if len(stripped) <= 3:
        return True

    tokens = stripped.split()

    # Rule 2 — any single word repeated 3+ times
    from collections import Counter
    counts = Counter(tokens)
    if any(v >= 3 for v in counts.values()):
        return True

    # Rule 3 — duplicate tokens make up more than half the message
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
    """
    Convert Gujarati spoken time phrases into numeric format.
    Example:
    'સવા ત્રણ' → '3:15'
    'સાડા ચાર' → '4:30'
    'પોણા બે' → '1:45'
    """

    original_text = text

    # normalize spacing
    text = text.lower().strip()

    # handle દોઢ / અઢી
    for phrase, (h, m) in SPECIAL_PHRASES.items():
        if phrase in text:
            time_str = f"{h}:{m:02d}"
            text = text.replace(phrase, time_str)

    text = normalize_variants(text)

    # સવા (quarter past)
    match = re.search(r"સવા\s*(\w+)", text)
    if match:
        hour_word = match.group(1)
        hour = GUJ_NUMBERS.get(hour_word)
        if hour:
            time_str = f"{hour}:{15}"
            text = text.replace(match.group(0), time_str)

    # સાડા (half past)
    match = re.search(r"સાડા\s*(\w+)", text)
    if match:
        hour_word = match.group(1)
        hour = GUJ_NUMBERS.get(hour_word)
        if hour:
            time_str = f"{hour}:{30}"
            text = text.replace(match.group(0), time_str)

    # પોણા (quarter to)
    match = re.search(r"પોણા\s*(\w+)", text)
    if match:
        hour_word = match.group(1)
        hour = GUJ_NUMBERS.get(hour_word)
        if hour:
            hour -= 1
            time_str = f"{hour}:{45}"
            text = text.replace(match.group(0), time_str)

    return text

# ── Fallback message selector (Fix #7) ────────────────────────────────────────
def _get_fallback_message(exc: Exception) -> str:
    """
    Maps known exception types to a polite Gujarati fallback phrase that is
    spoken aloud to the user instead of silence.
    """
    err_str = str(exc).lower()
    if "rate_limit" in err_str or "429" in err_str or "tokens per day" in err_str:
        return (
            "માફ કરશો, અત્યારે સર્વર ખૂબ વ્યસ્ત છે. "
            "થોડી વાર પછી ફરી પ્રયત્ન કરો."
        )
    if "timeout" in err_str or "timed out" in err_str:
        return (
            "માફ કરશો, જવાબ આવવામાં વધુ સમય લાગ્યો. "
            "કૃપા કરી ફરી પ્રયત્ન કરો."
        )
    if "connection" in err_str or "network" in err_str:
        return (
            "નેટવર્ક સમસ્યા આવી. "
            "કૃપા કરી ઇન્ટરનેટ તપાસો અને ફરી બોલો."
        )
    # Generic fallback
    return (
        "માફ કરશો, કંઈક ભૂલ થઈ. "
        "થોડી વાર પછી ફરી પ્રયત્ન કરો."
    )


# ── REST endpoints ──────────────────────────────────────────────────────────

@app.post("/user/login")
async def user_login(req: UserLoginRequest):
    """
    Registers a new user or returns existing user details.
    Phone number acts as the primary identifier.
    """
    if not req.phone_number or not req.phone_number.strip():
        raise HTTPException(status_code=400, detail="phone_number is required")

    if not DB_AVAILABLE:
        # Graceful degradation: allow login even without DB
        return {"status": "success", "phone_number": req.phone_number, "db": False}

    try:
        await asyncio.to_thread(
            create_user_if_not_exists,
            req.phone_number.strip(),
            req.name
        )
        return {"status": "success", "phone_number": req.phone_number.strip()}
    except Exception as e:
        print(f"[DB] /user/login error: {e}")
        raise HTTPException(status_code=500, detail="Database error during login")


@app.get("/appointments/{phone_number}")
async def get_appointments(phone_number: str):
    """
    Returns all appointments for a user ordered by most recent first.
    """
    if not DB_AVAILABLE:
        return []
    try:
        appointments = await asyncio.to_thread(get_user_appointments, phone_number)
        return appointments
    except Exception as e:
        print(f"[DB] /appointments/{phone_number} error: {e}")
        raise HTTPException(status_code=500, detail="Could not fetch appointments")

# ── Token usage logger ─────────────────────────────────────────────────────────
def print_token_usage(msg, step_name):
    usage = msg.response_metadata.get("token_usage", {})
    if usage:
        print(f"--- 📊 {step_name} | prompt={usage.get('prompt_tokens')} "
              f"completion={usage.get('completion_tokens')} "
              f"total={usage.get('total_tokens')} ---")

# ── Calendar tool runner (non-blocking) ───────────────────────────────────────
async def run_tool_async(tool_name: str, args: dict):
    func = globals()[tool_name]
    return await asyncio.to_thread(func, **args)

# ── Sentence splitter ──────────────────────────────────────────────────────────
# Splits Gujarati + mixed text on sentence-ending punctuation.
# Keeps chunks >= MIN_CHUNK_CHARS so we don't fire TTS for tiny fragments.
MIN_CHUNK_CHARS = 20

def split_into_sentences(text: str) -> List[str]:
    """
    Split reply text into TTS-ready sentence chunks.
    Handles Gujarati (।), Hindi (।) and standard punctuation (. ! ?).
    Short fragments are merged with the previous chunk to avoid
    tiny round-trips to the TTS API.
    """
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
            chunks[-1] += " " + buffer   # merge tiny tail into last chunk
        else:
            chunks.append(buffer)
    return chunks or [text]             # fallback: treat whole text as 1 chunk

# ── TTS with retry ─────────────────────────────────────────────────────────────
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=4),
    retry=retry_if_exception_type(Exception),
    before_sleep=lambda rs: log("[TTS_RETRY]", f"Attempt {rs.attempt_number} failed, retrying…")
)
async def tts_convert(text: str) -> str:
    """Call Sarvam TTS and return base64 audio string."""
    res = await client_tts.text_to_speech.convert(
        text=text,
        target_language_code="gu-IN",
        model="bulbul:v3",
        speaker="simran"
    )
    return res.audios[0]

# ── LLM with retry ────────────────────────────────────────────────────────────
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(min=1, max=5),
    retry=retry_if_exception_type(Exception),
    before_sleep=lambda rs: log("[LLM_RETRY]", f"Attempt {rs.attempt_number} failed, retrying…")
)
async def safe_llm_call(messages):
    return await llm_with_tools.ainvoke(messages)

# ── Brain ──────────────────────────────────────────────────────────────────────
async def run_brain(session_id: str, user_text: str, websocket: WebSocket):
    """
    Streaming TTS pipeline:
      1. LLM produces full reply text           (~1 s)
      2. Split reply into sentences
      3. Send full text to frontend immediately  (chat bubble appears)
      4. For each sentence in order:
           - Generate TTS chunk                 (~2 s per chunk)
           - Send audio_chunk to frontend       (frontend queues + plays sequentially)
      5. Send tts_done so frontend knows it's over

    Result: user hears first sentence at ~3 s total instead of waiting
    11 s for the entire audio blob.
    """
    t0    = datetime.now()
    today = t0.strftime("%Y-%m-%d")
    day   = t0.strftime("%A")

    # normalize Gujarati time phrases
    log("[NORMALIZER]", f"Before: {user_text}")
    user_text = normalize_gujarati_time(user_text)
    log("[NORMALIZER]", f"After: {user_text}")

    log("[BRAIN]", f"START | session='{session_id}' | text='{user_text}'")

    # ── Fix #2: Handle noisy/garbled STT — respond with clarification, skip tools ──
    if user_text == "__UNCLEAR__":
        clarification = "માફ કરશો, મને સ્પષ્ટ સંભળાયું નહીં. શું તમે ફરીથી કહી શકો?"
        log("[BRAIN]", "Noisy input — sending clarification without LLM call")
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

    # Initialise or retrieve session metadata.
    # NOTE: The WS handler pre-creates the session with empty history so the
    # phone_number is available immediately. We detect this here and inject the
    # system prompt, ensuring the LLM always knows today's date.
    if session_id not in chat_sessions:
        log("[BRAIN]", "New session — initialising with system prompt")
        chat_sessions[session_id] = {
            "history": [],
            "phone_number": None,
        }

    session_data = chat_sessions[session_id]
    history      = session_data["history"]
    phone_number = session_data.get("phone_number")

    # Add system prompt if missing (first call in this session)
    if not history or not isinstance(history[0], SystemMessage):
        log("[BRAIN]", f"Injecting system prompt | today={today} day={day}")
        history.insert(0, SystemMessage(content=prompts.get_system_prompt(today, day)))
    else:
        log("[BRAIN]", f"Existing session | history len={len(history)}")

    history.append(HumanMessage(content=user_text))
    recent_history = [history[0]] + history[-MAX_HISTORY:]


    # ── Step 1: LLM ──────────────────────────────────────────────────────────
    t_llm = datetime.now()
    log("[LLM]", f"ainvoke() | {len(recent_history)} messages")
    try:
        ai_msg = await safe_llm_call(recent_history)
        log("[LLM]", f"Done in {(datetime.now()-t_llm).total_seconds():.2f}s | "
            f"tool_calls={len(ai_msg.tool_calls)} | preview='{str(ai_msg.content)[:80]}'")
        print_token_usage(ai_msg, "Initial LLM")
    except Exception as e:
        log("[LLM]", f"FAILED: {e}\n{traceback.format_exc()}")
        raise

    # ── Step 2: Tool loop ─────────────────────────────────────────────────────
    tool_iteration = 0
    while ai_msg.tool_calls:
        tool_iteration += 1

        if tool_iteration > MAX_TOOL_ITERATIONS:
                log("[TOOLS]", f"Max tool iterations reached ({MAX_TOOL_ITERATIONS}). Forcing AI response.")
                break

        log("[TOOLS]", f"Iteration #{tool_iteration} — {len(ai_msg.tool_calls)} tool(s)")
        history.append(ai_msg)

        async def execute_tool(tool_call):
            tname = tool_call["name"]
            targs = tool_call["args"]
            # Inject phone_number for DB integration (ignored if None)
            if phone_number:
                targs = {**targs, "phone_number": phone_number}
            log("[TOOL]", f"Executing '{tname}' | args={targs}")
            await websocket.send_json({
                "type": "tool_call", "name": tname, "args": targs, "status": "running"
            })
            t_tool = datetime.now()
            try:
                obs    = await run_tool_async(tname, targs)
                status = "ok"
                log("[TOOL]", f"'{tname}' OK in {(datetime.now()-t_tool).total_seconds():.2f}s "
                    f"| result='{str(obs)[:100]}'")
            except Exception as e:
                obs, status = f"Error: {e}", "error"
                log("[TOOL]", f"'{tname}' FAILED: {e}")
            await websocket.send_json({
                "type": "tool_call", "name": tname, "args": targs,
                "status": status, "observation": str(obs)
            })
            return ToolMessage(content=str(obs), tool_call_id=tool_call["id"])

        tool_messages = await asyncio.gather(*[execute_tool(tc) for tc in ai_msg.tool_calls])
        history.extend(tool_messages)

        recent_history = [history[0]] + history[-MAX_HISTORY:]
        t_llm2 = datetime.now()
        log("[LLM]", f"Re-invoking after tool iteration #{tool_iteration}")
        try:
            ai_msg = await safe_llm_call(recent_history)
            log("[LLM]", f"Post-tool done in {(datetime.now()-t_llm2).total_seconds():.2f}s | "
                f"tool_calls={len(ai_msg.tool_calls)} | preview='{str(ai_msg.content)[:80]}'")
            print_token_usage(ai_msg, f"Post-Tool #{tool_iteration}")
        except Exception as e:
            log("[LLM]", f"POST-TOOL FAILED: {e}\n{traceback.format_exc()}")
            raise

    history.append(ai_msg)
    reply_text = ai_msg.content

    # ── Step 3: Send full text immediately (chat bubble appears before audio) ─
    sentences = split_into_sentences(reply_text)
    log("[TTS]", f"Reply split into {len(sentences)} chunk(s): {[s[:40]+'…' for s in sentences]}")

    await websocket.send_json({
        "type":        "ai_text",
        "text":        reply_text,
        "chunk_count": len(sentences)
    })

    # ── Step 4: Generate + stream TTS chunks sequentially ────────────────────
    # Sequential (not parallel) keeps audio in correct order without any
    # reordering logic on the frontend. Each chunk takes ~2 s; the user
    # hears chunk 1 before chunk 2 is even generated.
    #
    # FIX: Tell the frontend (and backend ai_speaking gate) that the AI is
    # about to speak BEFORE we begin generating TTS.  This closes the race
    # window where the brain_consumer lock releases (un-muting the mic) while
    # TTS generation is still running and audio has not yet reached the client.
    await websocket.send_json({"type": "ai_speaking_start"})
    log("[TTS]", "ai_speaking_start sent — mic MUTED while TTS generates")

    t_tts_total = datetime.now()
    for idx, sentence in enumerate(sentences):
        t_chunk = datetime.now()
        log("[TTS]", f"Chunk {idx+1}/{len(sentences)} start | '{sentence[:60]}'")
        try:
            audio_b64 = await tts_convert(sentence)
            log("[TTS]", f"Chunk {idx+1} ready in {(datetime.now()-t_chunk).total_seconds():.2f}s "
                f"| b64_len={len(audio_b64)}")
        except Exception as e:
            # Skip failed chunks — better to have a short gap than crash everything
            log("[TTS]", f"Chunk {idx+1} FAILED after retries: {e} — skipping")
            continue

        await websocket.send_json({
            "type":    "audio_chunk",
            "index":   idx,
            "total":   len(sentences),
            "text":    sentence,
            "audio":   audio_b64,
            "is_last": idx == len(sentences) - 1
        })
        log("[TTS]", f"Chunk {idx+1} sent to frontend")

    log("[TTS]", f"All chunks done | total_tts={( datetime.now()-t_tts_total).total_seconds():.2f}s")
    log("[BRAIN]", f"COMPLETE in {(datetime.now()-t0).total_seconds():.2f}s total")

    # Signal frontend: no more audio chunks for this response
    await websocket.send_json({"type": "tts_done"})


# ── WebSocket ──────────────────────────────────────────────────────────────────
@app.websocket("/ws/voice")
async def websocket_endpoint(websocket: WebSocket, phone_number: Optional[str] = None):
    """Voice WebSocket. Accepts optional ?phone_number=... query param to associate
    the session with a logged-in user so calendar ops can write to the DB."""
    await websocket.accept()

    # """નમસ્તે! તમે સમયસેતુ AI સાથે જોડાયા છો.  
    #     કૃપા કરીને જણાવશો કે તમને એપોઇન્ટમેન્ટ માટે કયો દિવસ અને સમય અનુકૂળ છે?
    #     """

    session_id = f"web_{datetime.now().strftime('%H%M%S')}"
    log("[WS]", f"Connection ACCEPTED | session_id='{session_id}' | phone='{phone_number}'")

    greeting_text = """નમસ્તે! તમે સમયસેતુ AI સાથે જોડાયા છો.  
        કૃપા કરીને જણાવશો કે તમને એપોઇન્ટમેન્ટ માટે કયો દિવસ અને સમય અનુકૂળ છે?
        """
    
    # ── Play greeting once when connection starts ──
    try:
        sentences = split_into_sentences(greeting_text)

        await websocket.send_json({
            "type": "ai_text",
            "text": greeting_text,
            "chunk_count": len(sentences)
        })

        await websocket.send_json({"type": "ai_speaking_start"})

        for idx, sentence in enumerate(sentences):
            audio_b64 = await tts_convert(sentence)

            await websocket.send_json({
                "type": "audio_chunk",
                "index": idx,
                "total": len(sentences),
                "text": sentence,
                "audio": audio_b64,
                "is_last": idx == len(sentences) - 1
            })

        await websocket.send_json({"type": "tts_done"})

    except Exception as e:
        log("[GREETING]", f"TTS greeting failed: {e}")

    # Store phone_number in session so run_brain() can pass it to calendar tools
    chat_sessions[session_id] = {"history": [], "phone_number": phone_number}

    brain_lock  = asyncio.Lock()
    ai_speaking = asyncio.Event()

    log("[WS]", "Opening Sarvam STT streaming connection (saaras:v3, gu-IN, 16000 Hz)")
    try:
        async with client_stt.speech_to_text_streaming.connect(
            model="saaras:v3", language_code="gu-IN", sample_rate=16000
        ) as sarvam_ws:
            log("[WS]", "Sarvam STT connection ESTABLISHED")

            _recv = _drop_ai = _sent = 0
            commit_queue: asyncio.Queue = asyncio.Queue()

            # ── Task 1: Browser audio → Sarvam STT ───────────────────────────
            async def browser_to_sarvam():
                nonlocal _recv, _drop_ai, _sent
                log("[AUDIO_TASK]", "browser_to_sarvam() started")
                try:
                    while True:
                        
                        try:
                            msg = await websocket.receive()
                        except WebSocketDisconnect:
                            log("[AUDIO_TASK]", "Client disconnected")
                            break  
                        except RuntimeError:
                            log("[AUDIO_TASK]", "Receive after disconnect — stopping")
                            break

                        if "text" in msg:
                            try:
                                ctrl      = json.loads(msg["text"])
                                ctrl_type = ctrl.get("type", "unknown")

                                if ctrl_type == "ai_speaking_start":
                                    ai_speaking.set()
                                    log("[CTRL]", "ai_speaking_start → mic MUTED")
                                elif ctrl_type == "ai_speaking_end":
                                    ai_speaking.clear()
                                    log("[CTRL]", f"ai_speaking_end → mic UN-MUTED | "
                                        f"recv={_recv} sent={_sent} drop_ai={_drop_ai}")
                                elif ctrl_type == "commit_transcript":
                                    text = ctrl.get("text", "").strip()
                                    if text:
                                        log("[COMMIT]", f"Frontend committed: '{text}'")
                                        # Drop commits that arrive while the AI is
                                        # speaking — these are speaker bleed captured by
                                        # the mic before the frontend mute took effect.
                                        if ai_speaking.is_set():
                                            log("[COMMIT]", f"DROPPED (AI speaking — bleed protection): '{text}'")
                                        elif is_noisy_transcript(text):
                                            log("[COMMIT]", f"DROPPED (noisy/garbled STT — will ask clarification): '{text}'")
                                            await commit_queue.put("__UNCLEAR__")
                                        else:
                                            await commit_queue.put(text)
                                elif ctrl_type == "set_user":
                                    # Frontend can also send phone via WS message as backup
                                    phone = ctrl.get("phone_number")
                                    if phone and session_id in chat_sessions:
                                        chat_sessions[session_id]["phone_number"] = phone
                                        log("[WS]", f"User phone updated via set_user: {phone}")
                                else:
                                    log("[CTRL]", f"Unknown type='{ctrl_type}'")
                            except Exception as e:
                                log("[CTRL]", f"JSON parse error: {e}")
                            continue

                        if "bytes" in msg:
                            _recv += 1
                            raw = msg["bytes"]
                            if ai_speaking.is_set():
                                _drop_ai += 1
                                # DIAGNOSTIC: log every dropped packet's RMS so we know
                                # how much AI audio is being successfully blocked
                                if _drop_ai <= 5 or _drop_ai % 20 == 0:
                                    rms = compute_rms(raw)
                                    log("[DIAG:DROP]", f"AI-speaking drop #{_drop_ai} | rms={rms:.0f}")
                                continue
                            try:
                                rms_val = compute_rms(raw)
                                # DIAGNOSTIC: log the first 5 packets after AI stops speaking
                                # to see if residual AI audio is leaking into Sarvam's stream
                                packets_after_unmute = _sent - getattr(browser_to_sarvam, '_sent_at_unmute', _sent)
                                if not ai_speaking.is_set() and _drop_ai > 0:
                                    # Just transitioned from dropping to sending
                                    if not hasattr(browser_to_sarvam, '_unmute_logged'):
                                        browser_to_sarvam._unmute_logged = True
                                        browser_to_sarvam._sent_at_unmute = _sent
                                        log("[DIAG:UNMUTE]", f"First pkt sent after AI stop | rms={rms_val:.0f} | total_dropped={_drop_ai}")
                                    elif (_sent - browser_to_sarvam._sent_at_unmute) <= 5:
                                        log("[DIAG:UNMUTE]", f"Early pkt #{_sent - browser_to_sarvam._sent_at_unmute} post-unmute | rms={rms_val:.0f} ({'SPEECH' if rms_val > 300 else 'silence'})")
                                if ai_speaking.is_set() == False and _drop_ai > 0:
                                    browser_to_sarvam._unmute_logged = False
                                    _drop_ai = 0  # reset after logging

                                await sarvam_ws.transcribe(
                                    audio=base64.b64encode(raw).decode("utf-8")
                                )
                                _sent += 1
                                if _sent % 50 == 1:
                                    rms = compute_rms(raw)
                                    log("[STT_SEND]", f"pkt#{_sent} | rms={rms:.0f} "
                                        f"({'speech' if rms > 300 else 'silence'})")
                            except Exception as e:
                                log("[STT_SEND]", f"FAILED: {e}")

                except WebSocketDisconnect:
                    log("[AUDIO_TASK]", f"WS disconnected | recv={_recv} sent={_sent}")
                except Exception as e:
                    log("[AUDIO_TASK]", f"Crashed: {e}\n{traceback.format_exc()}")

            # ── Task 2: Sarvam STT → frontend ────────────────────────────────
            async def sarvam_to_frontend():
                log("[STT_RECV]", "sarvam_to_frontend() started")
                try:
                    p_count = f_count = 0
                    async for response in sarvam_ws:
                        if not (hasattr(response, "type") and response.type == "data"):
                            continue
                        transcript = getattr(response.data, 'transcript', "").strip()
                        is_final   = getattr(response.data, 'is_final', False)
                        if not transcript:
                            continue

                        # ── DIAGNOSTIC: log AI speaking state alongside every transcript ──
                        speaking_state = "AI_SPEAKING" if ai_speaking.is_set() else "user_turn"
                        tag = "FINAL" if is_final else "partial"
                        log("[DIAG:STT]", f"{tag} [{speaking_state}]: '{transcript}'")

                        if is_final:
                            f_count += 1
                            log("[STT_RECV]", f"FINAL #{f_count}: '{transcript}'")
                            if ai_speaking.is_set():
                                log("[STT_RECV]", f"FINAL DROPPED (AI speaking — bleed): '{transcript}'")
                            elif is_noisy_transcript(transcript):
                                log("[STT_RECV]", f"FINAL DROPPED (noisy): '{transcript}'")
                                await commit_queue.put("__UNCLEAR__")
                            else:
                                await commit_queue.put(transcript)
                        else:
                            p_count += 1
                            if p_count % 5 == 1:
                                log("[STT_RECV]", f"Partial #{p_count}: '{transcript}'")
                        await websocket.send_json({
                            "type": "transcript", "text": transcript, "is_final": is_final
                        })
                except Exception as e:
                    log("[STT_RECV]", f"Crashed: {e}\n{traceback.format_exc()}")

            # ── Task 3: Brain consumer ────────────────────────────────────────
            async def brain_consumer():
                log("[BRAIN_CONSUMER]", "Started — waiting for commits")
                while True:
                    try:
                        sentence = await commit_queue.get()
                        log("[BRAIN_CONSUMER]", f"Dequeued: '{sentence}'")
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
                                # ── Fix #7: Send a graceful Gujarati fallback instead of silence ──
                                fallback_text = _get_fallback_message(e)
                                log("[BRAIN_CONSUMER]", f"Sending fallback TTS: '{fallback_text}'")
                                try:
                                    await websocket.send_json({
                                        "type": "ai_text",
                                        "text": fallback_text,
                                        "chunk_count": 1
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

            log("[WS]", "Launching 3 async tasks")
            await asyncio.gather(browser_to_sarvam(), sarvam_to_frontend(), brain_consumer())
            log("[WS]", "All tasks exited")

    except Exception as e:
        log("[WS]", f"Handler CRASHED: {e}\n{traceback.format_exc()}")


# ── Audio helpers ──────────────────────────────────────────────────────────────
def compute_rms(raw_bytes: bytes) -> float:
    if len(raw_bytes) < 2:
        return 0.0
    samples = struct.unpack(f"<{len(raw_bytes)//2}h", raw_bytes)
    return (sum(s * s for s in samples) / len(samples)) ** 0.5