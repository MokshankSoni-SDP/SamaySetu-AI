import os
import json
import base64
import asyncio
import struct
import traceback
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from sarvamai import AsyncSarvamAI, SarvamAI
from dotenv import load_dotenv
from typing import Dict, List
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

import prompts
from calendar_tool import *
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from datetime import datetime

# ── Logging helper ─────────────────────────────────────────────────────────────
def log(step: str, msg: str):
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{ts}] {step}: {msg}")

# ── Setup & Environment ────────────────────────────────────────────────────────
load_dotenv()
app = FastAPI()

SARVAM_API_KEY = os.getenv('SARVAM_API_KEY')

# FIX 1: Use AsyncSarvamAI for TTS so it doesn't block the event loop.
# The old SarvamAI (sync) client was running a blocking HTTP call inside
# an async function, stalling the entire event loop for 4–8 seconds.
client_stt = AsyncSarvamAI(api_subscription_key=SARVAM_API_KEY)
client_tts = AsyncSarvamAI(api_subscription_key=SARVAM_API_KEY)

# FIX 2: Use ainvoke (async) instead of invoke (sync) for the LLM.
# langchain_groq's .invoke() is a blocking call. In an async FastAPI
# handler it blocks the event loop for the full LLM round-trip (~1s).
# Using .ainvoke() lets other coroutines (audio streaming, etc.) run
# while waiting for Groq's response.
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1)
tools = [check_calendar_availability, book_appointment, cancel_appointment,
         reschedule_appointment, suggest_next_available_slot]
llm_with_tools = llm.bind_tools(tools)

app.mount("/static", StaticFiles(directory="static"), name="static")

chat_sessions: Dict[str, List] = {}
MAX_HISTORY = 5

# ── Token usage logger ─────────────────────────────────────────────────────────
def print_token_usage(msg, step_name):
    usage = msg.response_metadata.get("token_usage", {})
    if usage:
        print(f"--- 📊 {step_name} | prompt={usage.get('prompt_tokens')} "
              f"completion={usage.get('completion_tokens')} "
              f"total={usage.get('total_tokens')} ---")

# ── Calendar tool runner (async wrapper) ───────────────────────────────────────
# FIX 3: Google Calendar API calls are synchronous (blocking HTTP).
# Running them with asyncio.to_thread() moves them off the event loop
# into a thread pool, so the event loop stays responsive during the
# 1–4 second calendar API round-trips.
async def run_tool_async(tool_name: str, args: dict):
    func = globals()[tool_name]
    return await asyncio.to_thread(func, **args)

# ── Brain ──────────────────────────────────────────────────────────────────────
async def run_brain(session_id: str, user_text: str, websocket: WebSocket):
    t0 = datetime.now()
    today = t0.strftime("%Y-%m-%d")
    day   = t0.strftime("%A")

    log("[BRAIN]", f"START | session='{session_id}' | text='{user_text}'")

    if session_id not in chat_sessions:
        log("[BRAIN]", "New session — initialising with system prompt")
        chat_sessions[session_id] = [SystemMessage(content=prompts.get_system_prompt(today, day))]
    else:
        log("[BRAIN]", f"Existing session | history len={len(chat_sessions[session_id])}")

    history = chat_sessions[session_id]
    history.append(HumanMessage(content=user_text))
    recent_history = [history[0]] + history[-MAX_HISTORY:]

    # ── LLM call (async) ──────────────────────────────────────────────────────
    t_llm = datetime.now()
    log("[LLM]", f"ainvoke() start | {len(recent_history)} messages in context")
    try:
        # FIX 2 applied: ainvoke instead of invoke
        ai_msg = await llm_with_tools.ainvoke(recent_history)
        llm_ms = (datetime.now() - t_llm).total_seconds()
        log("[LLM]", f"Done in {llm_ms:.2f}s | tool_calls={len(ai_msg.tool_calls)} | preview='{str(ai_msg.content)[:80]}'")
        print_token_usage(ai_msg, "Initial LLM")
    except Exception as e:
        log("[LLM]", f"FAILED: {e}\n{traceback.format_exc()}")
        raise

    # ── Tool loop ─────────────────────────────────────────────────────────────
    tool_iteration = 0
    while ai_msg.tool_calls:
        tool_iteration += 1
        log("[TOOLS]", f"Iteration #{tool_iteration} — {len(ai_msg.tool_calls)} tool(s) requested")
        history.append(ai_msg)

        # FIX 3 applied: run all tools in this iteration concurrently
        # If the LLM ever requests multiple tools at once, they run in parallel.
        # Even for single-tool iterations this ensures the calendar call is
        # non-blocking (asyncio.to_thread keeps the event loop free).
        async def execute_tool(tool_call):
            tname = tool_call["name"]
            targs = tool_call["args"]
            log("[TOOL]", f"Executing '{tname}' | args={targs}")
            await websocket.send_json({"type": "tool_call", "name": tname, "args": targs, "status": "running"})
            t_tool = datetime.now()
            try:
                obs    = await run_tool_async(tname, targs)
                status = "ok"
                tool_ms = (datetime.now() - t_tool).total_seconds()
                log("[TOOL]", f"'{tname}' OK in {tool_ms:.2f}s | result='{str(obs)[:100]}'")
            except Exception as e:
                obs    = f"Error: {e}"
                status = "error"
                log("[TOOL]", f"'{tname}' FAILED: {e}")
            await websocket.send_json({"type": "tool_call", "name": tname, "args": targs,
                                       "status": status, "observation": str(obs)})
            return ToolMessage(content=str(obs), tool_call_id=tool_call["id"])

        # Run all tools for this iteration concurrently
        tool_messages = await asyncio.gather(*[execute_tool(tc) for tc in ai_msg.tool_calls])
        history.extend(tool_messages)

        recent_history = [history[0]] + history[-MAX_HISTORY:]
        t_llm2 = datetime.now()
        log("[LLM]", f"Re-invoking after tool iteration #{tool_iteration}")
        try:
            ai_msg = await llm_with_tools.ainvoke(recent_history)
            llm2_ms = (datetime.now() - t_llm2).total_seconds()
            log("[LLM]", f"Post-tool done in {llm2_ms:.2f}s | tool_calls={len(ai_msg.tool_calls)} | preview='{str(ai_msg.content)[:80]}'")
            print_token_usage(ai_msg, f"Post-Tool #{tool_iteration}")
        except Exception as e:
            log("[LLM]", f"POST-TOOL FAILED: {e}\n{traceback.format_exc()}")
            raise

    history.append(ai_msg)
    reply_text = ai_msg.content

    # ── TTS (async, non-blocking) ─────────────────────────────────────────────
    # FIX 1 applied: AsyncSarvamAI client — awaitable, doesn't block event loop.

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=4),
        retry=retry_if_exception_type(Exception),
        before_sleep=lambda retry_state: log("[RETRY]", f"TTS failed. Attempt {retry_state.attempt_number}...")
    )
    async def generate_tts(text):
        return await client_tts.text_to_speech.convert(
            text=text,
            target_language_code="gu-IN",
            model="bulbul:v3",
            speaker="simran"
        )

    t_tts = datetime.now()
    log("[TTS]", f"Request start | text_len={len(reply_text)} chars | preview='{reply_text[:80]}'")
    try:
        tts_res   = await generate_tts(reply_text)
        audio_b64 = tts_res.audios[0]
        tts_ms    = (datetime.now() - t_tts).total_seconds()
        log("[TTS]", f"Done in {tts_ms:.2f}s | audio_b64 len={len(audio_b64)} chars")
    except Exception as e:
        log("[TTS]", f"FAILED: {e}\n{traceback.format_exc()}")
        raise

    total_ms = (datetime.now() - t0).total_seconds()
    log("[BRAIN]", f"COMPLETE in {total_ms:.2f}s total")
    return reply_text, audio_b64


# ── WebSocket ──────────────────────────────────────────────────────────────────
@app.websocket("/ws/voice")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    session_id = f"web_{datetime.now().strftime('%H%M%S')}"
    log("[WS]", f"Connection ACCEPTED | session_id='{session_id}'")

    brain_lock  = asyncio.Lock()
    ai_speaking = asyncio.Event()

    log("[WS]", "Opening Sarvam STT streaming connection (saaras:v3, gu-IN, 16000 Hz)")
    try:
        async with client_stt.speech_to_text_streaming.connect(
            model="saaras:v3", language_code="gu-IN", sample_rate=16000
        ) as sarvam_ws:
            log("[WS]", "Sarvam STT connection ESTABLISHED")

            _recv = _drop_ai = _drop_vad = _sent = 0
            commit_queue: asyncio.Queue = asyncio.Queue()

            # ── Task 1: Browser audio → Sarvam STT ───────────────────────────
            async def browser_to_sarvam():
                nonlocal _recv, _drop_ai, _drop_vad, _sent
                log("[AUDIO_TASK]", "browser_to_sarvam() started")
                try:
                    while True:
                        msg = await websocket.receive()

                        # JSON control messages
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
                                        f"recv={_recv} sent={_sent} drop_ai={_drop_ai} drop_vad={_drop_vad}")

                                elif ctrl_type == "commit_transcript":
                                    text = ctrl.get("text", "").strip()
                                    if text:
                                        log("[COMMIT]", f"Frontend committed: '{text}'")
                                        await commit_queue.put(text)
                                    else:
                                        log("[COMMIT]", "Empty commit — ignored")

                                else:
                                    log("[CTRL]", f"Unknown type='{ctrl_type}'")

                            except Exception as e:
                                log("[CTRL]", f"JSON parse error: {e}")
                            continue

                        # Raw PCM audio
                        if "bytes" in msg:
                            _recv += 1
                            raw = msg["bytes"]

                            # Always drop audio while AI is speaking (echo prevention)
                            if ai_speaking.is_set():
                                _drop_ai += 1
                                continue

                            # ROOT CAUSE FIX:
                            # Previously we dropped silent packets here with is_silent().
                            # This caused Sarvam to receive NO DATA during pauses, so it
                            # could not detect end-of-utterance — it only returned a
                            # transcript when the NEXT speech burst arrived, causing
                            # 12–34 second STT delays.
                            #
                            # Fix: forward ALL audio to Sarvam (speech + silence).
                            # Sarvam's own internal VAD then detects the pause and
                            # emits a transcript within 1–2 seconds of the user stopping.
                            # The frontend's client-side VAD still handles the commit
                            # timer as a fallback, but Sarvam will now fire is_final=True
                            # much faster since it sees the actual silence data.
                            try:
                                await sarvam_ws.transcribe(audio=base64.b64encode(raw).decode("utf-8"))
                                _sent += 1
                                if _sent % 50 == 1:
                                    rms = compute_rms(raw)
                                    log("[STT_SEND]", f"pkt#{_sent} sent | rms={rms:.0f} ({'speech' if rms > 300 else 'silence'})")
                            except Exception as e:
                                log("[STT_SEND]", f"FAILED: {e}")

                except WebSocketDisconnect:
                    log("[AUDIO_TASK]", f"WS disconnected | recv={_recv} sent={_sent}")
                except Exception as e:
                    log("[AUDIO_TASK]", f"Crashed: {e}\n{traceback.format_exc()}")

            # ── Task 2: Sarvam STT → frontend transcript events ───────────────
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

                        if is_final:
                            f_count += 1
                            log("[STT_RECV]", f"FINAL #{f_count}: '{transcript}'")
                            await commit_queue.put(transcript)
                        else:
                            p_count += 1
                            if p_count % 5 == 1:
                                log("[STT_RECV]", f"Partial #{p_count}: '{transcript}'")

                        await websocket.send_json({"type": "transcript", "text": transcript, "is_final": is_final})

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
                                reply_text, audio_b64 = await run_brain(session_id, sentence, websocket)
                            except Exception as e:
                                log("[BRAIN_CONSUMER]", f"run_brain() FAILED: {e}")
                                continue

                            await websocket.send_json({"type": "ai_reply", "text": reply_text, "audio": audio_b64})
                            log("[BRAIN_CONSUMER]", "ai_reply sent | Lock RELEASED")

                    except asyncio.CancelledError:
                        log("[BRAIN_CONSUMER]", "Cancelled — exiting")
                        break
                    except Exception as e:
                        log("[BRAIN_CONSUMER]", f"Unexpected error: {e}\n{traceback.format_exc()}")

            log("[WS]", "Launching 3 async tasks")
            await asyncio.gather(browser_to_sarvam(), sarvam_to_frontend(), brain_consumer())
            log("[WS]", "All tasks exited — session ending")

    except Exception as e:
        log("[WS]", f"Handler CRASHED: {e}\n{traceback.format_exc()}")


# ── Audio helpers ──────────────────────────────────────────────────────────────
def compute_rms(raw_bytes: bytes) -> float:
    """Return RMS energy of a raw Int16 PCM buffer."""
    if len(raw_bytes) < 2:
        return 0.0
    samples = struct.unpack(f"<{len(raw_bytes)//2}h", raw_bytes)
    return (sum(s * s for s in samples) / len(samples)) ** 0.5