import os
import re
import json
import base64
import asyncio
import struct
import traceback
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from sarvamai import AsyncSarvamAI
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
client_stt = AsyncSarvamAI(api_subscription_key=SARVAM_API_KEY)
client_tts = AsyncSarvamAI(api_subscription_key=SARVAM_API_KEY)

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

    log("[BRAIN]", f"START | session='{session_id}' | text='{user_text}'")

    if session_id not in chat_sessions:
        log("[BRAIN]", "New session — initialising with system prompt")
        chat_sessions[session_id] = [SystemMessage(content=prompts.get_system_prompt(today, day))]
    else:
        log("[BRAIN]", f"Existing session | history len={len(chat_sessions[session_id])}")

    history = chat_sessions[session_id]
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
        log("[TOOLS]", f"Iteration #{tool_iteration} — {len(ai_msg.tool_calls)} tool(s)")
        history.append(ai_msg)

        async def execute_tool(tool_call):
            tname = tool_call["name"]
            targs = tool_call["args"]
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

            _recv = _drop_ai = _sent = 0
            commit_queue: asyncio.Queue = asyncio.Queue()

            # ── Task 1: Browser audio → Sarvam STT ───────────────────────────
            async def browser_to_sarvam():
                nonlocal _recv, _drop_ai, _sent
                log("[AUDIO_TASK]", "browser_to_sarvam() started")
                try:
                    while True:
                        msg = await websocket.receive()

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
                                        await commit_queue.put(text)
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
                                continue
                            try:
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
                        if is_final:
                            f_count += 1
                            log("[STT_RECV]", f"FINAL #{f_count}: '{transcript}'")
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