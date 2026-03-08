import os
import json
import base64
import asyncio
import traceback
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from sarvamai import AsyncSarvamAI, SarvamAI
from dotenv import load_dotenv
from typing import Dict, List

import prompts
from calendar_tool import *
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from datetime import datetime

# ── Logging helper ────────────────────────────────────────────────────────────
def log(step: str, msg: str):
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{ts}] {step}: {msg}")

# ── Setup & Environment ────────────────────────────────────────────────────────
load_dotenv()
app = FastAPI()

SARVAM_API_KEY = os.getenv('SARVAM_API_KEY')
client_stt = AsyncSarvamAI(api_subscription_key=SARVAM_API_KEY)
client_tts = SarvamAI(api_subscription_key=SARVAM_API_KEY)

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1)
tools = [check_calendar_availability, book_appointment, cancel_appointment, reschedule_appointment, suggest_next_available_slot]
llm_with_tools = llm.bind_tools(tools)

app.mount("/static", StaticFiles(directory="static"), name="static")

chat_sessions: Dict[str, List] = {}
MAX_HISTORY = 5

# ── Token usage logger ─────────────────────────────────────────────────────────
def print_token_usage(msg, step_name):
    usage = msg.response_metadata.get("token_usage", {})
    if usage:
        print(f"--- 📊 Token Usage: {step_name} ---")
        print(f"Prompt Tokens:     {usage.get('prompt_tokens')}")
        print(f"Completion Tokens: {usage.get('completion_tokens')}")
        print(f"Total Tokens:      {usage.get('total_tokens')}")
        print(f"-----------------------------------\n")

# ── Brain ──────────────────────────────────────────────────────────────────────
async def run_brain(session_id: str, user_text: str, websocket: WebSocket):
    today = datetime.now().strftime("%Y-%m-%d")
    day   = datetime.now().strftime("%A")

    log("[STEP 5] BRAIN", f"run_brain() called | session='{session_id}' | user_text='{user_text}'")

    if session_id not in chat_sessions:
        log("[STEP 5a] BRAIN", "New session — creating with system prompt")
        system_content = prompts.get_system_prompt(today, day)
        chat_sessions[session_id] = [SystemMessage(content=system_content)]
    else:
        log("[STEP 5a] BRAIN", f"Existing session | history len={len(chat_sessions[session_id])}")

    history = chat_sessions[session_id]
    history.append(HumanMessage(content=user_text))

    recent_history = [history[0]] + history[-MAX_HISTORY:]
    log("[STEP 5b] LLM", f"Invoking LLM with {len(recent_history)} messages")

    try:
        ai_msg = llm_with_tools.invoke(recent_history)
        log("[STEP 5c] LLM", f"LLM responded | tool_calls={len(ai_msg.tool_calls)} | preview='{str(ai_msg.content)[:80]}'")
        print_token_usage(ai_msg, "Initial Reasoning")
    except Exception as e:
        log("[STEP 5c] LLM", f"LLM FAILED: {e}\n{traceback.format_exc()}")
        raise

    tool_iteration = 0
    while ai_msg.tool_calls:
        tool_iteration += 1
        log("[STEP 6] TOOLS", f"Tool iteration #{tool_iteration} — {len(ai_msg.tool_calls)} tool(s)")
        history.append(ai_msg)

        for tool_call in ai_msg.tool_calls:
            tool_name = tool_call["name"]
            args      = tool_call["args"]
            log("[STEP 6a] TOOL_CALL", f"Executing: {tool_name} | args={args}")

            await websocket.send_json({"type": "tool_call", "name": tool_name, "args": args, "status": "running"})

            try:
                func        = globals()[tool_name]
                observation = func(**args)
                status      = "ok"
                log("[STEP 6b] TOOL_RESULT", f"'{tool_name}' OK | preview='{str(observation)[:120]}'")
            except Exception as e:
                observation = f"Error: {e}"
                status      = "error"
                log("[STEP 6b] TOOL_RESULT", f"'{tool_name}' FAILED: {e}")

            await websocket.send_json({"type": "tool_call", "name": tool_name, "args": args, "status": status, "observation": str(observation)})
            history.append(ToolMessage(content=str(observation), tool_call_id=tool_call["id"]))

        recent_history = [history[0]] + history[-MAX_HISTORY:]
        log("[STEP 6c] LLM", f"Re-invoking LLM after tool calls (iteration #{tool_iteration})")
        try:
            ai_msg = llm_with_tools.invoke(recent_history)
            log("[STEP 6c] LLM", f"Post-tool response | tool_calls={len(ai_msg.tool_calls)} | preview='{str(ai_msg.content)[:80]}'")
            print_token_usage(ai_msg, "Post-Tool Reasoning")
        except Exception as e:
            log("[STEP 6c] LLM", f"POST-TOOL LLM FAILED: {e}\n{traceback.format_exc()}")
            raise

    history.append(ai_msg)
    log("[STEP 7] TTS", f"Final answer ready. Sending to TTS. preview='{str(ai_msg.content)[:120]}'")

    try:
        tts_res  = client_tts.text_to_speech.convert(
            text=ai_msg.content,
            target_language_code="gu-IN",
            model="bulbul:v3",
            speaker="simran"
        )
        audio_b64 = tts_res.audios[0]
        log("[STEP 7] TTS", f"TTS OK | audio_b64 len={len(audio_b64)} chars")
    except Exception as e:
        log("[STEP 7] TTS", f"TTS FAILED: {e}\n{traceback.format_exc()}")
        raise

    return ai_msg.content, audio_b64


# ── WebSocket ──────────────────────────────────────────────────────────────────
@app.websocket("/ws/voice")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    session_id = f"web_{datetime.now().strftime('%H%M%S')}"
    log("[STEP 0] WS", f"Connection ACCEPTED | session_id='{session_id}'")

    brain_lock = asyncio.Lock()
    ai_speaking = asyncio.Event()

    log("[STEP 0a] WS", "Opening Sarvam STT streaming connection (saaras:v3, gu-IN, 16000 Hz)")
    try:
        async with client_stt.speech_to_text_streaming.connect(
            model="saaras:v3", language_code="gu-IN", sample_rate=16000
        ) as sarvam_ws:
            log("[STEP 0b] WS", "Sarvam STT connection ESTABLISHED")

            _packet_recv = 0
            _packet_dropped_ai = 0
            _packet_dropped_vad = 0
            _packet_sent_sarvam = 0

            # ── Queue: frontend commits land here for the brain task to consume ──
            # This decouples the fast audio path from the slow LLM path cleanly.
            commit_queue: asyncio.Queue = asyncio.Queue()

            # ── Task 1: Browser → Sarvam STT ─────────────────────────────────
            async def browser_to_sarvam():
                nonlocal _packet_recv, _packet_dropped_ai, _packet_dropped_vad, _packet_sent_sarvam
                log("[STEP 1] AUDIO_TASK", "browser_to_sarvam() started")
                try:
                    while True:
                        msg = await websocket.receive()

                        # ── Text/JSON control messages ────────────────────────
                        if "text" in msg:
                            try:
                                ctrl = json.loads(msg["text"])
                                ctrl_type = ctrl.get("type", "unknown")

                                if ctrl_type == "ai_speaking_start":
                                    ai_speaking.set()
                                    log("[CTRL]", "ai_speaking_start → mic MUTED")

                                elif ctrl_type == "ai_speaking_end":
                                    ai_speaking.clear()
                                    log("[CTRL]", f"ai_speaking_end → mic UN-MUTED | stats: recv={_packet_recv} sent={_packet_sent_sarvam} drop_ai={_packet_dropped_ai} drop_vad={_packet_dropped_vad}")

                                elif ctrl_type == "commit_transcript":
                                    # ── KEY FIX: Frontend detected silence and committed a partial ──
                                    # This is how we bridge the gap when Sarvam never sends is_final=true.
                                    text = ctrl.get("text", "").strip()
                                    if text:
                                        log("[COMMIT]", f"Frontend committed transcript: '{text}'")
                                        await commit_queue.put(text)
                                    else:
                                        log("[COMMIT]", "Received empty commit_transcript — ignoring")

                                else:
                                    log("[CTRL]", f"Unknown control type='{ctrl_type}'")

                            except Exception as e:
                                log("[CTRL]", f"Failed to parse JSON: {e} | raw='{msg.get('text','')[:80]}'")
                            continue

                        # ── Raw PCM audio bytes ───────────────────────────────
                        if "bytes" in msg:
                            _packet_recv += 1
                            raw = msg["bytes"]

                            if ai_speaking.is_set():
                                _packet_dropped_ai += 1
                                if _packet_dropped_ai % 20 == 1:
                                    log("[VAD]", f"Pkt #{_packet_recv} DROPPED (AI speaking mute)")
                                continue

                            if is_silent(raw):
                                _packet_dropped_vad += 1
                                if _packet_dropped_vad % 20 == 1:
                                    log("[VAD]", f"Pkt #{_packet_recv} DROPPED (silence)")
                                continue

                            #log("[STEP 3] STT_SEND", f"Pkt #{_packet_recv} → Sarvam | size={len(raw)} bytes")
                            try:
                                await sarvam_ws.transcribe(audio=base64.b64encode(raw).decode("utf-8"))
                                _packet_sent_sarvam += 1
                                #log("[STEP 3] STT_SEND", f"Pkt #{_packet_recv} sent (total={_packet_sent_sarvam})")
                            except Exception as e:
                                log("[STEP 3] STT_SEND", f"FAILED to send to Sarvam: {e}")

                except WebSocketDisconnect:
                    log("[STEP 1] AUDIO_TASK", f"WS disconnected | recv={_packet_recv} sent={_packet_sent_sarvam}")
                except Exception as e:
                    log("[STEP 1] AUDIO_TASK", f"Crashed: {e}\n{traceback.format_exc()}")

            # ── Task 2: Sarvam STT responses → forward transcript to frontend ─
            async def sarvam_to_frontend():
                """
                Forwards Sarvam transcripts to the browser.
                - Partials: browser uses them to update the live transcript display
                  and arm its silence-commit timer.
                - Finals (if Sarvam sends them): browser commits immediately.
                
                NOTE: In practice saaras:v3 over browser WS often only sends
                partials. The frontend's silence timer handles the commit path.
                The `commit_queue` (filled by browser_to_sarvam on commit_transcript
                messages) drives the brain.
                """
                log("[STEP 4] STT_RECV", "sarvam_to_frontend() started")
                try:
                    partial_count = 0
                    final_count   = 0

                    async for response in sarvam_ws:
                        log("[STEP 4] STT_RECV", f"Raw Sarvam response | type={getattr(response,'type','N/A')}")

                        if not (hasattr(response, "type") and response.type == "data"):
                            continue

                        transcript = getattr(response.data, 'transcript', "").strip()
                        is_final   = getattr(response.data, 'is_final', False)

                        if not transcript:
                            continue

                        if is_final:
                            final_count += 1
                            log("[STEP 4] STT_RECV", f"FINAL #{final_count}: '{transcript}'")
                        else:
                            partial_count += 1
                            if partial_count % 5 == 1:
                                log("[STEP 4] STT_RECV", f"Partial #{partial_count}: '{transcript}'")

                        # Forward to browser (browser decides what to do with it)
                        await websocket.send_json({"type": "transcript", "text": transcript, "is_final": is_final})

                        # If Sarvam itself says final, also push to commit_queue
                        # so the brain fires immediately without waiting for silence timeout
                        if is_final:
                            log("[STEP 4] STT_RECV", f"Sarvam sent is_final=True — pushing to commit_queue")
                            await commit_queue.put(transcript)

                except Exception as e:
                    log("[STEP 4] STT_RECV", f"Crashed: {e}\n{traceback.format_exc()}")

            # ── Task 3: Brain consumer — processes committed sentences ─────────
            async def brain_consumer():
                """
                Drains commit_queue and runs the LLM+TTS pipeline for each sentence.
                Uses brain_lock to prevent overlapping calls.
                """
                log("[BRAIN_CONSUMER]", "brain_consumer() started — waiting for commits")
                while True:
                    try:
                        sentence = await commit_queue.get()
                        log("[BRAIN_CONSUMER]", f"Dequeued sentence: '{sentence}'")

                        if brain_lock.locked():
                            log("[BRAIN_CONSUMER]", "Brain BUSY — skipping this sentence")
                            continue

                        async with brain_lock:
                            log("[BRAIN_CONSUMER]", "Brain lock ACQUIRED")
                            await websocket.send_json({"type": "processing_start"})

                            try:
                                reply_text, audio_b64 = await run_brain(session_id, sentence, websocket)
                                log("[BRAIN_CONSUMER]", f"run_brain() done | preview='{str(reply_text)[:80]}'")
                            except Exception as e:
                                log("[BRAIN_CONSUMER]", f"run_brain() FAILED: {e}")
                                continue

                            await websocket.send_json({"type": "ai_reply", "text": reply_text, "audio": audio_b64})
                            log("[BRAIN_CONSUMER]", "ai_reply sent to frontend")
                            log("[BRAIN_CONSUMER]", "Brain lock RELEASED")

                    except asyncio.CancelledError:
                        log("[BRAIN_CONSUMER]", "Task cancelled — exiting")
                        break
                    except Exception as e:
                        log("[BRAIN_CONSUMER]", f"Unexpected error: {e}\n{traceback.format_exc()}")

            log("[STEP 0c] WS", "Launching 3 async tasks: browser_to_sarvam, sarvam_to_frontend, brain_consumer")
            await asyncio.gather(
                browser_to_sarvam(),
                sarvam_to_frontend(),
                brain_consumer(),
            )
            log("[STEP 0c] WS", "All tasks exited — session ending")

    except Exception as e:
        log("[STEP 0] WS", f"Handler CRASHED: {e}\n{traceback.format_exc()}")


# ── VAD helper ─────────────────────────────────────────────────────────────────
def is_silent(raw_bytes: bytes, threshold: int = 300) -> bool:
    import struct
    if len(raw_bytes) < 2:
        return True
    samples = struct.unpack(f"<{len(raw_bytes)//2}h", raw_bytes)
    rms = (sum(s * s for s in samples) / len(samples)) ** 0.5
    return rms < threshold