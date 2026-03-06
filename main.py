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

# 1. Setup & Environment
load_dotenv()
app = FastAPI()

SARVAM_API_KEY = os.getenv('SARVAM_API_KEY')
client_stt = AsyncSarvamAI(api_subscription_key=SARVAM_API_KEY)
client_tts = SarvamAI(api_subscription_key=SARVAM_API_KEY)

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1)
tools = [check_calendar_availability, book_appointment, cancel_appointment, reschedule_appointment, suggest_next_available_slot]
llm_with_tools = llm.bind_tools(tools)

app.mount("/static", StaticFiles(directory="static"), name="static")

# --- RESTORED MEMORY LOGIC ---
chat_sessions: Dict[str, List] = {}
MAX_HISTORY = 5

def print_token_usage(msg, step_name):
    usage = msg.response_metadata.get("token_usage", {})
    if usage:
        print(f"--- 📊 Token Usage: {step_name} ---")
        print(f"Prompt Tokens: {usage.get('prompt_tokens')}")
        print(f"Completion Tokens: {usage.get('completion_tokens')}")
        print(f"Total Tokens: {usage.get('total_tokens')}")
        print(f"-----------------------------------\n")

# 2. Updated Shared Brain Logic with History Persistence
async def run_brain(session_id, user_text):
    today = datetime.now().strftime("%Y-%m-%d")
    day = datetime.now().strftime("%A")

    # Initialize history if session is new
    if session_id not in chat_sessions:
        system_content = prompts.get_system_prompt(today, day)
        chat_sessions[session_id] = [SystemMessage(content=system_content)]
    
    history = chat_sessions[session_id]
    history.append(HumanMessage(content=user_text))
    
    # Sliding window: Always keep System Message + the last N interactions
    recent_history = [history[0]] + history[-MAX_HISTORY:]
    
    print(f"DEBUG: Calling LLM for session {session_id}...")
    ai_msg = llm_with_tools.invoke(recent_history)
    print_token_usage(ai_msg, "Initial Reasoning")
    
    while ai_msg.tool_calls:
        history.append(ai_msg)
        for tool_call in ai_msg.tool_calls:
            tool_name = tool_call["name"]
            args = tool_call["args"]
            print(f"DEBUG: Tool Call: {tool_name}({args})")
            
            func = globals()[tool_name]
            observation = func(**args)
            
            history.append(ToolMessage(content=str(observation), tool_call_id=tool_call["id"]))
        
        recent_history = [history[0]] + history[-MAX_HISTORY:]
        ai_msg = llm_with_tools.invoke(recent_history)
        print_token_usage(ai_msg, "Post-Tool Reasoning")
    
    history.append(ai_msg)
    
    # Get TTS Audio
    tts_res = client_tts.text_to_speech.convert(
        text=ai_msg.content, target_language_code="gu-IN", 
        model="bulbul:v3", speaker="simran"
    )
    return ai_msg.content, tts_res.audios[0]

# 3. WebSocket Relay
@app.websocket("/ws/voice")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    # Unique ID for the web session
    session_id = f"web_{datetime.now().strftime('%H%M%S')}"
    print(f"DEBUG: Connection opened. Session: {session_id}")
    
    async with client_stt.speech_to_text_streaming.connect(
        model="saaras:v3", language_code="gu-IN", sample_rate=16000
    ) as sarvam_ws:
        
        async def browser_to_sarvam():
            try:
                while True:
                    data = await websocket.receive_bytes()
                    await sarvam_ws.transcribe(audio=base64.b64encode(data).decode("utf-8"))
            except (WebSocketDisconnect, Exception):
                pass

        async def sarvam_to_browser():
            try:
                async for response in sarvam_ws:
                    if hasattr(response, "type") and response.type == "data":
                        transcript = getattr(response.data, 'transcript', "").strip()
                        if transcript:
                            await websocket.send_json({"type": "transcript", "text": transcript})
                            
                            is_final = getattr(response.data, 'is_final', False)
                            # End of sentence detection
                            if is_final or transcript.endswith(('.', '?', '!', '।')):
                                print(f"DEBUG: Sentence Complete: {transcript}")
                                reply_text, audio_b64 = await run_brain(session_id, transcript)
                                
                                await websocket.send_json({
                                    "type": "ai_reply", 
                                    "text": reply_text, 
                                    "audio": audio_b64
                                })
            except Exception:
                print(f"DEBUG Error:\n{traceback.format_exc()}")

        await asyncio.gather(browser_to_sarvam(), sarvam_to_browser())