import streamlit as st
import os
import base64
import io
import uuid
from datetime import datetime
from dotenv import load_dotenv
from streamlit_mic_recorder import mic_recorder

# Project Imports
import prompts
from calendar_tool import (check_calendar_availability, book_appointment, 
                           cancel_appointment, reschedule_appointment, suggest_next_available_slot)

# LangChain / Sarvam Imports
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from sarvamai import SarvamAI

# 1. Setup & Environment
load_dotenv()
SARVAM_API_KEY = os.getenv('SARVAM_API_KEY')
client_sarvam = SarvamAI(api_subscription_key=SARVAM_API_KEY)

# 2. Streamlit Session State (Memory)
if "session_id" not in st.session_state:
    st.session_state.session_id = f"web_{uuid.uuid4().hex[:6]}"
if "history" not in st.session_state:
    today_date = datetime.now().strftime("%Y-%m-%d")
    day = datetime.now().strftime("%A")
    system_content = prompts.get_system_prompt(today_date, day)
    st.session_state.history = [SystemMessage(content=system_content)]

# 3. Model Initialization
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1)
tools = [check_calendar_availability, book_appointment, cancel_appointment, 
         reschedule_appointment, suggest_next_available_slot]
llm_with_tools = llm.bind_tools(tools)

# --- UI LAYOUT ---
st.set_page_config(page_title="SamaySetu AI", page_icon="📅")
st.title("🎙️ SamaySetu AI")
st.subheader("Gujarati Voice Appointment Assistant")

# --- CORE FUNCTIONS ---

def get_ai_response(user_text):
    """Processes text through the LLM and Tools"""
    st.session_state.history.append(HumanMessage(content=user_text))
    
    # Simple sliding window for memory
    recent_history = [st.session_state.history[0]] + st.session_state.history[-5:]
    ai_msg = llm_with_tools.invoke(recent_history)
    
    while ai_msg.tool_calls:
        st.session_state.history.append(ai_msg)
        for tool_call in ai_msg.tool_calls:
            tool_name = tool_call["name"]
            args = tool_call["args"]
            
            # Tool Routing
            if tool_name == "check_calendar_availability": obs = check_calendar_availability(**args)
            elif tool_name == "book_appointment": obs = book_appointment(**args)
            elif tool_name == "cancel_appointment": obs = cancel_appointment(**args)
            elif tool_name == "reschedule_appointment": obs = reschedule_appointment(**args)
            elif tool_name == "suggest_next_available_slot": obs = suggest_next_available_slot(**args)
            else: obs = "Error: Tool not found."
            
            st.session_state.history.append(ToolMessage(content=str(obs), tool_call_id=tool_call["id"]))
        
        recent_history = [st.session_state.history[0]] + st.session_state.history[-5:]
        ai_msg = llm_with_tools.invoke(recent_history)
    
    st.session_state.history.append(ai_msg)
    return ai_msg.content

def speak_text(text):
    """Converts text to speech and plays it in the browser"""
    response = client_sarvam.text_to_speech.convert(
        text=text,
        target_language_code="gu-IN",
        model="bulbul:v3",
        speaker="simran"
    )
    audio_bytes = base64.b64decode(response.audios[0])
    st.audio(audio_bytes, format="audio/wav", autoplay=True)

# --- MAIN INTERACTION ---

# 1. Microphone Input
st.write("ચેટ શરૂ કરવા માટે માઇક્રોફોન બટન દબાવો:")
audio_data = mic_recorder(start_prompt="🎤 Start Recording", stop_prompt="🛑 Stop & Send", key='recorder')

if audio_data:
    # 2. STT (File-based)
    with st.spinner("તમારો અવાજ ઓળખી રહ્યો છું..."):
        # Save temp file
        with open("temp_audio.wav", "wb") as f:
            f.write(audio_data['bytes'])
        
        # Sarvam STT
        with open("temp_audio.wav", "rb") as f:
            stt_res = client_sarvam.speech_to_text.translate(
                file=f,
                model="saaras:v3"
            )
        
        user_query = stt_res.transcript
        st.info(f"તમે કહ્યું: {user_query}")

        # 3. Brain (LLM + Tools)
        with st.spinner("વિચારી રહ્યો છું..."):
            ai_reply = get_ai_response(user_query)
            st.success(f"AI: {ai_reply}")

        # 4. TTS (Playback)
        speak_text(ai_reply)

# --- CHAT HISTORY VIEW ---
if st.checkbox("Show Chat History"):
    for msg in st.session_state.history:
        if isinstance(msg, HumanMessage): st.text(f"User: {msg.content}")
        if isinstance(msg, SystemMessage): st.text("System Prompt Active")
        if hasattr(msg, 'content') and msg.content and not isinstance(msg, (HumanMessage, SystemMessage)):
            st.text(f"AI: {msg.content}")