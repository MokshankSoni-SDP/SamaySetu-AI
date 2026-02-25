import os
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, List
from dotenv import load_dotenv

# LangChain Imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from calendar_tool import (check_calendar_availability, book_appointment,cancel_appointment,reschedule_appointment)

from datetime import datetime
today_date = datetime.now().strftime("%Y-%m-%d")
day = datetime.now().strftime("%A")

# 1. Setup & Environment
load_dotenv()
gemini_api_key = os.getenv('GEMINI_API_KEY')
os.environ["GOOGLE_API_KEY"] = gemini_api_key

groq_api_key = os.getenv('GROQ_API_KEY')

app = FastAPI()

# 2. Memory: In-memory dictionary to store history for each session
# Key: session_id (e.g., "user_123"), Value: List of Messages
chat_sessions: Dict[str, List] = {}

# 3. Model Initialization
#llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
# Using Llama-3.3-70b-versatile for high reasoning & tool calling accuracy
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    groq_api_key=groq_api_key,
    temperature=0.1 # Kept low for consistent tool calling
)

# Bind tools so the brain knows it can "act"
tools = [check_calendar_availability, book_appointment,cancel_appointment,reschedule_appointment]
llm_with_tools = llm.bind_tools(tools)

# 4. Request Schema
class UserRequest(BaseModel):
    session_id: str  # Added to track different users/calls
    text: str

@app.post("/chat")
async def process_request(request: UserRequest):
    sid = request.session_id
    
    # Initialize history for new sessions
    if sid not in chat_sessions:
        chat_sessions[sid] = [
            SystemMessage(content=f"""
            You are a professional Gujarati appointment assistant.

            Today's date is {today_date} and day is {day}.
            All times are in Indian Standard Time (IST).

            ==============================
            STRICT OUTPUT RULES
            ==============================

            1. Always respond in polite, professional Gujarati.
            2. NEVER include internal logic, JSON, XML, function tags, or tool syntax in your spoken reply.
            3. When a tool is required:
               - Respond ONLY with valid JSON.
               - Do NOT include any explanation or extra text.
               - Format:
                 {{
                   "tool_name": "tool_name_here",
                   "arguments": {{ ... }}
                 }}

            4. After the tool result is returned to you, generate ONLY the final conversational Gujarati reply.
            5. Never mix tool JSON and conversational text in the same response.

            ==============================
            GENERAL BEHAVIOR RULES
            ==============================
            
            1. Interpret all dates and times relative to the current system date.
            2. If the user says:
               - "આજે" → interpret as current date.
               - "કાલે" → interpret as one day after current date.
               - "પરમદિવસે" → interpret as two days after current date.
               - Day names (e.g., Monday) → interpret as the next upcoming occurrence.

            ==============================
            FUNCTION USAGE RULES
            ==============================

            - Use 'check_calendar_availability' before booking or rescheduling.
            - Use 'book_appointment' ONLY after explicit confirmation.
            - Use 'reschedule_appointment' for changing appointments (requires old and new time).
            - Use 'cancel_appointment' for cancellation (ask for double confirmation first).
            - If required information is missing, ask clearly in Gujarati before calling any tool.
            - If a slot is busy, politely suggest an alternative.
            - Always use ISO format: YYYY-MM-DDTHH:MM:SS
            - Never include timezone offsets.

            Be precise. Be professional. Keep responses concise.
            """)
        ]
    
    # Retrieve existing history and add the new user message
    history = chat_sessions[sid]
    history.append(HumanMessage(content=request.text))
    
    # Step 1: Ask Gemini what to do (It might respond with text OR a tool call)
    ai_msg = llm_with_tools.invoke(history)
    
    # Step 2: Handle Tool Calls (The "Acting" part)
    # Inside process_request, replace Step 2 with this:
    if ai_msg.tool_calls:
        history.append(ai_msg)
        
        for tool_call in ai_msg.tool_calls:
            # Route to the correct tool based on the name Gemini selected
            tool_name = tool_call["name"]
            args = tool_call["args"]

            print("TOOL CALL ARGS:", args)
            
            if tool_name == "check_calendar_availability":
                observation = check_calendar_availability(**args)
            elif tool_name == "book_appointment":
                observation = book_appointment(**args)
            elif tool_name == "cancel_appointment":
                observation = cancel_appointment(**args)
            elif tool_name == "reschedule_appointment":
                observation = reschedule_appointment(**args)
            else:
                observation = "Error: Tool not found."
    
            # Feed the result back to Gemini
            history.append(ToolMessage(
                content=str(observation),
                tool_call_id=tool_call["id"]
            ))
            
        # Step 3: Call the model again with the tool result to get the final Gujarati text
        final_response = llm.invoke(history)
        history.append(final_response)
        return {"reply": final_response.content}
    
    # If no tool was needed, just save the AI's direct response and return
    history.append(ai_msg)
    return {"reply": ai_msg.content}