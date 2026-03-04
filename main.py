import os
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, List
from dotenv import load_dotenv

import config
import prompts

# LangChain Imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from calendar_tool import (check_calendar_availability, book_appointment,cancel_appointment,reschedule_appointment, suggest_next_available_slot)

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
tools = [check_calendar_availability, book_appointment,cancel_appointment,reschedule_appointment,suggest_next_available_slot]
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

        system_content = prompts.get_system_prompt(today_date, day)

        chat_sessions[sid] = [
            SystemMessage(content=system_content)
        ]
    
    # Retrieve existing history and add the new user message
    history = chat_sessions[sid]
    history.append(HumanMessage(content=request.text))
    
    # Step 1: Ask Gemini what to do (It might respond with text OR a tool call)
    ai_msg = llm_with_tools.invoke(history)
    
    # Step 2: Handle Tool Calls (The "Acting" part)
    while ai_msg.tool_calls:
        history.append(ai_msg)
        
        for tool_call in ai_msg.tool_calls:
            # Route to the correct tool based on the name Gemini selected
            tool_name = tool_call["name"]
            args = tool_call["args"]

            print("TOOL CALL ARGS:",tool_name, args)
            
            if tool_name == "check_calendar_availability":
                observation = check_calendar_availability(**args)
            elif tool_name == "book_appointment":
                observation = book_appointment(**args)
            elif tool_name == "cancel_appointment":
                observation = cancel_appointment(**args)
            elif tool_name == "reschedule_appointment":
                observation = reschedule_appointment(**args)
            elif tool_name == "suggest_next_available_slot":
                observation = suggest_next_available_slot(**args)
            else:
                observation = "Error: Tool not found."
    
            # Feed the result back to Gemini
            history.append(
                ToolMessage(
                    content=str(observation),
                    tool_call_id=tool_call["id"]
                )
            )
        
        # IMPORTANT
        ai_msg = llm_with_tools.invoke(history)

    # Now we have the final text response
    history.append(ai_msg)
    
    return {"reply": ai_msg.content}
    
    