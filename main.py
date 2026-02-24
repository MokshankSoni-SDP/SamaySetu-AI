import os
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, List
from dotenv import load_dotenv

# LangChain Imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from calendar_tool import check_calendar_availability

# 1. Setup & Environment
load_dotenv()
gemini_api_key = os.getenv('GEMINI_API_KEY')
os.environ["GOOGLE_API_KEY"] = gemini_api_key

app = FastAPI()

# 2. Memory: In-memory dictionary to store history for each session
# Key: session_id (e.g., "user_123"), Value: List of Messages
chat_sessions: Dict[str, List] = {}

# 3. Model Initialization
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
# Bind tools so the brain knows it can "act"
llm_with_tools = llm.bind_tools([check_calendar_availability])

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
            SystemMessage(content="""
                You are a helpful Gujarati appointment assistant. 
                1. Always reply in Gujarati.
                2. Use the 'check_calendar_availability' tool to check slots.
                3. Assume today's date is 2026-02-24.
                4. If a slot is free, confirm it in Gujarati. If busy, suggest another time.
                5. Remember previous context (dates/times) discussed in the chat.
            """)
        ]
    
    # Retrieve existing history and add the new user message
    history = chat_sessions[sid]
    history.append(HumanMessage(content=request.text))
    
    # Step 1: Ask Gemini what to do (It might respond with text OR a tool call)
    ai_msg = llm_with_tools.invoke(history)
    
    # Step 2: Handle Tool Calls (The "Acting" part)
    if ai_msg.tool_calls:
        history.append(ai_msg) # Record the AI's intent to use a tool
        
        for tool_call in ai_msg.tool_calls:
            # Execute the actual Python function from calendar_tool.py
            observation = check_calendar_availability(**tool_call["args"])
            
            # Create a ToolMessage to feed the result back to the AI
            # This is critical for the AI to "see" the result
            tool_message = ToolMessage(
                content=observation,
                tool_call_id=tool_call["id"]
            )
            history.append(tool_message)
            
        # Step 3: Call the model again with the tool result to get the final Gujarati text
        final_response = llm.invoke(history)
        history.append(final_response)
        return {"reply": final_response.content}
    
    # If no tool was needed, just save the AI's direct response and return
    history.append(ai_msg)
    return {"reply": ai_msg.content}