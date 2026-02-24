import os
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, List
from dotenv import load_dotenv

# LangChain Imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from calendar_tool import check_calendar_availability, book_appointment

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
tools = [check_calendar_availability, book_appointment]
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
            SystemMessage(content="""
                You are a helpful Gujarati appointment assistant. 
                1. Always reply in Gujarati.
                2. Use the 'check_calendar_availability' tool to check slots.
                3. ONLY use 'book_appointment' after the user explicitly confirms they want to book a free slot.
                4. Assume today's date is 2026-02-24.
                5. If a slot is free, confirm it in Gujarati. If busy, suggest another time.
                6. Remember previous context (dates/times) discussed in the chat.
                7. Once booked successfully, provide the confirmation in Gujarati.
                8. IMPORTANT: All user requests are in Indian Standard Time (IST) and the current country is also India.
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
            else:
                observation = "Error: Tool not found."
    
            # Feed the result back to Gemini
            history.append(ToolMessage(
                content=observation,
                tool_call_id=tool_call["id"]
            ))
            
        # Step 3: Call the model again with the tool result to get the final Gujarati text
        final_response = llm.invoke(history)
        history.append(final_response)
        return {"reply": final_response.content}
    
    # If no tool was needed, just save the AI's direct response and return
    history.append(ai_msg)
    return {"reply": ai_msg.content}