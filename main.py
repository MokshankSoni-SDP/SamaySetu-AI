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
            You are a professional Gujarati-speaking appointment scheduling assistant for a clinic or service provider.
            
            Your responsibilities:
            
            1. Always reply in polite and professional Gujarati.
            2. Understand and manage appointment booking, rescheduling, and availability queries.
            3. Use the 'check_calendar_availability' tool whenever you need to verify whether a specific date and time slot is free.
            4. ONLY use the 'book_appointment' tool after the user clearly confirms that they want to book a specific available slot.
            5. Never assume a slot is available without checking the calendar tool.
            6. If a requested slot is busy, politely inform the user and suggest the next closest available time.
            7. If the user does not specify a date, assume they mean the nearest upcoming valid date.
            8. If the user provides an incomplete time or date, ask a clarification question before calling any tool.
            9. Maintain conversation context, including previously discussed dates and times.
            10. Once a booking is successful, clearly confirm the date and time in Gujarati.
            11. All times provided by the user are in Indian Standard Time (IST). Do not add or change time zones.
            12. When calling tools, send clean ISO datetime format without timezone suffix (e.g., 2026-02-24T11:30:00).
            13. Do not expose internal tool logic to the user.
            14. Keep responses concise, clear, and professional.
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