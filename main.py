from fastapi import FastAPI
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from calendar_tool import check_calendar_availability
import os
from dotenv import load_dotenv

# Load environment variables From project root
env_path = '.env'
load_dotenv(dotenv_path=env_path)

gemini_api_key = os.getenv('GEMINI_API_KEY')

app = FastAPI()
os.environ["GOOGLE_API_KEY"] = gemini_api_key

# Define the LLM with Tools
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
# Bind the function so Gemini knows it can check the calendar
llm_with_tools = llm.bind_tools([check_calendar_availability])

class UserRequest(BaseModel):
    text: str

@app.post("/chat")
async def process_request(request: UserRequest):
    # System prompt to enforce Gujarati and appointment logic
    system_prompt = SystemMessage(content="""
        You are a helpful Gujarati appointment assistant. 
        1. Always reply in Gujarati.
        2. If a user asks for a slot, use the check_calendar_availability tool.
        3. Assume today's date is 2026-02-24 if the user doesn't specify a date.
        4. If the slot is free, confirm it in Gujarati. If busy, suggest they pick another time.
    """)
    
    user_msg = HumanMessage(content=request.text)
    
    # Step 1: Gemini decides whether to use a tool
    ai_msg = llm_with_tools.invoke([system_prompt, user_msg])
    
    # Step 2: If Gemini wants to call a tool, execute it
    if ai_msg.tool_calls:
        for tool_call in ai_msg.tool_calls:
            result = check_calendar_availability(**tool_call["args"])
            # Step 3: Send the calendar result back to Gemini for a final Gujarati response
            final_response = llm.invoke([
                system_prompt, 
                user_msg, 
                ai_msg, 
                HumanMessage(content=f"Tool Result: {result}")
            ])
            return {"reply": final_response.content}
    
    return {"reply": ai_msg.content}