# 📘 SamaySetu AI – Gujarati Voice Appointment Booking Assistant

## 🚀 Project Overview

SamaySetu AI is a Gujarati-speaking voice-based appointment booking assistant that:

- 🎤 Understands spoken Gujarati
- 📅 Checks real-time availability in Google Calendar
- ✅ Books appointments upon confirmation
- 🔊 Responds back in natural Gujarati speech

The system integrates:

- Speech-to-Text (Sarvam AI)
- LLM reasoning (Google Gemini via LangChain)
- Google Calendar API
- Text-to-Speech (Sarvam AI)
- FastAPI backend
- Real-time voice agent

---

## 🛠 Prerequisites

Before running the project, ensure you have the following:

* **Python 3.9+**
* **Google Cloud Project:** With the Google Calendar API enabled.
* **Sarvam AI API Key:** For STT and TTS services.
* **Google Gemini API Key:** For the language model logic.

---


# 🏗️ System Architecture / Workflow

```
User Speech (Gujarati)
        ↓
Sarvam STT (Streaming)
        ↓
voice_agent.py
        ↓
FastAPI Backend (/chat endpoint)
        ↓
Gemini LLM (Tool Calling via LangChain)
        ↓
Google Calendar Tool
   ├── check_calendar_availability()
   └── book_appointment()
        ↓
LLM generates Gujarati reply
        ↓
Sarvam TTS (Bulbul v3)
        ↓
User hears spoken response
```

---

# 📂 Project Structure

```
├── main.py              # FastAPI backend (LLM + tool logic)
├── calendar_tool.py     # Google Calendar integration
├── voice_agent.py       # Real-time voice interface
├── requirements.txt
├── .env                 # Environment variables (to be created)
├── service_account.json # Google Service Account credentials
```

---

# 🧠 Core Files Explanation

### 1️⃣ `main.py`
- Initializes FastAPI
- Loads Gemini model (`gemini-2.5-flash`)
- Binds calendar tools
- Maintains session memory
- Handles `/chat` endpoint

### 2️⃣ `calendar_tool.py`
- Connects to Google Calendar
- Checks availability
- Books appointments
- Uses service account authentication
- Uses IST timezone handling

### 3️⃣ `voice_agent.py`
- Connects to Sarvam STT streaming
- Detects pause via VAD
- Sends transcript to backend
- Receives AI response
- Converts reply to Gujarati speech

---

# 🔑 Required API Keys & Credentials

You must create a `.env` file in the root directory.

## 📄 Create `.env` file

```
GEMINI_API_KEY=your_gemini_api_key
SARVAM_API_KEY=your_sarvam_api_key
CALENDER_ID=your_google_calendar_id
```

---

## 🔐 Required External Accounts

### 1️⃣ Google Gemini API
- Create project in Google Cloud
- Enable Gemini API
- Generate API Key
- Paste into `.env`

---

### 2️⃣ Google Calendar API

Steps:

1. Create Service Account in Google Cloud
2. Enable Google Calendar API
3. Download JSON credentials
4. Rename file to:

```
service_account.json
```

5. Place it in project root directory

6. Share your Google Calendar with the service account email  
Give **"Make changes to events"** permission.

---

### 3️⃣ Sarvam AI

- Create account at Sarvam AI
- Generate API subscription key
- Paste into `.env`

---

# 🛠 Installation & Setup

## Step 1️⃣ Create Virtual Environment

```bash
python -m venv venv
```

Activate:

**Windows**
```
venv\Scripts\activate
```

**Mac/Linux**
```
source venv/bin/activate
```

---

## Step 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Project

## 🔹 Step 1: Start Backend

```bash
uvicorn main:app --reload
```

Backend runs at:

```
http://127.0.0.1:8000
```

---

## 🔹 Step 2: Run Voice Agent

Open a new terminal and run:

```bash
python voice_agent.py
```

You will hear a greeting:

```
નમસ્તે! હું સમયસેતુ AI છું...
```

Then speak in Gujarati.

---

# 💬 How to Communicate with the Project

## 1️⃣ Voice Interaction (Recommended)

Speak naturally in Gujarati:

Examples:

- "મને કાલે 11 વાગ્યે અપોઇન્ટમેન્ટ જોઈએ છે."
- "હા, બુક કરી દેજો."
- "11:30 નો સ્લોટ ચેક કરજો."

The system:
- Understands the request
- Checks Google Calendar
- Responds professionally
- Books appointment upon confirmation

---

## 2️⃣ API Testing via Postman / Curl

Send POST request to:

```
http://127.0.0.1:8000/chat
```

Body:

```json
{
  "session_id": "test_user",
  "text": "મને 11 વાગ્યાનો સ્લોટ જોઈએ છે."
}
```

Response:

```json
{
  "reply": "11 વાગ્યાનો સ્લોટ ઉપલબ્ધ છે..."
}
```

---

# 🧩 Technologies Used

| Component | Technology |
|------------|------------|
| Backend | FastAPI |
| LLM | Google Gemini (gemini-2.5-flash) |
| Tool Orchestration | LangChain |
| Calendar | Google Calendar API |
| STT | Sarvam AI (saaras:v3) |
| TTS | Sarvam AI (bulbul:v3) |
| Async HTTP | httpx |
| Audio Handling | PyAudio, sounddevice |

---

# ⚠️ Important Notes

- All times are handled in IST (Asia/Kolkata)
- Each appointment duration is 30 minutes
- Session memory is stored in-memory (not persistent)
- Requires active internet connection for APIs
- Free-tier Gemini has request limits

---

# 📌 Future Improvements

- Persistent database storage
- Appointment cancellation & rescheduling
- Streaming TTS for lower latency
- Local LLM support (Ollama)
- Multi-user scaling
- Authentication & role management

---

# 🎯 Final Outcome

SamaySetu AI demonstrates:

- Real-time Gujarati voice interaction
- Tool-based LLM reasoning
- Calendar automation
- Production-style conversational AI architecture
- End-to-end speech-to-speech assistant system
