# 📘 SamaySetu AI – Gujarati Voice Appointment Booking Assistant

## 🚀 Project Overview

SamaySetu AI is a Gujarati-speaking voice-based appointment booking assistant that:

- 🎤 Understands spoken Gujarati
- 📅 Checks real-time availability in Google Calendar
- ✅ Books appointments upon confirmation
- 🔊 Responds back in natural Gujarati speech
- 🧠 Retrieves business facts and data using Qdrant (RAG integration)

The system integrates:

- Speech-to-Text (Sarvam AI)
- LLM reasoning (Groq & NVIDIA NIM via LangChain)
- Google Calendar API
- Text-to-Speech (Sarvam AI)
- FastAPI backend (with WebSockets for low-latency streaming)
- Qdrant Vector Database for Knowledge Base (RAG)
- Web-based User Interface

---

## 🛠 Prerequisites

Before running the project, ensure you have the following:

* **Python 3.9+**
* **Google Cloud Project:** With the Google Calendar API enabled and service account json.
* **Sarvam AI API Key:** For STT and TTS services.
* **Groq API Key / NVIDIA API Key:** For language model reasoning.
* **Qdrant:** Local installation or Docker container running on port 6333.

---

# 🏗️ System Architecture / Workflow

```
User Speech via Browser (Gujarati)
        ↓
FastAPI WebSocket
        ↓
Sarvam STT (Streaming)
        ↓
LLM (Groq / NVIDIA via config.py) + LangChain Tools
   ├── Google Calendar Tool
   │   ├── check_calendar_availability()
   │   └── book_appointment()
   └── Facts Module (RAG)
       └── Qdrant Vector DB
        ↓
LLM generates Gujarati reply
        ↓
Sarvam TTS (Bulbul v3)
        ↓
Browser plays spoken response
```

---

# 📂 Project Structure

```
├── main.py              # FastAPI backend (APIs and WebSockets)
├── brain.py             # LLM logic, memory, and routing
├── config.py            # Central configuration (LLM models, API keys, etc.)
├── calendar_tool.py     # Google Calendar integration
├── modules/             # Additional modules (e.g., facts_module for RAG)
├── static/              # Frontend web UI (HTML, JS, CSS)
├── requirements.txt
├── .env                 # Environment variables
├── service_account.json # Google Service Account credentials
```

---

# 🧠 Core Files Explanation

### 1️⃣ `main.py`
- Initializes FastAPI & WebSockets.
- Handles Auth / Sessions for Tenants.
- Interfaces with STT/TTS services and passes context to `brain.py`.
- Exposes Admin APIs to configure the bot dynamically.

### 2️⃣ `brain.py`
- Constructs the LLM pipeline dynamically based on provider (`config.py`).
- Maintains memory state and session history.
- Performs conversation noise filtering and explicit language switches.
- Executes calendar and module tool calls intelligently.

### 3️⃣ `config.py`
- Stores easy-to-edit configuration variables.
- Swap between **NVIDIA NIM** and **Groq** free LLM endpoints with a single word.
- Editable model names and model-specific API keys.
- Calendar defaults and Qdrant settings.

---

# 🔑 Required API Keys & Credentials

You must create a `.env` file in the root directory for certain keys, while others can be easily swapped in `config.py`.

## 📄 Create `.env` file

```
SARVAM_API_KEY=your_sarvam_api_key
GROQ_SMALL_LLM=your_groq_small_key (Optional if you use .env fallback)
```

## 📄 `config.py` Setup

Inside `config.py`, provide the LLM provider, keys and models:

```python
LLM_PROVIDER = 'nvidia' # or 'groq'

NVIDIA_API_KEY = "your_nvidia_api_key"
NVIDIA_MODEL_NAME = "openai/gpt-oss-20b"
NVIDIA_SMALL_API_KEY = "your_nvidia_small_api_key"

GROQ_MODEL_NAME = "llama-3.3-70b-versatile"
# etc...
```

---

## 🔐 Required External Accounts

### 1️⃣ Groq API / NVIDIA API Catalog
- Sign up at Groq Console or NVIDIA API Catalog.
- Generate API keys.
- Place them in `config.py` or `.env` as required.
- Provides free tiers for testing LLaMA 3.3 and other powerful models.

---

### 2️⃣ Google Calendar API

Steps:

1. Create Service Account in Google Cloud
2. Enable Google Calendar API
3. Download JSON credentials
4. Rename file to `service_account.json` and place in project root.
5. In the Admin Dashboard of the app, connect the specific calendar.

---

### 3️⃣ Sarvam AI

- Create account at Sarvam AI
- Generate API subscription key
- Paste into `.env` (SARVAM_API_KEY)

### 4️⃣ Qdrant (Knowledge Base)

- Download and start Qdrant locally (e.g. via Docker `docker run -p 6333:6333 qdrant/qdrant` or via official binaries).
- Ensure it's running on `http://localhost:6333`.

---

# 🛠 Installation & Setup

## Step 1️⃣ Create Virtual Environment

```bash
python -m venv venv
```

Activate:

**Windows**
```bash
venv\Scripts\activate
```

**Mac/Linux**
```bash
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

Ensure Qdrant is running if you use the Facts module, then start the application server:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## 🔹 Step 2: Use the Web Interface

Open a browser and navigate to:

```
http://127.0.0.1:8000/
```

- Navigate to `/admin` to setup business data, toggle modules (Booking/Facts), upload knowledge, and connect your Google Calendar.
- Speak directly to the bot from the root page (`/`) patient view!
- To test available NVIDIA models outside the web app, you can run `python test_nvidia_models.py`.

---

# 💬 How to Communicate with the Project

## Voice Interaction (Web UI)

Speak naturally directly through your browser microphone:

Examples:

- "મને કાલે 11 વાગ્યે અપોઇન્ટમેન્ટ જોઈએ છે."
- "હા, બુક કરી દેજો."
- "11:30 નો સ્લોટ ચેક કરજો."
- "અમારા ક્લિનિકમાં કઈ કઈ સુવિધાઓ છે?" (RAG Fact retrieval)

The system:
- Understands the request
- Retrieves knowledge via Qdrant or checks Calendar availability
- Responds professionally via voice
- Books appointment upon confirmation

---

# 🧩 Technologies Used

| Component | Technology |
|------------|------------|
| Backend | FastAPI |
| WebSockets | Real-time audio streaming |
| LLM Reasoning | NVIDIA NIM / Groq APIs |
| Tool Orchestration | LangChain |
| Vector DB (RAG) | Qdrant |
| Embeddings | `sentence-transformers` |
| Calendar | Google Calendar API |
| STT | Sarvam AI (saaras:v3 streaming) |
| TTS | Sarvam AI (bulbul:v3) |
| Database | PostgreSQL (Local/Aiven) |

---

# ⚠️ Important Notes

- All times are handled in IST (Asia/Kolkata)
- Each appointment duration can be customized in the admin panel.
- Ensure microphone permissions are allowed in your browser.
- Requires active internet connection for LLM, STT, and TTS APIs.

---

# 📌 Future Improvements

- Persistent LLM conversation storage
- Appointment cancellation & rescheduling (Beta)
- Multi-user scaling optimizations
- Expanded RAG pipelines for massive knowledge bases

---

# 🎯 Final Outcome

SamaySetu AI demonstrates:

- Real-time Gujarati voice interaction
- Configurable LLM tool-calling architectures
- Seamless Google Calendar & dynamic Knowledge base automation
- Fast-paced audio stream handling in FastAPI
