# 📘 SamaySetu AI — Multilingual Voice Appointment Booking Platform

> **"SamaySetu" (સમયસેતુ)** — _Time Bridge_ — a real-time, voice-first AI receptionist platform that bridges your patients/customers to your business, speaking the way they do.

---

## 🧭 What Is SamaySetu AI?

SamaySetu AI is a **production-ready, multi-tenant AI receptionist platform** designed for clinics, service businesses, and any appointment-driven organization. At its core, it gives businesses a fully automated, voice-based AI agent that:

- **Understands spoken Gujarati, Hindi, and English** — detecting the user's preferred language dynamically, even mid-conversation.
- **Books, cancels, and reschedules appointments** — by autonomously calling Google Calendar APIs via LLM tool-use.
- **Answers business-specific factual questions** — by searching a private vector knowledge base (RAG) built from admin-uploaded content.
- **Responds back as natural speech** — converting the LLM's text reply into high-quality audio, streamed back to the user's browser in real time.

It is **not a simple chatbot**. It is a full-stack autonomous agent with stateful memory, a two-LLM (main + small) pipeline, multi-module plug-and-play architecture, WebSocket streaming, and a complete multi-tenant admin system.

---

## 🔭 The Problem It Solves

Small clinics and service businesses spend significant staff time on phone calls just to manage appointments — checking availability, booking, rescheduling, and answering basic questions ("What are your fees?", "Where are you located?"). This is repetitive, time-consuming, and often leads to missed calls and lost revenue.

SamaySetu AI replaces this entirely. A patient opens a link, speaks in their language, and the AI receptionist handles everything autonomously — 24/7 — with full calendar integration and business knowledge at its disposal.

---

## 🏗️ High-Level Architecture

```
Patient's Browser (microphone audio)
        │
        ▼ (WebSocket — raw PCM audio chunks)
┌─────────────────────────────────────────────────┐
│                  main.py (FastAPI)               │
│  ┌──────────────────────────────────────────┐   │
│  │  Sarvam STT Streaming (saaras:v3)        │   │
│  │  → Converts audio chunks → Gujarati text │   │
│  └──────────────────────────────────────────┘   │
│           │                                      │
│           ▼ (transcript + session context)       │
│  ┌──────────────────────────────────────────┐   │
│  │              brain.py                    │   │
│  │  ┌─────────────────────────────────┐     │   │
│  │  │ Small LLM (memory extraction)   │     │   │
│  │  │ → Extracts intent, date, time,  │     │   │
│  │  │   language pref from user text  │     │   │
│  │  └─────────────────────────────────┘     │   │
│  │  ┌─────────────────────────────────┐     │   │
│  │  │ Main LLM (tool-calling agent)   │     │   │
│  │  │ → Decides what to do            │     │   │
│  │  │ → Calls tools if needed         │     │   │
│  │  │ → Generates reply text          │     │   │
│  │  └─────────────────────────────────┘     │   │
│  │         │ tool calls                      │   │
│  │  ┌──────┴──────────────────────────┐     │   │
│  │  │  Module Registry (tool router)  │     │   │
│  │  │  ├── BOOKING_MODULE             │     │   │
│  │  │  │   └── Google Calendar API   │     │   │
│  │  │  └── FACTS_MODULE              │     │   │
│  │  │      └── Qdrant Vector DB (RAG) │     │   │
│  │  └─────────────────────────────────┘     │   │
│  └──────────────────────────────────────────┘   │
│           │ (text reply)                         │
│  ┌──────────────────────────────────────────┐   │
│  │  Sarvam TTS Streaming (bulbul:v3)        │   │
│  │  → Converts text → MP3 audio chunks      │   │
│  └──────────────────────────────────────────┘   │
│           │ (WebSocket audio frames)             │
└─────────────────────────────────────────────────┘
        │
        ▼
Patient's Browser (plays audio response)
```

---

## ⚙️ Core System Components — Deep Dive

### The FastAPI Application Layer

The application entry point manages everything from the network boundary inward:

- **WebSocket voice sessions**: Raw PCM audio chunks stream in from the browser, fed directly to Sarvam's STT streaming endpoint. The transcript fires an async pipeline into `brain.py`.
- **Echo & noise filtering**: Before hitting the LLM, transcripts are checked using `is_noisy_transcript()` (detects garbled/repeated STT artifacts) and `is_echo_of_ai()` (detects when the STT accidentally picks up the AI's own spoken output).
- **Streaming TTS session (`StreamingTTSSession`)**: An async worker class that uses a queue to serialize TTS requests. It connects to Sarvam's WebSocket TTS endpoint, streams sentence-by-sentence (for lower latency), and falls back to REST TTS if the stream fails.
- **Multi-tenant admin REST APIs**: Full CRUD for bot configuration, appointment management, module toggles, knowledge base upload, and calendar connection.
- **Live preview chat**: Admins can test the bot's persona and language in text mode directly from the dashboard — no audio needed.
- **Auth system**: Separate session stores for customers (phone-based), admins (email+password+token), and superadmins (secret key). Tenant isolation enforced at every endpoint.

---

### The Agent Intelligence Layer

This is the cognitive core of SamaySetu AI. Every voice turn runs through a carefully orchestrated pipeline:

#### Two-LLM Pipeline

| LLM | Role | Model |
|---|---|---|
| **Small LLM** | Memory extraction — cheap, fast, zero-tool | `llama-3.1-8b-instant` (Groq) / NVIDIA NIM small |
| **Main LLM** | Conversation + tool orchestration — accurate, powerful | `llama-3.3-70b-versatile` (Groq) / NVIDIA NIM large |

**Why two LLMs?** Running the full 70B model for every simple state update (e.g., "user said 'tomorrow at 3'") is wasteful. The small LLM runs a pure JSON extraction task in ~200ms, keeping the structured memory state up to date. The main LLM then uses this clean state to reason and act correctly.

#### Memory State Machine

Every session maintains a structured memory object (`dict`) that the small LLM updates on each turn:

```json
{
  "intent": "book | cancel | reschedule | facts | query | none",
  "language_preference": "gu-IN | hi-IN | en-IN | null",
  "appointment": {
    "date": "YYYY-MM-DD",
    "time": "HH:MM",
    "duration": "minutes"
  },
  "reschedule": {
    "old_time": "YYYY-MM-DDTHH:MM:SS",
    "new_time": "YYYY-MM-DDTHH:MM:SS"
  },
  "pending_action": "waiting_for_confirmation | none"
}
```

This memory persists across the entire session. When a user gives partial info across multiple turns ("I want an appointment" → "tomorrow" → "3 PM"), the memory correctly accumulates these pieces without losing prior context.

#### Confirmation Safety Gate

Before any mutating tool call (`book_appointment`, `cancel_appointment`, `reschedule_appointment`), the system enforces a **mandatory confirmation gate**:
1. The main LLM is instructed to ask for explicit confirmation first.
2. `brain.py` independently intercepts any tool call to these mutating tools and cross-checks whether `pending_action == "waiting_for_confirmation"` and whether the user expressed a genuine "yes" (detected via multilingual yes-markers).
3. If confirmation is absent, the tool call is **blocked** and the user is asked to confirm again.

This is a safety-critical feature — no appointment can ever be booked without the user explicitly saying yes.

#### Language Detection & Switching

A zero-cost Python function (`detect_requested_language()`) runs on every transcript. It matches against Gujarati-script, Devanagari, and romanized language keyword patterns. If detected, the TTS speaker and STT language code switch immediately for the next turn — no LLM call required for this.

The Gujarati time normalizer (`normalize_gujarati_time()`) converts spoken Gujarati time expressions like "સવા ત્રણ" (quarter past three), "સાડા ચાર" (half past four), or "પોણા છ" (quarter to six) into ISO `HH:MM` format the calendar tools can use.

### The Plug-and-Play Module System

SamaySetu AI uses a **module registry pattern** that makes the bot's capabilities configurable per tenant.

#### Module Registry (`module_registry.py`)

- Resolves which modules are enabled for a tenant (from the DB).
- Dynamically assembles the tool list that gets bound to the main LLM.
- Tools are cached per tenant+module combination.

#### `BOOKING_MODULE`

Provides 5 LangChain tools wrapping `calendar_tool.py`:

| Tool | What It Does |
|---|---|
| `check_calendar_availability` | Checks if a specific slot is free |
| `book_appointment` | Creates a Google Calendar event |
| `cancel_appointment` | Removes an existing event |
| `reschedule_appointment` | Moves an existing event to a new time |
| `suggest_next_available_slot` | Scans ahead and returns free slots |

All calendar operations respect **business hours** (configurable per tenant, including multiple time periods per day), **IST timezone**, and **slot duration**.

#### `FACTS_MODULE` (RAG)

Provides the `get_facts` tool:
- Admin uploads raw text (clinic info, fees, services, etc.) from the admin panel.
- `main.py` chunks this text (20–30 words per chunk) and upserts it into **Qdrant** (local vector DB).
- On a user question, Qdrant performs **dense vector similarity search** using `sentence-transformers/all-MiniLM-L6-v2` embeddings (384-dim).
- Top-K retrieved chunks are returned to the LLM to compose a grounded answer.
- The LLM is **strictly forbidden** from answering factual business questions from its own training knowledge — it must use `get_facts` first.

---

### Dynamic System Prompt Builder

The system prompt is assembled dynamically on every call, composing:
- **Bot persona** — name, receptionist name, business description (from tenant DB config).
- **Date/time context** — today's date, day name, current IST time (injected to prevent past-slot booking).
- **Language configuration** — which language to default to, and the instruction to ask users for preference at conversation start.
- **Module sections** — enabled modules inject their full instruction sets; disabled modules inject a "not available" guard. This is how the same prompt builder powers both booking-only, facts-only, and combined tenants.
- **Memory usage rules** — detailed instructions on how to use vs. ignore the memory state correctly.
- **Safety rules** — confirmation gates, domain restrictions (no off-topic answers), knowledge boundaries.

---

### Multi-Tenant PostgreSQL Layer

The PostgreSQL schema (via `psycopg2`, not an ORM) includes tables for:

| Table | Purpose |
|---|---|
| `tenants` | Business accounts — name, type, status |
| `tenant_admins` | Admin credentials (SHA-256 hashed passwords) |
| `bot_configs` | Per-tenant bot settings — persona, hours, calendar |
| `users` | Customer records — phone number, name |
| `appointments` | Booked appointments — linked to tenant + user |
| `module_configs` | Per-tenant module on/off switches |
| `knowledge_chunks` | Uploaded RAG text (chunked, stored for re-indexing) |
| `calendar_tokens` | Per-tenant Google service account JSON + calendar ID |

All queries are tenant-scoped — a tenant can never see another tenant's data.

---

## Full Conversation Flow (Step-by-Step)

```
1. User opens browser → WebSocket connected to /ws/{tenant_id}/{phone}
2. User speaks → browser streams raw audio chunks every ~100ms
3. main.py pipes chunks to Sarvam STT streaming endpoint
4. STT returns partial transcripts; silence detection fires after timeout
5. Final transcript passed to brain.run_brain()
6. brain.py: noise/echo filter → drop if noise
7. brain.py: Small LLM → update memory JSON (async, parallel with main LLM)
8. brain.py: Main LLM (with bound tools) → analyze transcript + memory → generate reply
9. If LLM calls a tool:
   a. If mutating tool → check confirmation gate → block if not confirmed
   b. If confirmed → execute tool (calendar op or RAG search)
   c. Tool result injected back into LLM → LLM generates final text
10. Text reply → split into sentences → fed to StreamingTTSSession
11. Sarvam TTS streams MP3 audio chunks → WebSocket → browser plays audio
12. Session state (memory, history, lang) saved for next turn
```

---

## 📂 Project Structure

```
SamaySetu AI3/
├── main.py              # FastAPI server — WebSockets, REST APIs, auth, TTS streaming
├── brain.py             # LLM agent — memory, tool orchestration, language detection
├── prompts.py           # Dynamic system prompt builder (module-aware)
├── calendar_tool.py     # Google Calendar tool functions
├── config.py            # Global constants — timeouts, model names, RAG params
├── modules/
│   ├── module_registry.py   # Tool loader & per-tenant module resolver
│   └── facts_module.py      # Qdrant RAG search implementation
├── services/
│   ├── calendar_provider.py     # Calendar connection verification
│   └── module_request_email.py  # Email notification service
├── database/
│   ├── models.py   # Table creation (psycopg2 raw SQL)
│   ├── crud.py     # All DB read/write operations
│   └── db.py       # Connection pool setup
├── static/
│   ├── index.html       # Landing / entry
│   ├── customer.html    # Voice UI for end users
│   ├── admin.html       # Admin dashboard
│   └── superadmin.html  # Platform management panel
├── requirements.txt
├── .env                 # Secret keys (not committed)
└── service_account.json # Google service account credentials (not committed)
```

---

## 🛠 Prerequisites

- **Python 3.9+**
- **PostgreSQL** — Local or cloud (tested with Aiven PostgreSQL)
- **Google Cloud Project** — Calendar API enabled + service account JSON
- **Sarvam AI Account** — For STT (`saaras:v3`) and TTS (`bulbul:v3`) APIs
- **Groq API Key and/or NVIDIA NIM API Key** — For LLM inference
- **Qdrant** — Running locally (`http://localhost:6333`) for the FACTS module

---

## 🔑 Configuration

### `.env` File

```env
# Sarvam AI
SARVAM_API_KEY=your_sarvam_api_key

# LLM Provider: 'groq' or 'nvidia'
LLM_PROVIDER=groq

# Groq (used when LLM_PROVIDER=groq)
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL_NAME=llama-3.3-70b-versatile
GROQ_SMALL_API_KEY=your_groq_small_key
GROQ_SMALL_MODEL_NAME=llama-3.1-8b-instant

# NVIDIA NIM (used when LLM_PROVIDER=nvidia)
NVIDIA_API_KEY=your_nvidia_api_key
NVIDIA_MODEL_NAME=openai/gpt-oss-20b
NVIDIA_SMALL_API_KEY=your_nvidia_small_api_key
NVIDIA_SMALL_MODEL_NAME=meta/llama-3.1-8b-instruct

# PostgreSQL
DATABASE_URL=postgresql://user:pass@host:port/dbname

# Superadmin
SUPERADMIN_SECRET=your_secure_superadmin_key
```

### `config.py` (Tunable Parameters)

```python
MAX_HISTORY         = 4      # LLM conversation turns to keep
MAX_TOOL_ITERATIONS = 2      # Max tool call cycles per turn
WARMUP_FRAMES       = 10     # STT warmup audio frames
FACTS_TOP_K         = 3      # RAG top-k retrieved chunks
```

---

## 🚀 Installation & Running

### Step 1 — Create Virtual Environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate
```

### Step 2 — Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3 — Start Qdrant (for FACTS module)

```bash
docker run -p 6333:6333 qdrant/qdrant
# Or run the Qdrant binary directly
```

### Step 4 — Start the Server

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Step 5 — Access the App

| URL | Purpose |
|---|---|
| `http://localhost:8000/` | Customer voice interface |
| `http://localhost:8000/admin` | Admin dashboard |
| `http://localhost:8000/superadmin` | Superadmin panel |

---

## 🔐 Google Calendar Setup

1. Create a **Service Account** in Google Cloud Console.
2. Enable the **Google Calendar API** for your project.
3. Download the service account **JSON credentials**.
4. In the admin dashboard → **Calendar** tab → paste the JSON + your calendar ID.
5. Share your Google Calendar with the service account's email address (give it "Make changes to events" permission).

The bot will verify the connection and report success/failure immediately.

---

## 💬 Example Voice Interactions

**Appointment Booking:**
> "મને કાલે 11 વાગ્યે ડૉક્ટર સાથે અપોઇન્ટમેન્ટ જોઈએ છે."
> → AI checks availability → asks confirmation → books on Google Calendar

**Factual Query (RAG):**
> "અમારા ક્લિનિકમાં ફી કેટલી છે?"
> → AI calls `get_facts("consultation fee")` → answers from knowledge base

**Language Switch:**
> "Please speak in English."
> → AI switches language mid-conversation, no restart required

**Reschedule:**
> "11 વાગ્ય ની appointment 3 વાગ્ય ે ખસેડો."
> → AI identifies old + new time → confirms → reschedules via Calendar API

---

## 🧩 Technology Stack

| Component | Technology |
|---|---|
| Backend Framework | FastAPI (async) |
| Voice Streaming | WebSockets (raw PCM audio) |
| Speech-to-Text | Sarvam AI — `saaras:v3` (streaming) |
| Text-to-Speech | Sarvam AI — `bulbul:v3` (streaming) |
| LLM Inference | NVIDIA NIM / Groq (LLaMA 3.x models) |
| LLM Orchestration | LangChain (tool-calling, message schema) |
| Vector Database | Qdrant (local) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Calendar | Google Calendar API v3 (service account) |
| Database | PostgreSQL (`psycopg2`) |
| Retry Logic | `tenacity` |

---

## ⚠️ Important Notes

- All times are handled in **IST (Asia/Kolkata)**. The server injects the current IST time into the LLM prompt to prevent past-slot booking.
- Microphone permissions must be granted in the browser for the voice interface.
- The FACTS module requires Qdrant to be running before the server starts (it warms up the embedding model at startup).
- Appointment confirmation is a **hard safety gate** — the LLM cannot book/cancel/reschedule without explicit user consent, enforced at the code level, not just via prompting.
- Each tenant's data (users, appointments, bot config, knowledge) is fully isolated.

---

## 📌 Roadmap / Future Improvements

- [ ] Persistent conversation storage across sessions
- [ ] Appointment cancellation & rescheduling via full voice flow (beta testing)
- [ ] WhatsApp / Telegram channel integration
- [ ] Expanded RAG pipelines with re-ranking for larger knowledge bases
- [ ] Analytics dashboard — call volume, booking rates, peak hours
- [ ] Multi-language STT model selection per tenant

---

## 🎯 Summary

SamaySetu AI demonstrates a production-grade, end-to-end autonomous AI agent pipeline — combining real-time streaming audio I/O, a stateful two-LLM reasoning system, calendar automation, vector knowledge retrieval, and a full multi-tenant SaaS backend — all in a single deployable Python application.
