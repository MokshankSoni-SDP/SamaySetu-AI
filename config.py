#--------------calendar_tool----------------
BUSINESS_START_HOUR = 9   # 9:00 AM
BUSINESS_END_HOUR   = 18

DEFAULT_APPOINTMENT_DURATION = 30  # In minutes

CALENDAR_TIMEZONE = 'Asia/Kolkata'


#--------------LLM Selection----------------
# Options: 'groq', 'nvidia'
LLM_PROVIDER = 'nvidia'

# --- NVIDIA Models ---
NVIDIA_API_KEY = "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" 
NVIDIA_MODEL_NAME = "openai/gpt-oss-20b" 

NVIDIA_SMALL_API_KEY = "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" # You can change this to a different API key
NVIDIA_SMALL_MODEL_NAME = "openai/gpt-oss-20b"

# --- Groq Models ---
GROQ_MODEL_NAME = "llama-3.3-70b-versatile"
GROQ_SMALL_MODEL_NAME = "llama-3.1-8b-instant"
GROQ_SMALL_API_KEY = "" # Leave blank to use .env, or put your separate key here

#--------------main----------------
MAX_HISTORY         = 4
MAX_TOOL_ITERATIONS = 2

MIN_CHUNK_CHARS = 20

#-------- frames sent to Sarvam STT to warm it up before actual conversation
WARMUP_FRAMES = 10

#------------ max reconnect attempts to Sarvam STT on idle disconnect
MAX_RECONNECTS = 15

#----------- extra buffer after AI finishes speaking (prevents echo)
AI_POST_TTS_BUFFER = 0.90


#--------------FACTS_MODULE (RAG)----------------
# Qdrant local binary URL (run: ./qdrant in your terminal)
QDRANT_URL        = "http://localhost:6333"
QDRANT_COLLECTION = "knowledge_base"
EMBEDDING_MODEL   = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM     = 384

# Chunking
KNOWLEDGE_CHUNK_MIN_WORDS = 20
KNOWLEDGE_CHUNK_MAX_WORDS = 30

# RAG retrieval
FACTS_TOP_K = 3
