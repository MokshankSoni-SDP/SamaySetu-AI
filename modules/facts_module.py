"""
modules/facts_module.py
-----------------------
FACTS_MODULE: RAG-based informational query handler.
Uses Qdrant (local, path-based) + sentence-transformers/all-MiniLM-L6-v2.

LATENCY OPTIMISATIONS APPLIED:
  FIX 1 — Embedding model loaded EAGERLY at import time (not lazily per-call).
           First import triggers load once; every subsequent call hits the cached instance.
  FIX 2 — Qdrant client initialised EAGERLY at import time.
  FIX 3 — Embedding results cached via lru_cache(maxsize=1024).
           Repeated queries (same wording) skip encode() entirely.
  FIX 4 — warmup() called from FastAPI lifespan so first real request is cold-start free.

Setup:
  pip install qdrant-client sentence-transformers
  (No Docker / binary needed — uses local path storage)

Collection: knowledge_base
  vector size: 384  |  distance: Cosine  |  payload: tenant_id, content
"""

from __future__ import annotations

import re
import traceback
import uuid
from functools import lru_cache
from typing import Optional, List

# ─────────────────────────────────────────────────────────────────────────────
# FIX 1 + FIX 2 — EAGER GLOBALS
# Loaded once when this module is first imported (at FastAPI startup via warmup).
# asyncio.to_thread() keeps the blocking encode() off the event loop but does NOT
# cause a reload — Python module-level variables are process-wide singletons.
# ─────────────────────────────────────────────────────────────────────────────

QDRANT_PATH     = "./qdrant_data"
COLLECTION_NAME = "knowledge_base"
VECTOR_SIZE     = 384

# These are set to None here and populated by _bootstrap() which is called at the
# bottom of this file so the module is fully usable after import.
_embedding_model = None
_qdrant_client   = None


def _bootstrap():
    """
    Called once at module load time.  Loads the embedding model and connects to
    Qdrant.  Failures are logged but do NOT prevent the app from starting — the
    FACTS tool will return a clean error message until the issue is resolved.
    """
    global _embedding_model, _qdrant_client

    # ── Embedding model ───────────────────────────────────────────────────────
    try:
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2",
            device="cpu",
        )
        print("[FACTS] ✓ Embedding model loaded (all-MiniLM-L6-v2, CPU)")
    except Exception as e:
        print(f"[FACTS] ✗ Embedding model load FAILED: {e}  "
              "(install sentence-transformers — FACTS_MODULE will be unavailable)")

    # ── Qdrant client ─────────────────────────────────────────────────────────
    try:
        from qdrant_client import QdrantClient
        _qdrant_client = QdrantClient(path=QDRANT_PATH)
        _ensure_collection(_qdrant_client)
        print(f"[FACTS] ✓ Qdrant initialised (path={QDRANT_PATH})")
    except Exception as e:
        print(f"[FACTS] ✗ Qdrant init FAILED: {e}  "
              "(install qdrant-client — FACTS_MODULE will be unavailable)")


def _ensure_collection(client):
    """Create the Qdrant collection if it doesn't exist yet."""
    from qdrant_client.models import Distance, VectorParams
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
        print(f"[FACTS] Created Qdrant collection '{COLLECTION_NAME}'")


# ─────────────────────────────────────────────────────────────────────────────
# FIX 3 — CACHED EMBEDDING
# lru_cache keeps the last 1024 (text → vector) pairs in memory.
# For a voice bot, common questions repeat constantly → most encode() calls
# become instant dict lookups (~0 ms) instead of ~300 ms CPU inference.
# ─────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1024)
def _cached_embed(text: str) -> tuple:
    """
    Returns embedding as a tuple (hashable, so lru_cache works).
    Raises RuntimeError if model failed to load.
    """
    if _embedding_model is None:
        raise RuntimeError(
            "Embedding model not loaded. "
            "Run: pip install sentence-transformers"
        )
    vec = _embedding_model.encode(text, normalize_embeddings=True)
    return tuple(vec.tolist())


def embed_text(text: str) -> List[float]:
    """Public embed helper — returns list[float] from cached tuple."""
    return list(_cached_embed(text))


# ─────────────────────────────────────────────────────────────────────────────
# FIX 4 — WARMUP HOOK (called from FastAPI lifespan)
# ─────────────────────────────────────────────────────────────────────────────

def warmup():
    """
    Pre-warm the embedding model and Qdrant client.
    Call this from FastAPI lifespan / startup so the first real request is not cold.

    Usage in main.py:
        @asynccontextmanager
        async def lifespan(app):
            await asyncio.to_thread(warmup_facts_module)
            yield
    """
    # Model + client are already loaded by _bootstrap() at import time.
    # This function just runs a dummy encode so the JIT / BLAS routines are warm.
    if _embedding_model is not None:
        _ = embed_text("warmup query")
        print("[FACTS] ✓ Warmup encode complete")
    else:
        print("[FACTS] ✗ Warmup skipped — model not available")


# ─────────────────────────────────────────────────────────────────────────────
# Chunking
# ─────────────────────────────────────────────────────────────────────────────

def is_heading(line: str) -> bool:
    line = line.strip()

    if not line or len(line.split()) > 10:
        return False

    # Ends with colon OR looks like title
    if line.endswith(":"):
        return True

    # Title case heuristic
    if line.istitle():
        return True

    # Keyword-based detection
    keywords = [
        "address", "location", "timing", "hours", "doctor",
        "services", "charges", "fees", "overview",
        "facilities", "contact", "clinic"
    ]

    lower = line.lower()
    if any(k in lower for k in keywords):
        return True

    return False


def chunk_text(text: str, max_words: int = 120) -> List[str]:
    """
    Advanced chunking:
    - Detect headings automatically
    - Group into sections
    - Ensure each chunk starts with heading
    - Preserve context during splits
    """

    lines = [l.strip() for l in text.split("\n") if l.strip()]

    sections = []
    current_heading = "General Information"
    current_content = []

    for line in lines:
        if is_heading(line):
            # Save previous section
            if current_content:
                sections.append((current_heading, " ".join(current_content)))
                current_content = []

            current_heading = line
        else:
            current_content.append(line)

    if current_content:
        sections.append((current_heading, " ".join(current_content)))

    # 🔹 Now chunk sections smartly
    chunks = []

    for heading, content in sections:
        words = content.split()

        if len(words) <= max_words:
            chunks.append(f"{heading}\n{content}")
        else:
            # Split but keep heading
            start = 0
            while start < len(words):
                sub_chunk = words[start:start + max_words]
                chunk_text = " ".join(sub_chunk)

                chunks.append(f"{heading}\n{chunk_text}")
                start += max_words

    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Index / delete knowledge
# ─────────────────────────────────────────────────────────────────────────────

def index_knowledge(tenant_id: str, content: str) -> int:
    """
    Chunks content, embeds each chunk, upserts into Qdrant.
    Returns number of chunks indexed.
    """
    if _qdrant_client is None:
        raise RuntimeError("Qdrant not initialised — FACTS_MODULE unavailable")

    from qdrant_client.models import PointStruct

    chunks = chunk_text(content)
    if not chunks:
        return 0

    points = []
    for chunk in chunks:
        vector = embed_text(chunk)
        points.append(PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload={"tenant_id": tenant_id, "content": chunk},
        ))

    _qdrant_client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"[FACTS] Indexed {len(points)} chunks for tenant={tenant_id}")
    return len(points)


def delete_tenant_knowledge(tenant_id: str):
    """Remove all Qdrant vectors for a tenant."""
    if _qdrant_client is None:
        return
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    _qdrant_client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=Filter(
            must=[FieldCondition(key="tenant_id", match=MatchValue(value=tenant_id))]
        ),
    )
    print(f"[FACTS] Deleted Qdrant vectors for tenant={tenant_id}")


# ─────────────────────────────────────────────────────────────────────────────
# Retrieval
# ─────────────────────────────────────────────────────────────────────────────

def retrieve_facts(tenant_id: str, query: str, top_k: int = 3) -> List[str]:
    """
    Embed query (cached), search Qdrant for top_k chunks for this tenant.

    Supports both qdrant-client < 1.7 (.search) and >= 1.7 (.query_points).
    """
    if _qdrant_client is None:
        raise RuntimeError("Qdrant not initialised — FACTS_MODULE unavailable")

    from qdrant_client.models import Filter, FieldCondition, MatchValue

    query_vector  = embed_text(query)   # ← FIX 3: cached
    tenant_filter = Filter(
        must=[FieldCondition(key="tenant_id", match=MatchValue(value=tenant_id))]
    )

    print(f"[FACTS] Searching for: '{query}' (tenant: {tenant_id})")

    # New API (qdrant-client >= 1.7)
    if hasattr(_qdrant_client, "query_points"):
        response = _qdrant_client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            query_filter=tenant_filter,
            limit=top_k,
            with_payload=True,
        )
        hits = response.points if hasattr(response, "points") else response
        return [h.payload["content"] for h in hits if h.payload and h.payload.get("content")]

    # Old API (qdrant-client < 1.7)
    results = _qdrant_client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        query_filter=tenant_filter,
        limit=top_k,
        with_payload=True,
    )
    return [h.payload["content"] for h in results if h.payload.get("content")]


# ─────────────────────────────────────────────────────────────────────────────
# LangChain-compatible tool
# ─────────────────────────────────────────────────────────────────────────────

# Tenant context injected by brain.py before every tool call (same pattern as calendar_tool)
tenant_context = None


def get_facts(query: str, phone_number: Optional[str] = None) -> str:
    """
    RAG tool: retrieves relevant facts from the knowledge base for the current tenant.

    The LLM should translate the user question to English before calling this tool
    for better embedding match quality.

    Args:
        query       : User's question (in English preferred).
        phone_number: Ignored — kept for uniform tool signature.
    """
    print(f"[FACTS_FLOW] get_facts called | query='{query}'")

    if tenant_context is None:
        print("[FACTS_FLOW] Error: tenant_context is None")
        return "Error: Facts module context not initialised."

    session = tenant_context.chat_sessions.get(tenant_context.session_id)
    if not session:
        print(f"[FACTS_FLOW] Error: no session for id='{tenant_context.session_id}'")
        return "Error: No active session for facts lookup."

    tenant_id = session.get("tenant_id")
    if not tenant_id:
        print("[FACTS_FLOW] Error: tenant_id missing from session")
        return "Error: Tenant ID not available."

    try:
        facts = retrieve_facts(tenant_id, query, top_k=3)
        print(f"[FACTS_FLOW] Retrieved {len(facts)} fact(s)")
        if not facts:
            return "No relevant information found in the knowledge base."
        return "FACTS:\n" + "\n---\n".join(facts)
    except RuntimeError as e:
        print(f"[FACTS_FLOW] RuntimeError: {e}")
        return f"Error: Facts module unavailable — {e}"
    except Exception as e:
        print(f"[FACTS_FLOW] Unknown error: {e}")
        traceback.print_exc()
        return f"Error retrieving facts: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# MODULE-LEVEL BOOTSTRAP — runs once at import
# ─────────────────────────────────────────────────────────────────────────────
_bootstrap()