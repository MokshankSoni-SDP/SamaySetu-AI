"""
modules/facts_module.py
-----------------------
FACTS_MODULE: RAG-based informational query handler.
Uses Qdrant (local binary) + sentence-transformers/all-MiniLM-L6-v2.

Setup:
  1. Download Qdrant binary: https://github.com/qdrant/qdrant/releases
     Place it at ./qdrant or add to PATH.
  2. Run: ./qdrant  (starts on localhost:6333)
  3. This module connects automatically.

Collection: knowledge_base
  - vector size: 384
  - distance:    Cosine
  - payload:     tenant_id, content
"""

import re
import traceback
from typing import Optional, List

# ── Embedding model (loaded once, CPU) ───────────────────────────────────────
_embedding_model = None

def _get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
            print("[FACTS] Embedding model loaded (all-MiniLM-L6-v2, CPU)")
        except ImportError:
            raise RuntimeError(
                "[FACTS] sentence-transformers not installed. "
                "Run: pip install sentence-transformers"
            )
    return _embedding_model


# ── Qdrant client (lazy init) ─────────────────────────────────────────────────
_qdrant_client = None
QDRANT_PATH = "./qdrant_data"
COLLECTION_NAME = "knowledge_base"
VECTOR_SIZE = 384


def _get_qdrant_client():
    global _qdrant_client
    if _qdrant_client is None:
        try:
            from qdrant_client import QdrantClient
            # Changed from url=QDRANT_URL to path=QDRANT_PATH
            _qdrant_client = QdrantClient(path=QDRANT_PATH)
            
            _ensure_collection()
            print(f"[FACTS] Qdrant initialized in Local Mode (Storage: {QDRANT_PATH})")
        except ImportError:
            raise RuntimeError(
                "[FACTS] qdrant-client not installed. "
                "Run: pip install qdrant-client"
            )
        except Exception as e:
            raise RuntimeError(
                f"[FACTS] Failed to initialize Qdrant at {QDRANT_PATH}. "
                f"Error: {e}"
            )
    return _qdrant_client


def _ensure_collection():
    """Create the Qdrant collection if it doesn't exist."""
    from qdrant_client.models import Distance, VectorParams
    client = _qdrant_client
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
        print(f"[FACTS] Created Qdrant collection '{COLLECTION_NAME}'")
    else:
        print(f"[FACTS] Qdrant collection '{COLLECTION_NAME}' already exists")


# ── Chunking ──────────────────────────────────────────────────────────────────

def chunk_text(text: str, min_words: int = 20, max_words: int = 30) -> List[str]:
    """
    Splits raw text into chunks of 20-30 words.
    Tries to break at sentence boundaries first, then falls back to word count.
    """
    # Split on sentence-ending punctuation
    sentences = re.split(r'(?<=[.!?।\u0964])\s+', text.strip())
    chunks = []
    buffer_words = []

    for sentence in sentences:
        words = sentence.split()
        for word in words:
            buffer_words.append(word)
            if len(buffer_words) >= max_words:
                chunks.append(" ".join(buffer_words))
                buffer_words = []

    # Flush remaining words
    if buffer_words:
        if len(buffer_words) >= min_words or not chunks:
            chunks.append(" ".join(buffer_words))
        else:
            # Too short — merge with last chunk
            if chunks:
                chunks[-1] += " " + " ".join(buffer_words)
            else:
                chunks.append(" ".join(buffer_words))

    return [c for c in chunks if c.strip()]


# ── Embed text ────────────────────────────────────────────────────────────────

def embed_text(text: str) -> List[float]:
    model = _get_embedding_model()
    return model.encode(text, normalize_embeddings=True).tolist()


# ── Index knowledge for a tenant ──────────────────────────────────────────────

def index_knowledge(tenant_id: str, content: str) -> int:
    """
    Chunks content, embeds each chunk, and upserts into Qdrant.
    Returns number of chunks indexed.
    """
    from qdrant_client.models import PointStruct
    import uuid

    client = _get_qdrant_client()
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

    client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"[FACTS] Indexed {len(points)} chunks for tenant={tenant_id}")
    return len(points)


def delete_tenant_knowledge(tenant_id: str):
    """Remove all Qdrant vectors for a tenant."""
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    client = _get_qdrant_client()
    client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=Filter(
            must=[FieldCondition(key="tenant_id", match=MatchValue(value=tenant_id))]
        ),
    )
    print(f"[FACTS] Deleted all Qdrant vectors for tenant={tenant_id}")


# ── Query / retrieve ──────────────────────────────────────────────────────────

def retrieve_facts(tenant_id: str, query: str, top_k: int = 3) -> List[str]:
    """
    Embed query, search Qdrant for top_k chunks belonging to this tenant.
    Returns list of content strings.

    Supports both qdrant-client < 1.7 (.search) and >= 1.7 (.query_points).
    The local path-based client in newer versions only exposes query_points().
    """
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    client = _get_qdrant_client()
    query_vector = embed_text(query)

    print(f"[FACTS] Searching Qdrant for: '{query}' (tenant: {tenant_id})")

    tenant_filter = Filter(
        must=[FieldCondition(key="tenant_id", match=MatchValue(value=tenant_id))]
    )

    # Try new API first (qdrant-client >= 1.7 — works for both local and HTTP clients)
    if hasattr(client, "query_points"):
        from qdrant_client.models import NamedVector
        response = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            query_filter=tenant_filter,
            limit=top_k,
            with_payload=True,
        )
        # query_points returns a QueryResponse with a .points attribute
        hits = response.points if hasattr(response, "points") else response
        return [hit.payload["content"] for hit in hits if hit.payload and hit.payload.get("content")]

    # Fallback: old API (qdrant-client < 1.7)
    results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        query_filter=tenant_filter,
        limit=top_k,
        with_payload=True,
    )
    return [hit.payload["content"] for hit in results if hit.payload.get("content")]


# ── LangChain-compatible tool function ───────────────────────────────────────

# Tenant context injected by main.py (same pattern as calendar_tool)
tenant_context = None


def get_facts(query: str, phone_number: Optional[str] = None) -> str:
    """
    RAG tool: retrieves relevant facts from the knowledge base for the current tenant.
    Converts query to English internally for better embedding quality.
    """
    print(f"[FACTS_FLOW] get_facts called with query='{query}', phone_number='{phone_number}'")
    # Resolve tenant from context
    if tenant_context is None:
        print("[FACTS_FLOW] Error: tenant_context is None")
        return "Error: Facts module context not initialised."

    session = tenant_context.chat_sessions.get(tenant_context.session_id)
    if not session:
        print(f"[FACTS_FLOW] Error: No active session found for session_id '{tenant_context.session_id}'")
        return "Error: No active session for facts lookup."

    tenant_id = session.get("tenant_id")
    if not tenant_id:
        print("[FACTS_FLOW] Error: tenant_id not found in session")
        return "Error: Tenant ID not available."

    print(f"[FACTS_FLOW] Tenant resolved as '{tenant_id}'. Calling retrieve_facts...")
    try:
        facts = retrieve_facts(tenant_id, query, top_k=3)
        print(f"[FACTS_FLOW] retrieved {len(facts)} facts from Qdrant.")
        if not facts:
            print("[FACTS_FLOW] No facts found for the query.")
            return "No relevant information found in the knowledge base."
        print("[FACTS_FLOW] Successfully formatted facts to return.")
        return "FACTS:\n" + "\n---\n".join(facts)
    except RuntimeError as e:
        print(f"[FACTS_FLOW] Tool error (RuntimeError): {e}")
        return f"Error: Facts module unavailable — {e}"
    except Exception as e:
        print(f"[FACTS_FLOW] Unknown error retrieving facts: {e}")
        traceback.print_exc()
        return f"Error retrieving facts: {e}"