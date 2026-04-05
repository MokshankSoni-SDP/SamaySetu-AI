"""
brain.py
--------
Core AI reasoning engine for SamaySetu AI.
Handles memory extraction, dynamic tool loading, LLM invocation, and TTS dispatch.

LATENCY OPTIMISATIONS APPLIED:
  FIX 4 — Prompt size kept in check via max_history trim.
  FIX 5 — Memory extraction and module loading are PARALLELISED with asyncio.gather().
           Both run simultaneously instead of sequentially, saving the memory-LLM
           round-trip time for every request.
  FIX 6 — LLM with tools is built once per (tenant, module-set) combo and cached via
           module_registry._tools_cache.  The cached tool list is reused directly
           so bind_tools() is not called on every request.
"""

import re
import re as _re
import json
import asyncio
import traceback
import struct
from collections import Counter
from typing import Optional, List, Dict, TYPE_CHECKING

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage
from datetime import datetime
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_not_exception_type

try:
    from groq import BadRequestError
except ImportError:
    BadRequestError = Exception

import prompts
import config
from modules.module_registry import (
    get_enabled_modules_for_tenant,
    build_tools_for_tenant,
    BOOKING_MODULE,
    FACTS_MODULE,
)

if TYPE_CHECKING:
    from fastapi import WebSocket


# ── LLM clients ──────────────────────────────────────────────────────────────
import os

GROQ_SMALL_LLM_KEY = os.getenv("GROQ_SMALL_LLM")
small_llm = ChatGroq(
    api_key=GROQ_SMALL_LLM_KEY,
    model="llama-3.1-8b-instant",
    temperature=0
)
_main_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1)

max_history       = config.MAX_HISTORY
max_tool_iterations = config.MAX_TOOL_ITERATIONS
min_chunk_chars   = config.MIN_CHUNK_CHARS


# ── FIX 6: Per-(tenant, modules) LLM cache ───────────────────────────────────
# bind_tools() is not free — it serialises all tool schemas.
# We cache the bound LLM per tenant+module-set so it is created only once.
_llm_cache: Dict[str, object] = {}   # cache_key → llm_with_tools


def _get_llm_cache_key(tenant_id: str, enabled_modules: List[str]) -> str:
    return f"{tenant_id}::{','.join(sorted(enabled_modules))}"


def get_llm_with_tools(tenant_id: str, enabled_modules: List[str]):
    """Returns a cached (llm_with_tools, tools) pair for this tenant+modules combo."""
    key = _get_llm_cache_key(tenant_id, enabled_modules)
    fresh_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1)
    if key not in _llm_cache:
        tools = build_tools_for_tenant(tenant_id, enabled_modules)
        if tools:
            _llm_cache[key] = (fresh_llm.bind_tools(tools), tools)
        else:
            _llm_cache[key] = (fresh_llm, [])
        print(f"[BRAIN] LLM+tools cached for key={key}")
    return _llm_cache[key]


def invalidate_llm_cache(tenant_id: Optional[str] = None):
    """Clear LLM cache when module config changes. Pass None to clear all."""
    if tenant_id is None:
        _llm_cache.clear()
    else:
        keys_to_del = [k for k in _llm_cache if k.startswith(f"{tenant_id}::")]
        for k in keys_to_del:
            del _llm_cache[k]
        print(f"[BRAIN] LLM cache cleared for tenant={tenant_id}")


# ── Logging ───────────────────────────────────────────────────────────────────

def log(step: str, msg: str):
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{ts}] {step}: {msg}")


def _log_llm_retry(rs):
    exc = rs.outcome.exception()
    log("[LLM_RETRY]", f"Attempt {rs.attempt_number} failed. Exception: {exc}")
    try:
        if hasattr(exc, 'body') and isinstance(exc.body, dict):
            err = exc.body.get('error', {})
            if 'failed_generation' in err:
                log("[LLM_RETRY_DETAILS]", f"LLM Attempted: {err['failed_generation']}")
            elif 'message' in err:
                log("[LLM_RETRY_DETAILS]", f"LLM Error: {err['message']}")
    except Exception:
        pass


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(min=1, max=5),
    retry=retry_if_not_exception_type((BadRequestError, ValueError)),
    before_sleep=_log_llm_retry
)
async def safe_llm_call(llm_with_tools, messages):
    return await llm_with_tools.ainvoke(messages)


# ── Malformed tool-call recovery ──────────────────────────────────────────────

def _log_groq_error(exc: Exception, label: str = "[LLM_ERROR]"):
    """
    Dumps every meaningful field from a Groq BadRequestError for diagnosis.
    Call this BEFORE any recovery attempt so every failure is recorded.
    """
    log(label, f"Exception type : {type(exc).__name__}")
    log(label, f"Exception str  : {str(exc)[:600]}")
    try:
        body = getattr(exc, 'body', None)
        log(label, f"exc.body (raw) : {body}")
        if isinstance(body, str):
            body = json.loads(body)
        if isinstance(body, dict):
            err = body.get('error', {})
            log(label, f"error.type     : {err.get('type')}")
            log(label, f"error.code     : {err.get('code')}")
            log(label, f"error.message  : {str(err.get('message', ''))[:400]}")
            failed_gen = err.get('failed_generation', '')
            if failed_gen:
                log(label, f"failed_gen     : {failed_gen!r}")
            else:
                log(label, "failed_gen     : <not present>")
    except Exception as dump_err:
        log(label, f"Could not parse exc.body: {dump_err}")
    # Also log status_code if available
    sc = getattr(exc, 'status_code', getattr(exc, 'status', None))
    if sc:
        log(label, f"HTTP status    : {sc}")


def _log_messages_sent(messages, label: str = "[LLM_MSGS]"):
    """
    Logs a brief summary of each message in the list so we can verify
    the conversation shape Groq receives.
    """
    log(label, f"Total messages : {len(messages)}")
    for i, m in enumerate(messages):
        mtype = type(m).__name__
        content = ""
        try:
            if isinstance(m.content, str):
                content = m.content[:120].replace('\n', ' ')
            else:
                content = str(m.content)[:120]
        except Exception:
            content = "<unreadable>"
        tool_calls_info = ""
        if hasattr(m, 'tool_calls') and m.tool_calls:
            tool_calls_info = f" | tool_calls={[tc.get('name','?') for tc in m.tool_calls]}"
        tool_call_id = ""
        if hasattr(m, 'tool_call_id') and m.tool_call_id:
            tool_call_id = f" | tool_call_id={m.tool_call_id}"
        log(label, f"  [{i}] {mtype}{tool_calls_info}{tool_call_id} | '{content}'")


def _parse_malformed_tool_call(exc: Exception):
    """
    Tries to extract a tool call from a Groq tool_use_failed BadRequestError.
    Returns a synthetic AIMessage or None.
    """
    try:
        body = getattr(exc, 'body', None) or {}
        if isinstance(body, str):
            body = json.loads(body)
        failed_gen = body.get('error', {}).get('failed_generation', '')
        if not failed_gen:
            log("[TOOL_RECOVERY]", "No 'failed_generation' in error body — cannot recover")
            return None

        log("[TOOL_RECOVERY]", f"Parsing malformed generation: {failed_gen!r}")

        match = re.search(
            r'<function=([\w_]+)\s*[>]?\s*(\{.*?\})\s*(?:</function>|>|$)',
            failed_gen,
            re.DOTALL,
        )
        if not match:
            log("[TOOL_RECOVERY]", "Regex did not match — cannot recover")
            return None

        tool_name = match.group(1)
        try:
            tool_args = json.loads(match.group(2))
        except json.JSONDecodeError as je:
            log("[TOOL_RECOVERY]", f"JSON parse of args failed: {je}")
            return None

        log("[TOOL_RECOVERY]", f"Recovered: tool='{tool_name}' args={tool_args}")

        from langchain_core.messages import AIMessage
        import uuid
        return AIMessage(
            content="",
            tool_calls=[{"name": tool_name, "args": tool_args, "id": f"recovered_{uuid.uuid4().hex[:8]}"}],
        )
    except Exception as parse_err:
        log("[TOOL_RECOVERY]", f"Parse failed: {parse_err}")
        return None


# ── Memory extraction ─────────────────────────────────────────────────────────

async def extract_memory(user_text: str, memory: dict, lang_code: str = "gu-IN",
                          enabled_modules: List[str] = None):
    """Small LLM call to update memory state."""
    if enabled_modules is None:
        enabled_modules = [BOOKING_MODULE]
    try:
        prompt = prompts.get_memory_extraction_prompt(lang_code, enabled_modules)
        today  = datetime.now().strftime("%Y-%m-%d")
        messages = [
            SystemMessage(content=prompt),
            HumanMessage(content=f"""Today date: {today}

Current memory:
{json.dumps(memory)}

User input:
{user_text}
""")
        ]
        response  = await small_llm.ainvoke(messages)
        extracted = safe_json_parse(response.content)
        return extracted
    except Exception as e:
        print("[MEMORY] Extraction failed:", e)
        return {}


def merge_memory(old, new):
    for key, value in new.items():
        if isinstance(value, dict):
            old[key] = merge_memory(old.get(key, {}), value)
        else:
            if value is not None:
                old[key] = value
    return old


# ── Pure-Python language detector (zero LLM cost, <1ms) ─────────────────────
# Reads the user's message for explicit language requests and returns the
# requested language code. The main LLM is already instructed to ask the user
# for their preferred language at the start of every conversation; once the user
# answers, this detector picks it up and switches STT + TTS accordingly.

_LANG_NAMES = {
    "gu-IN": ["gujarati", "ગુજરાતી", "gujrati", "guj", "gujaratima", "gujarati ma"],
    "hi-IN": ["hindi", "હિન્દી", "हिंदी", "hindi ma", "hindi mein"],
    "en-IN": ["english", "ઇંગ્લિશ", "inglish", "angrezi", "અંગ્રેજી", "अंग्रेजी", "eng"],
}

_REQUEST_VERBS = [
    # Gujarati script
    "વાત કરી શકો", "વાત કરી શકશો", "વાત કરો", "બોલો", "બોલી શકો",
    # Romanised / Hindi
    "baat karo", "baat kar", "boliye", "bolna", "bolo", "bolsho",
    "bol sak", "baat kar sak",
    # English
    "speak", "talk", "switch to", "change to",
    "in english", "in gujarati", "in hindi",
    "prefer", "language",
]


def detect_requested_language(text: str) -> Optional[str]:
    """
    Returns 'gu-IN', 'hi-IN', 'en-IN', or None.
    Fires only on EXPLICIT language requests — not on general conversation.
    Zero LLM cost, runs in <1 ms.
    """
    t = text.lower().strip()
    has_verb = any(v in t for v in _REQUEST_VERBS)
    # For short messages (≤4 words) allow just the language name alone
    # e.g. user simply says "Hindi" or "Gujarati" as their preference answer.
    is_short = len(t.split()) <= 4
    if not has_verb and not is_short:
        return None

    matches = {code: any(n in t for n in names)
               for code, names in _LANG_NAMES.items()}
    hits = [code for code, found in matches.items() if found]
    if len(hits) != 1:
        return None   # ambiguous or none named
    return hits[0]


def _lang_label(lang_code: str) -> str:
    return {"en-IN": "English", "hi-IN": "Hindi", "gu-IN": "Gujarati"}.get(lang_code, lang_code)


def _extract_hours_ranges(error_text: str) -> Optional[str]:
    m = re.search(
        r"business hours:\s*(.+?)\.\s*Please choose a different time",
        error_text,
        flags=re.IGNORECASE,
    )
    if not m:
        return None
    return m.group(1).strip()


# ── Noise / echo filters ──────────────────────────────────────────────────────

COMMON_STT_VARIANTS = {
    "સ્વા": "સવા", "સવા": "સવા",
    "સાડા": "સાડા", "પોના": "પોણા", "પોણા": "પોણા"
}

GUJ_NUMBERS = {
    "એક": 1, "બે": 2, "ત્રણ": 3, "ચાર": 4,
    "પાંચ": 5, "છ": 6, "સાત": 7, "આઠ": 8,
    "નવ": 9, "દસ": 10, "અગિયાર": 11, "બાર": 12,
    "ek": 1, "eak": 1, "aek": 1, "1": 1,
    "be": 2, "bay": 2, "2": 2,
    "tran": 3, "teen": 3, "tin": 3, "3": 3,
    "char": 4, "chaar": 4, "4": 4,
    "panch": 5, "paanch": 5, "5": 5,
    "chha": 6, "chhah": 6, "cha": 6, "6": 6,
    "saat": 7, "sat": 7, "7": 7,
    "aath": 8, "ath": 8, "8": 8,
    "nav": 9, "9": 9,
    "das": 10, "10": 10,
    "agiyar": 11, "eleven": 11, "11": 11,
    "bar": 12, "bara": 12, "baara": 12, "twelve": 12, "12": 12,
}
SPECIAL_PHRASES = {"દોઢ": (1, 30), "અઢી": (2, 30)}
PHONETIC_PREFIX_MAP = {
    "sava ": "સવા ", "saava ": "સવા ",
    "sada ": "સાડા ", "saada ": "સાડા ", "sade ": "સાડા ",
    "pona ": "પોણા ", "pauna ": "પોણા ", "paune ": "પોણા ",
}


def is_noisy_transcript(text: str) -> bool:
    stripped = _re.sub(r'[\s.,!?।\u0964]+', ' ', text).strip()
    if len(stripped) <= 3:
        return True
    tokens = stripped.split()
    counts = Counter(tokens)
    if any(v >= 3 for v in counts.values()):
        return True
    if len(tokens) >= 4:
        unique = len(set(tokens))
        if unique / len(tokens) < 0.5:
            return True
    return False


def is_echo_of_ai(user_text: str, ai_text: str, threshold: float = 0.75) -> bool:
    if not ai_text or not user_text:
        return False
    u_tokens = set(user_text.lower().split())
    a_tokens = set(ai_text.lower().split())
    if not u_tokens or len(u_tokens) < 3:
        return False
    intersection = u_tokens & a_tokens
    union = u_tokens | a_tokens
    return len(intersection) / len(union) >= threshold


def normalize_variants(text):
    for wrong, correct in COMMON_STT_VARIANTS.items():
        text = text.replace(wrong, correct)
    return text


def normalize_phonetic_numbers(text: str) -> str:
    t = text.lower()
    for roman, gujarati in PHONETIC_PREFIX_MAP.items():
        t = t.replace(roman, gujarati)
    return t


def normalize_gujarati_time(text: str) -> str:
    text = text.lower().strip()
    text = normalize_phonetic_numbers(text)
    for phrase, (h, m) in SPECIAL_PHRASES.items():
        if phrase in text:
            text = text.replace(phrase, f"{h}:{m:02d}")
    text = normalize_variants(text)
    for prefix, minutes, offset in [("સવા", 15, 0), ("સાડા", 30, 0), ("પોણા", 45, -1)]:
        match = re.search(rf"{prefix}\s*(\w+)", text)
        if match:
            hour_word = match.group(1)
            hour = GUJ_NUMBERS.get(hour_word) or GUJ_NUMBERS.get(hour_word.lower())
            if hour:
                h = hour + offset
                text = text.replace(match.group(0), f"{h}:{minutes}")
    return text


# ── TTS helpers ───────────────────────────────────────────────────────────────

_TOOL_BLOCK_RE = re.compile(
    r'<function=[^>]*>.*?(?:</function>|$)'
    r'|\{[^{}]{0,800}"(?:tool_name|function_name|arguments|tool_use_id)"[^{}]{0,800}\}',
    re.DOTALL,
)


def is_tool_output(text: str) -> bool:
    t = text.strip()
    return (
        t.startswith("{") or
        t.startswith("[{") or
        "<function=" in t or
        '"tool_name"' in t or
        '"function_name"' in t
    )


def clean_for_tts(text: str) -> str:
    text = _TOOL_BLOCK_RE.sub("", text)
    text = re.sub(r'\{[^}]{0,500}\}', "", text)
    return text.strip()


def split_into_sentences(text: str) -> List[str]:
    raw = re.split(r'(?<=[.!?।\u0964])\s+', text.strip())
    chunks, buffer = [], ""
    for part in raw:
        part = part.strip()
        if not part:
            continue
        buffer = (buffer + " " + part).strip() if buffer else part
        if len(buffer) >= min_chunk_chars:
            chunks.append(buffer)
            buffer = ""
    if buffer:
        if chunks and len(buffer) < min_chunk_chars:
            chunks[-1] += " " + buffer
        else:
            chunks.append(buffer)
    return chunks or [text]


def safe_json_parse(text):
    try:
        return json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise


def compute_rms(raw_bytes: bytes) -> float:
    if len(raw_bytes) < 2:
        return 0.0
    samples = struct.unpack(f"<{len(raw_bytes)//2}h", raw_bytes)
    return (sum(s * s for s in samples) / len(samples)) ** 0.5


def _get_fallback_message(exc: Exception) -> str:
    err_str = str(exc).lower()
    if "rate_limit" in err_str or "429" in err_str:
        return "માફ કરશો, અત્યારે સર્વર ખૂબ વ્યસ્ત છે. થોડી વાર પછી ફરી પ્રયત્ન કરો."
    if "timeout" in err_str or "timed out" in err_str:
        return "માફ કરશો, જવાબ આવવામાં વધુ સમય લાગ્યો. કૃપા કરી ફરી પ્રયત્ન કરો."
    if "connection" in err_str or "network" in err_str:
        return "નેટવર્ક સમસ્યા આવી. કૃપા કરી ઇન્ટરનેટ તપાસો અને ફરી બોલો."
    return "માફ કરશો, કંઈક ભૂલ થઈ. થોડી વાર પછી ફરી પ્રયત્ન કરો."


def print_token_usage(msg, step_name):
    usage = msg.response_metadata.get("token_usage", {})
    if usage:
        print(f"--- 📊 {step_name} | prompt={usage.get('prompt_tokens')} "
              f"completion={usage.get('completion_tokens')} "
              f"total={usage.get('total_tokens')} ---")


# ── Tool context injection ────────────────────────────────────────────────────

def _inject_tool_context(session_id: str, chat_sessions: dict, enabled_modules: List[str]):
    """Injects session context into module tools so they can resolve tenant_id."""

    class _Ctx:
        pass

    ctx = _Ctx()
    ctx.session_id    = session_id
    ctx.chat_sessions = chat_sessions

    if BOOKING_MODULE in enabled_modules:
        try:
            import calendar_tool
            calendar_tool.tenant_context = ctx
        except ImportError:
            pass

    if FACTS_MODULE in enabled_modules:
        try:
            from modules import facts_module
            facts_module.tenant_context = ctx
        except ImportError:
            pass


async def _run_tool_async(tool_name: str, args: dict):
    """Dynamically resolve and run a tool by name in a thread."""
    try:
        import calendar_tool
        func = getattr(calendar_tool, tool_name, None)
        if func:
            return await asyncio.to_thread(func, **args)
    except ImportError:
        pass

    try:
        from modules import facts_module
        func = getattr(facts_module, tool_name, None)
        if func:
            return await asyncio.to_thread(func, **args)
    except ImportError:
        pass

    raise ValueError(f"Unknown tool: {tool_name}")


# ── Main brain (run_brain) ────────────────────────────────────────────────────

async def run_brain(
    session_id: str,
    user_text: str,
    websocket,
    tts_session,
    chat_sessions: Dict,
    tts_convert_fn,
):
    """
    Core reasoning loop.

    Pipeline:
      1. Normalize input
      2. Handle unclear/noisy transcripts (early return)
      3. Resolve enabled modules for tenant (cached in session)
      4. FIX 5: Parallelise memory extraction + module/LLM resolution
      5. Build system prompt (module-aware)
      6. Run main LLM with tools
      7. Tool execution loop
      8. Stream TTS response
    """
    t0    = datetime.now()
    today = t0.strftime("%Y-%m-%d")
    day   = t0.strftime("%A")

    log("[NORMALIZER]", f"Before: {user_text}")
    user_text = normalize_gujarati_time(user_text)
    log("[NORMALIZER]", f"After: {user_text}")

    log("[BRAIN]", f"START | session='{session_id}' | text='{user_text}'")

    # ── Handle noisy / unclear input ──────────────────────────────────────────
    if user_text == "__UNCLEAR__":
        bot_cfg    = chat_sessions.get(session_id, {}).get("bot_config") or {}
        fb_speaker = bot_cfg.get("tts_speaker", "simran")
        fb_lang    = bot_cfg.get("language_code", "gu-IN")
        clarification = prompts.LANG_PACK.get(fb_lang, prompts.LANG_PACK["gu-IN"])["unclear_msg"]
        log("[BRAIN]", "Noisy input — sending clarification")
        sentences = split_into_sentences(clarification)
        await websocket.send_json({"type": "ai_text", "text": clarification, "chunk_count": len(sentences)})
        await websocket.send_json({"type": "ai_speaking_start"})
        if tts_session:
            done_evt = asyncio.Event()
            await tts_session.speak(sentences, fb_speaker, fb_lang, 0, done_evt)
            await done_evt.wait()
        else:
            for idx, sentence in enumerate(sentences):
                audio_b64 = await tts_convert_fn(sentence, fb_speaker, fb_lang)
                await websocket.send_json({
                    "type": "audio_chunk", "index": idx, "total": len(sentences),
                    "text": sentence, "audio": audio_b64, "is_last": idx == len(sentences) - 1
                })
            await websocket.send_json({"type": "tts_done"})
        return

    # ── Init session ──────────────────────────────────────────────────────────
    if session_id not in chat_sessions:
        chat_sessions[session_id] = {
            "history": [],
            "phone_number": None,
            "tenant_id": None,
            "bot_config": None,
            "memory": {
                "intent": None,
                "language_preference": None,
                "pending_action": "waiting_for_confirmation",
                "appointment": {"date": None, "time": None, "duration": None},
                "reschedule": {"old_time": None, "new_time": None},
                "date_context": {"resolved_date": None, "source": "none"},
            }
        }

    session_data = chat_sessions[session_id]
    # Ensure memory schema exists even when session was pre-created by websocket handler.
    session_data.setdefault("memory", {})
    session_data["memory"].setdefault("intent", None)
    session_data["memory"].setdefault("language_preference", None)
    session_data["memory"].setdefault("pending_action", "waiting_for_confirmation")
    session_data["memory"].setdefault("appointment", {"date": None, "time": None, "duration": None})
    session_data["memory"].setdefault("reschedule", {"old_time": None, "new_time": None})
    session_data["memory"].setdefault("date_context", {"resolved_date": None, "source": "none"})
    session_data.setdefault("language_prompt_asked", False)
    history      = session_data["history"]
    phone_number = session_data.get("phone_number")
    bot_config   = session_data.get("bot_config") or {}
    tts_speaker  = bot_config.get("tts_speaker", "simran")
    tts_lang     = bot_config.get("language_code", "gu-IN")
    tenant_id    = session_data.get("tenant_id")

    # ── Resolve enabled modules (cached in session, DB-fetched once) ──────────
    enabled_modules = session_data.get("enabled_modules")
    if enabled_modules is None:
        if tenant_id:
            # Run in thread so the synchronous DB call doesn't block the event loop
            enabled_modules = await asyncio.to_thread(get_enabled_modules_for_tenant, tenant_id)
        else:
            enabled_modules = [BOOKING_MODULE]
        session_data["enabled_modules"] = enabled_modules
    log("[MODULES]", f"Enabled: {enabled_modules}")

    # ── FIX 5: PARALLELISE memory extraction + LLM/tool resolution ────────────
    # Memory extraction requires a small-LLM network call (~500–800 ms).
    # get_llm_with_tools() is synchronous and usually instant (cached), but on
    # the very first call it does bind_tools() which has some CPU overhead.
    # Running both concurrently shaves the memory-LLM round-trip off the critical path.
    memory = session_data.get("memory", {})

    mem_task    = asyncio.create_task(extract_memory(user_text, memory, tts_lang, enabled_modules))
    llm_task    = asyncio.get_event_loop().run_in_executor(
        None, get_llm_with_tools, tenant_id or "", enabled_modules
    )

    new_memory_result, llm_result = await asyncio.gather(mem_task, llm_task)

    llm_with_tools, active_tools = llm_result

    # Merge + save memory
    prev_lang_pref = memory.get("language_preference")
    updated_memory = merge_memory(memory, new_memory_result)
    # Do not trust extracted language_preference unless explicitly requested this turn.
    updated_memory["language_preference"] = prev_lang_pref
    print(updated_memory)
    session_data["memory"] = updated_memory

    log("[MEMORY]", f"Updated: {updated_memory}")

    # ── Pure-Python language switch (<1 ms, zero LLM cost) ───────────────────
    # Checks the user's message for an explicit language request (e.g. "Hindi",
    # "speak in Gujarati", "English please").  When the LLM asks the user for
    # their preferred language at the start of the conversation the user's one-
    # word / short answer is enough to trigger this.
    _SPEAKER_MAP = {"gu-IN": "simran", "hi-IN": "simran", "en-IN": "simran"}
    requested_lang = detect_requested_language(user_text)
    if requested_lang:
        updated_memory["language_preference"] = requested_lang
        session_data["memory"] = updated_memory
        session_data["language_prompt_asked"] = True
    if requested_lang and requested_lang != tts_lang:
        log("[LANG_SWITCH]", f"Language switched: {tts_lang} → {requested_lang}")
        tts_lang    = requested_lang
        tts_speaker = _SPEAKER_MAP.get(requested_lang, tts_speaker)
        # Persist into live bot_config so next STT reconnect reads the new lang
        session_data["bot_config"]["language_code"] = tts_lang
        session_data["bot_config"]["tts_speaker"]   = tts_speaker
        # Signal sarvam_sender to close current STT connection and reopen with new lang
        lang_evt = session_data.get("language_switch_event")
        if lang_evt is not None:
            lang_evt.set()
            log("[LANG_SWITCH]", f"language_switch_event SET → STT will reconnect as {tts_lang}")


    # ── Inject tenant context into module tools ───────────────────────────────
    # Deterministic first-turn language question:
    # if no language chosen yet and this is first user turn, ask language now.
    non_system_msgs = [m for m in history if not isinstance(m, SystemMessage)]
    if (
        not updated_memory.get("language_preference")
        and not session_data.get("language_prompt_asked", False)
        and len(non_system_msgs) == 0
        and requested_lang is None
    ):
        receptionist = bot_config.get("receptionist_name") or "Priya"
        question_map = {
            "gu-IN": f"નમસ્તે! હું {receptionist} છું. તમે ગુજરાતી, હિન્દી કે અંગ્રેજીમાં વાત કરશો?",
            "hi-IN": f"नमस्ते! मैं {receptionist} हूँ। क्या आप गुजराती, हिन्दी या अंग्रेज़ी में बात करना चाहेंगे?",
            "en-IN": f"Hello! I'm {receptionist}. Would you prefer Gujarati, Hindi, or English?",
        }
        ask_text = question_map.get(tts_lang, question_map["en-IN"])
        session_data["language_prompt_asked"] = True
        history.append(AIMessage(content=ask_text))
        session_data["last_ai_text"] = ask_text
        sentences = split_into_sentences(ask_text)
        log("[BRAIN]", f"Reply: '{ask_text[:100]}' | {len(sentences)} sentence(s) [forced language ask]")
        await websocket.send_json({"type": "ai_text", "text": ask_text, "chunk_count": len(sentences)})
        await websocket.send_json({"type": "ai_speaking_start"})
        if tts_session:
            done_evt = asyncio.Event()
            import random as _rand
            resp_id = _rand.randint(1, 999999)
            await tts_session.speak(sentences, tts_speaker, tts_lang, resp_id, done_evt)
            await done_evt.wait()
        else:
            for idx, sentence in enumerate(sentences):
                audio_b64 = await tts_convert_fn(sentence, tts_speaker, tts_lang)
                await websocket.send_json({
                    "type": "audio_chunk", "index": idx, "total": len(sentences),
                    "text": sentence, "audio": audio_b64, "is_last": idx == len(sentences) - 1
                })
            await websocket.send_json({"type": "tts_done"})
        log("[BRAIN]", f"DONE in {(datetime.now()-t0).total_seconds():.2f}s")
        return

    _inject_tool_context(session_id, chat_sessions, enabled_modules)

    # ── Build system prompt ───────────────────────────────────────────────────
    memory_context = f"\n\n=== MEMORY STATE ===\n{json.dumps(updated_memory)}"
    system_prompt  = (
        prompts.get_system_prompt(today, day, bot_config, enabled_modules)
        + memory_context
    )
    preferred_lang = updated_memory.get("language_preference")
    if preferred_lang:
        system_prompt += (
            f"\n\n=== LANGUAGE LOCK ===\n"
            f"The user has already chosen {_lang_label(preferred_lang)} ({preferred_lang}). "
            f"Do NOT ask language preference again in this conversation. "
            f"Reply in {_lang_label(preferred_lang)} unless user explicitly asks to switch language."
        )

    if history and isinstance(history[0], SystemMessage):
        history[0] = SystemMessage(content=system_prompt)
    else:
        history.insert(0, SystemMessage(content=system_prompt))

    history.append(HumanMessage(content=user_text))

    # FIX 4: trim history to avoid bloating prompt tokens
    non_system = [m for m in history if not isinstance(m, SystemMessage)]
    recent_history = [history[0]] + non_system[-max_history:]

    # ── First LLM call ────────────────────────────────────────────────────────
    t_llm = datetime.now()
    log("[LLM]", f"ainvoke() | {len(recent_history)} msgs | modules={enabled_modules}")
    # Log registered tool names so we can verify what Groq sees
    if active_tools:
        log("[LLM]", f"Bound tools    : {[t.name for t in active_tools]}")
    else:
        log("[LLM]", "Bound tools    : <none>")
    # Log message shape for diagnosis
    _log_messages_sent(recent_history, "[LLM_MSGS]")
    try:
        ai_msg = await safe_llm_call(llm_with_tools, recent_history)
        log("[LLM]", f"Done in {(datetime.now()-t_llm).total_seconds():.2f}s | "
            f"tool_calls={len(ai_msg.tool_calls)}")
        print_token_usage(ai_msg, "Initial LLM")
    except BadRequestError as e:
        _log_groq_error(e, "[LLM_ERROR]")
        log("[LLM]", "BadRequestError (tool_use_failed) — attempting recovery")
        recovered = _parse_malformed_tool_call(e)
        if recovered:
            log("[LLM]", f"Recovery successful — executing '{recovered.tool_calls[0]['name']}'")
            ai_msg = recovered
        else:
            log("[LLM]", "Recovery failed — re-raising")
            raise
    except Exception as e:
        log("[LLM]", f"FAILED: {e}\n{traceback.format_exc()}")
        raise

    # ── Tool execution loop ───────────────────────────────────────────────────
    tool_iteration = 0
    while ai_msg.tool_calls:
        tool_iteration += 1
        if tool_iteration > max_tool_iterations:
            log("[TOOLS]", f"Max tool iterations ({max_tool_iterations}). Forcing response.")
            break
        log("[TOOLS]", f"Iteration #{tool_iteration} — {len(ai_msg.tool_calls)} tool(s)")
        history.append(ai_msg)

        async def execute_tool(tool_call):
            tname = tool_call["name"]
            targs = tool_call["args"]
            if phone_number:
                targs = {**targs, "phone_number": phone_number}

            log("[TOOL]", f"Executing '{tname}' | args={targs}")
            await websocket.send_json({"type": "tool_call", "name": tname, "args": targs, "status": "running"})
            t_tool = datetime.now()

            try:
                obs = await _run_tool_async(tname, targs)
                obs_text = str(obs)
                if (
                    tname == "check_calendar_availability"
                    and "We only accept appointments during these business hours" in obs_text
                ):
                    session_data["_out_of_hours_error"] = {
                        "error_text": obs_text,
                        "requested_start": targs.get("start_time_str"),
                    }
                if "success" in str(obs).lower():
                    print("$$$state memory cleared after success$$$")
                    lang_pref = session_data.get("memory", {}).get("language_preference")
                    session_data["memory"] = {
                        "intent": "none",
                        "language_preference": lang_pref,
                        "pending_action": "waiting_for_confirmation",
                        "appointment": {"date": None, "time": None, "duration": None},
                        "reschedule": {"old_time": None, "new_time": None},
                        "date_context": {"resolved_date": None, "source": "none"}
                    }
                    # Force module re-fetch next turn in case config changed
                    session_data.pop("enabled_modules", None)

                status = "ok"
                log("[TOOL]", f"'{tname}' OK in {(datetime.now()-t_tool).total_seconds():.2f}s")
            except Exception as e:
                obs, status = f"Error: {e}", "error"
                log("[TOOL]", f"'{tname}' FAILED: {e}")

            await websocket.send_json({
                "type": "tool_call", "name": tname, "args": targs,
                "status": status, "result": str(obs)
            })
            return ToolMessage(content=str(obs), tool_call_id=tool_call["id"])

        tool_results   = await asyncio.gather(*[execute_tool(tc) for tc in ai_msg.tool_calls])
        history.extend(tool_results)
        recent_history = [history[0]] + history[-max_history:]

        out_of_hours = session_data.pop("_out_of_hours_error", None)
        if out_of_hours:
            ranges = _extract_hours_ranges(out_of_hours.get("error_text", "")) or "the configured business hours"
            req_start = out_of_hours.get("requested_start")
            req_time = None
            if isinstance(req_start, str):
                try:
                    req_dt = datetime.fromisoformat(req_start)
                    req_time = req_dt.strftime("%I:%M %p").lstrip("0")
                except Exception:
                    req_time = None

            if req_time:
                reply = (
                    f"There is no slot at {req_time} because it is outside business hours. "
                    f"We accept appointments only during {ranges}. "
                    "Please choose a time within these periods."
                )
            else:
                reply = (
                    f"That requested time is outside business hours. "
                    f"We accept appointments only during {ranges}. "
                    "Please choose a time within these periods."
                )
            ai_msg = AIMessage(content=reply)
            break

        t_llm2 = datetime.now()
        log("[LLM]", "Post-tool ainvoke()")
        _log_messages_sent(recent_history, "[LLM_POST_MSGS]")
        try:
            ai_msg = await safe_llm_call(llm_with_tools, recent_history)
            log("[LLM]", f"Done in {(datetime.now()-t_llm2).total_seconds():.2f}s")
            print_token_usage(ai_msg, "Post-tool LLM")
        except BadRequestError as e:
            _log_groq_error(e, "[LLM_POST_ERROR]")
            log("[LLM]", "Post-tool BadRequestError — attempting recovery")
            recovered = _parse_malformed_tool_call(e)
            if recovered:
                log("[LLM]", "Post-tool recovery successful")
                ai_msg = recovered
            else:
                log("[LLM]", "Post-tool recovery failed — re-raising")
                raise
        except Exception as e:
            log("[LLM]", f"Post-tool FAILED: {e}")
            raise

    # ── Final reply ───────────────────────────────────────────────────────────
    reply_text = ai_msg.content if isinstance(ai_msg.content, str) else str(ai_msg.content)
    history.append(ai_msg)

    # FIX 4: trim history to keep tokens under control
    if len(history) > max_history * 2 + 2:
        history[:] = [history[0]] + history[-(max_history * 2):]

    session_data["last_ai_text"] = reply_text

    if is_tool_output(reply_text):
        log("[TTS_FILTER]", f"Blocked tool output from TTS: '{reply_text[:80]}'")
        log("[BRAIN]", f"DONE (tool-only, no TTS) in {(datetime.now()-t0).total_seconds():.2f}s")
        return

    tts_text = clean_for_tts(reply_text)
    if not tts_text:
        log("[TTS_FILTER]", "Reply cleaned to empty — skipping TTS")
        log("[BRAIN]", f"DONE (empty) in {(datetime.now()-t0).total_seconds():.2f}s")
        return

    sentences = split_into_sentences(tts_text)
    log("[BRAIN]", f"Reply: '{tts_text[:100]}' | {len(sentences)} sentence(s)")

    import random as _rand
    resp_id = _rand.randint(1, 999999)

    await websocket.send_json({
        "type": "ai_text", "text": reply_text, "chunk_count": len(sentences)
    })
    await websocket.send_json({"type": "ai_speaking_start"})

    if tts_session:
        done_evt = asyncio.Event()
        await tts_session.speak(sentences, tts_speaker, tts_lang, resp_id, done_evt)
        await done_evt.wait()
    else:
        for idx, sentence in enumerate(sentences):
            audio_b64 = await tts_convert_fn(sentence, tts_speaker, tts_lang)
            await websocket.send_json({
                "type": "audio_chunk", "index": idx, "total": len(sentences),
                "text": sentence, "audio": audio_b64, "is_last": idx == len(sentences) - 1
            })
        await websocket.send_json({"type": "tts_done"})

    log("[BRAIN]", f"DONE in {(datetime.now()-t0).total_seconds():.2f}s")
