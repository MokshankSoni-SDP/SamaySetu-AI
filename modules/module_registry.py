"""
modules/module_registry.py
--------------------------
Central registry for all SamaySetu modules.
Determines which tools and prompt sections to activate per tenant
based on their enabled modules in the DB.

LATENCY OPTIMISATION (FIX 6):
  Per-tenant tool lists are cached in _tools_cache after the first build.
  Subsequent requests for the same tenant skip build_tools_for_tenant() entirely.
  Cache is invalidated explicitly when module config changes (admin panel).

Currently supported modules:
  BOOKING_MODULE  — Google Calendar appointment system
  FACTS_MODULE    — RAG-based knowledge retrieval

Adding a new module:
  1. Create modules/your_module.py
  2. Register it in build_tools_for_tenant()
  3. Add its prompt section to prompts.py
"""

from typing import List, Dict, Any, Optional
from langchain_core.tools import StructuredTool
from pydantic import BaseModel

# ── Module name constants ─────────────────────────────────────────────────────
BOOKING_MODULE = "BOOKING_MODULE"
FACTS_MODULE   = "FACTS_MODULE"

ALL_MODULES = [BOOKING_MODULE, FACTS_MODULE]


# ─────────────────────────────────────────────────────────────────────────────
# FIX 6 — PER-TENANT TOOLS CACHE
# Tools list is built once per tenant and cached until invalidated.
# build_tools_for_tenant() touches several imports and wraps functions —
# caching this saves ~5–15 ms per request.
# ─────────────────────────────────────────────────────────────────────────────

_tools_cache: Dict[str, List] = {}   # tenant_id → tool list


def invalidate_tools_cache(tenant_id: Optional[str] = None):
    """
    Clear cached tools for a tenant (call after admin changes module config).
    Pass None to clear the entire cache.
    """
    if tenant_id is None:
        _tools_cache.clear()
        print("[MODULE_REGISTRY] Entire tools cache cleared")
    elif tenant_id in _tools_cache:
        del _tools_cache[tenant_id]
        print(f"[MODULE_REGISTRY] Tools cache cleared for tenant={tenant_id}")


# ─────────────────────────────────────────────────────────────────────────────
# Module resolution
# ─────────────────────────────────────────────────────────────────────────────

def get_enabled_modules_for_tenant(tenant_id: str) -> List[str]:
    """
    Load enabled module names for a tenant from the DB.
    Falls back to [BOOKING_MODULE] if DB is unavailable (backward-compat).
    """
    try:
        from database.crud import get_tenant_modules
        modules_dict = get_tenant_modules(tenant_id)
        enabled = [name for name, on in modules_dict.items() if on]
        # If nothing configured, default to booking only
        return enabled if enabled else [BOOKING_MODULE]
    except Exception as e:
        print(f"[MODULE_REGISTRY] DB unavailable, defaulting to BOOKING_MODULE: {e}")
        return [BOOKING_MODULE]


# ─────────────────────────────────────────────────────────────────────────────
# Tool schemas
# ─────────────────────────────────────────────────────────────────────────────

class GetFactsInput(BaseModel):
    # Only 'query' exposed to the LLM.
    # phone_number is injected by brain.py BEFORE the function call.
    # Including it here caused Groq to emit malformed function call syntax.
    query: str


def _wrap(func, name: str, description: str):
    return StructuredTool.from_function(func=func, name=name, description=description)


# ─────────────────────────────────────────────────────────────────────────────
# Tool builder  (result is cached per tenant)
# ─────────────────────────────────────────────────────────────────────────────

def build_tools_for_tenant(tenant_id: str, enabled_modules: List[str]) -> List:
    """
    Assembles the tool list for a tenant based on enabled modules.
    Result is memoised in _tools_cache keyed on (tenant_id, sorted modules).

    Returns a list of StructuredTool objects compatible with LangChain bind_tools().
    """
    # ── Cache key: tenant + sorted module list ────────────────────────────────
    cache_key = f"{tenant_id}::{','.join(sorted(enabled_modules))}"
    if cache_key in _tools_cache:
        return _tools_cache[cache_key]

    tools = []

    # ── BOOKING_MODULE ────────────────────────────────────────────────────────
    if BOOKING_MODULE in enabled_modules:
        try:
            from calendar_tool import (
                check_calendar_availability,
                book_appointment,
                cancel_appointment,
                reschedule_appointment,
                suggest_next_available_slot,
            )
            tools += [
                _wrap(check_calendar_availability,  "check_calendar_availability", "Check if a time slot is available"),
                _wrap(book_appointment,             "book_appointment",             "Book an appointment"),
                _wrap(cancel_appointment,           "cancel_appointment",           "Cancel an existing appointment"),
                _wrap(reschedule_appointment,       "reschedule_appointment",       "Reschedule an appointment"),
                _wrap(suggest_next_available_slot,  "suggest_next_available_slot",  "Suggest the next free slots"),
            ]
            print(f"[MODULE_REGISTRY] BOOKING_MODULE tools loaded for tenant={tenant_id}")
        except ImportError as e:
            print(f"[MODULE_REGISTRY] WARNING: Could not load booking tools: {e}")

    # ── FACTS_MODULE ──────────────────────────────────────────────────────────
    if FACTS_MODULE in enabled_modules:
        try:
            from modules.facts_module import get_facts
            get_facts_tool = StructuredTool.from_function(
                func=get_facts,
                name="get_facts",
                description=(
                    "Search the business knowledge base to answer factual questions. "
                    "Call this whenever the user asks about: fees, charges, prices, costs, "
                    "location, address, directions, services offered, treatments, specializations, "
                    "working hours, timings, schedule, doctors, staff, policies, or any other "
                    "factual detail about this business. "
                    "Pass the user's question as a short English search query in the 'query' field."
                ),
                args_schema=GetFactsInput,
                return_direct=False,
            )
            tools.append(get_facts_tool)
            print(f"[MODULE_REGISTRY] FACTS_MODULE tool loaded for tenant={tenant_id}")
        except ImportError as e:
            print(f"[MODULE_REGISTRY] WARNING: Could not load facts tool: {e}")

    if not tools:
        print(f"[MODULE_REGISTRY] WARNING: No tools available for tenant={tenant_id}")
    else:
        print(f"[MODULE_REGISTRY] Tools for tenant={tenant_id}: {[t.name for t in tools]}")

    # ── Store in cache ────────────────────────────────────────────────────────
    _tools_cache[cache_key] = tools
    return tools


def get_module_status(tenant_id: str) -> Dict[str, bool]:
    """Returns a dict of {module_name: is_enabled} for a tenant."""
    enabled = set(get_enabled_modules_for_tenant(tenant_id))
    return {m: (m in enabled) for m in ALL_MODULES}