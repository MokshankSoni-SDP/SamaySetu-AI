"""
modules/module_registry.py
--------------------------
Central registry for all SamaySetu modules.
Determines which tools and prompt sections to activate per tenant
based on their enabled modules in the DB.

Currently supported modules:
  BOOKING_MODULE  — Google Calendar appointment system (existing)
  FACTS_MODULE    — RAG-based knowledge retrieval (new)

Adding a new module:
  1. Create modules/your_module.py
  2. Register it in MODULE_REGISTRY below
  3. Add its prompt section to prompts.py
"""

from typing import List, Dict, Any, Optional
from langchain_core.tools import StructuredTool
from pydantic import BaseModel

# ── Module name constants ─────────────────────────────────────────────────────
BOOKING_MODULE = "BOOKING_MODULE"
FACTS_MODULE   = "FACTS_MODULE"

ALL_MODULES = [BOOKING_MODULE, FACTS_MODULE]


def get_enabled_modules_for_tenant(tenant_id: str) -> List[str]:
    """
    Load enabled module names for a tenant from the DB.
    Falls back to [BOOKING_MODULE] if DB is unavailable (backward-compat).
    """
    try:
        from database.crud import get_enabled_modules
        modules = get_enabled_modules(tenant_id)
        # If no modules configured at all (new tenant not yet seeded), default to booking
        if not modules:
            return [BOOKING_MODULE]
        return modules
    except Exception as e:
        print(f"[MODULE_REGISTRY] DB unavailable, defaulting to BOOKING_MODULE: {e}")
        return [BOOKING_MODULE]

from langchain_core.tools import StructuredTool

class CheckAvailabilityInput(BaseModel):
    date: str
    time: str

# Repeat for each tool OR use simple wrapping:

def wrap_tool(func, name, description):
    return StructuredTool.from_function(
        func=func,
        name=name,
        description=description,
    )

class GetFactsInput(BaseModel):
    # IMPORTANT: Only 'query' is exposed to the LLM.
    # phone_number is injected by brain.py's execute_tool() before calling the function.
    # Including phone_number here caused Groq to generate malformed function call syntax:
    #   <function=get_facts{"query":...}> instead of <function=get_facts>{"query":...}
    query: str

def build_tools_for_tenant(tenant_id: str, enabled_modules: List[str]) -> List:
    """
    Dynamically assembles the tool list for a tenant based on enabled modules.
    Returns a list of callable tool functions compatible with LangChain bind_tools().
    """
    tools = []

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
                wrap_tool(check_calendar_availability, "check_calendar_availability", "Check available slots"),
                wrap_tool(book_appointment, "book_appointment", "Book appointment"),
                wrap_tool(cancel_appointment, "cancel_appointment", "Cancel appointment"),
                wrap_tool(reschedule_appointment, "reschedule_appointment", "Reschedule appointment"),
                wrap_tool(suggest_next_available_slot, "suggest_next_available_slot", "Suggest next slot"),
            ]
            print(f"[MODULE_REGISTRY] BOOKING_MODULE tools loaded for tenant={tenant_id}")
        except ImportError as e:
            print(f"[MODULE_REGISTRY] WARNING: Could not load booking tools: {e}")

    if FACTS_MODULE in enabled_modules:
        try:
            from modules.facts_module import get_facts
            
            get_facts_tool = StructuredTool.from_function(
                func = get_facts,
                name = "get_facts",
                description = (
                    "Search the business knowledge base to answer factual questions. "
                    "Call this tool whenever the user asks about: fees, charges, prices, costs, "
                    "location, address, directions, services offered, treatments, specializations, "
                    "working hours, timings, schedule, doctors, staff, policies, or any other "
                    "factual detail about this business. "
                    "Pass the user's question as a short English search query in the 'query' field."
                ),
                args_schema = GetFactsInput,
                return_direct = False,
            )
            tools.append(get_facts_tool)

            print(f"[MODULE_REGISTRY] FACTS_MODULE tool loaded for tenant={tenant_id}")
        except ImportError as e:
            print(f"[MODULE_REGISTRY] WARNING: Could not load facts tool: {e}")

    print("[TOOLS DEBUG] Tools passed to LLM")
    for t in tools:
        try:
            print(f"- {t.name}")
        except:
            print(f"- {t}")

    return tools


def get_module_status(tenant_id: str) -> Dict[str, bool]:
    """Returns a dict of {module_name: is_enabled} for a tenant."""
    enabled = set(get_enabled_modules_for_tenant(tenant_id))
    return {m: (m in enabled) for m in ALL_MODULES}