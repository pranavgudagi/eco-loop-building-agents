"""
LangGraph node functions for the closed-loop control agent.

State machine:
    Perceive -> Reason -> Act -> Verify -> (Correct | done)

Each node is a plain function that takes an AgentState dict and returns
an updated AgentState dict. LangGraph orchestrates the transitions.

Nodes:
  perceive:  Build compact snapshot from live EnergyPlus state.
  reason:    LLM decides which tools to call (or "no action needed").
  act:       Execute the LLM's tool calls, catching + logging failures.
  verify:    Compare pre/post state; decide if correction is needed.
  correct:   LLM proposes a corrective action based on outcome.
"""

import time
from typing import TypedDict, Optional, Any
from dataclasses import dataclass, field

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

import config
from mcp_server import get_driver, get_persistence
from mcp_server.tools.sensors import get_full_snapshot
from mcp_server.tools.actuators import set_cooling_setpoint, set_heating_setpoint
from agent.llm_client import LLMClient
from agent.prompts import (
    SYSTEM_PROMPT,
    build_reason_prompt,
    build_correct_prompt,
)


# ---------------------------------------------------------------------------
# Agent state
# ---------------------------------------------------------------------------
class AgentState(TypedDict, total=False):
    """
    State object passed between LangGraph nodes.
    total=False means all fields are optional (avoids init boilerplate).
    """
    # Snapshots
    pre_snapshot: dict          # state before action
    post_snapshot: dict         # state after action (populated by verify)

    # LLM output
    llm_text: str               # freeform reasoning from LLM
    tool_calls: list            # tool calls the LLM emitted

    # Action outcomes
    executed_actions: list      # list of {name, args, result}
    action_error: Optional[str]

    # Correction state
    correction_iteration: int   # 0 for first pass, incremented per correct
    correction_needed: bool
    correction_reason: str

    # Bookkeeping
    node_history: list          # names of nodes visited (for debug)


# ---------------------------------------------------------------------------
# LangChain tool wrappers for the LLM
# ---------------------------------------------------------------------------
# The LLM sees these as tools it can call. When the LLM emits a tool_call,
# our Act node dispatches through this table.

from langchain_core.tools import tool

@tool
def llm_set_cooling_setpoint(temperature_c: float, reasoning: str) -> str:
    """Override the building-wide cooling setpoint (°C).

    Args:
        temperature_c: Target cooling setpoint in Celsius. Will be clamped to safe range.
        reasoning: Brief one-sentence justification for this change.
    """
    result = set_cooling_setpoint(temperature_c, reasoning=reasoning)
    return (f"Applied cooling setpoint {result['applied_value']}°C"
            f"{' (clamped from ' + str(result['original_request']) + ')' if result['was_clamped'] else ''}")


@tool
def llm_set_heating_setpoint(temperature_c: float, reasoning: str) -> str:
    """Override the building-wide heating setpoint (°C).

    Args:
        temperature_c: Target heating setpoint in Celsius. Will be clamped to safe range.
        reasoning: Brief one-sentence justification for this change.
    """
    result = set_heating_setpoint(temperature_c, reasoning=reasoning)
    return (f"Applied heating setpoint {result['applied_value']}°C"
            f"{' (clamped from ' + str(result['original_request']) + ')' if result['was_clamped'] else ''}")


# Only expose cooling — heating is nonsensical in Bangalore July
LLM_TOOLS = [llm_set_cooling_setpoint]
TOOL_DISPATCH = {
    "llm_set_cooling_setpoint": llm_set_cooling_setpoint,
}

# ---------------------------------------------------------------------------
# Shared LLM client (constructed once, reused across cycles)
# ---------------------------------------------------------------------------
_llm_client: Optional[LLMClient] = None

def get_llm_client() -> LLMClient:
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient().bind_tools(LLM_TOOLS)
    return _llm_client


# ---------------------------------------------------------------------------
# Rolling history (in-process, so LLM can see recent trends without DB reads)
# ---------------------------------------------------------------------------
_history: list = []  # list of compact summary dicts
_recent_actions: list = []  # last few actions from persistence

def _push_history(snapshot: dict):
    zones = snapshot.get("zones", {})
    temps = [z.get("temp") for z in zones.values() if z.get("temp") is not None]
    if not temps:
        return
    avg_temp = sum(temps) / len(temps)
    hvac = snapshot.get("total_hvac_power", {})
    _history.append({
        "time": snapshot.get("time", {}),
        "avg_temp": avg_temp,
        "outdoor": snapshot.get("outdoor_temp", 0),
        "total_power": hvac.get("total_watts", 0),
    })
    # Keep last 12 hours (12 entries at 1/hour)
    while len(_history) > 12:
        _history.pop(0)


# ---------------------------------------------------------------------------
# Node: Perceive
# ---------------------------------------------------------------------------
def perceive(state: AgentState) -> AgentState:
    """Read the current simulation state into a compact snapshot."""
    snapshot = get_full_snapshot()
    _push_history(snapshot)

    node_history = state.get("node_history", []) + ["perceive"]
    return {
        "pre_snapshot": snapshot,
        "node_history": node_history,
        "correction_iteration": state.get("correction_iteration", 0),
    }


# ---------------------------------------------------------------------------
# Node: Reason
# ---------------------------------------------------------------------------
def reason(state: AgentState) -> AgentState:
    """Call the LLM to decide which (if any) tools to invoke."""
    snapshot = state["pre_snapshot"]
    llm = get_llm_client()
    persistence = get_persistence()

    user_prompt = build_reason_prompt(snapshot, _history, _recent_actions)
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ]

    response = llm.invoke(messages)

    # Log this LLM call to persistence
    persistence.log_llm_call(
        sim_time=snapshot.get("time", {}),
        node_name="reason",
        prompt={"system_len": len(SYSTEM_PROMPT), "user_len": len(user_prompt)},
        response=response.text,
        tool_calls=[{"name": tc.get("name"), "args": tc.get("args")}
                    for tc in response.tool_calls],
        latency_ms=response.latency_ms,
        tokens_in=response.tokens_in,
        tokens_out=response.tokens_out,
        error=response.error,
    )

    node_history = state.get("node_history", []) + ["reason"]
    return {
        "llm_text": response.text,
        "tool_calls": response.tool_calls,
        "node_history": node_history,
    }


# ---------------------------------------------------------------------------
# Node: Act
# ---------------------------------------------------------------------------
def act(state: AgentState) -> AgentState:
    """Execute every tool call the LLM emitted."""
    tool_calls = state.get("tool_calls", [])
    executed = []
    error = None

    for tc in tool_calls:
        name = tc.get("name")
        args = tc.get("args", {})
        fn = TOOL_DISPATCH.get(name)
        if fn is None:
            executed.append({"name": name, "args": args,
                             "result": f"UNKNOWN_TOOL: {name}"})
            continue
        try:
            result = fn.invoke(args)
            executed.append({"name": name, "args": args, "result": result})
            # Also remember it for the next cycle's context
            _recent_actions.append({
                "sim_day": state["pre_snapshot"].get("time", {}).get("day"),
                "sim_hour": state["pre_snapshot"].get("time", {}).get("hour"),
                "kind": name.replace("llm_", ""),
                "value_after": args.get("temperature_c"),
                "reasoning": args.get("reasoning", ""),
            })
            while len(_recent_actions) > 6:
                _recent_actions.pop(0)
        except Exception as e:
            error = f"{name}: {type(e).__name__}: {e}"
            executed.append({"name": name, "args": args, "result": f"ERROR: {error}"})

    node_history = state.get("node_history", []) + ["act"]
    return {
        "executed_actions": executed,
        "action_error": error,
        "node_history": node_history,
    }


# ---------------------------------------------------------------------------
# Node: Verify
# ---------------------------------------------------------------------------
def verify(state: AgentState) -> AgentState:
    """
    Compare pre/post state. In this simple design, the "post" state is
    just a fresh snapshot immediately after the write. Real verification
    happens over the next control cycle when Perceive fires again — this
    node's main job is deciding whether a Correct pass is needed NOW.

    Correction triggers:
      - Any zone temperature outside comfort band during occupied hours.
      - An action_error from Act (tool call raised or unknown tool).
      - A [CLAMPED] event (LLM asked for out-of-range setpoint).
    """
    executed = state.get("executed_actions", [])
    action_error = state.get("action_error")

    correction_needed = False
    reason_str = ""

    if action_error:
        correction_needed = True
        reason_str = f"Action error: {action_error}"
    else:
        for act_result in executed:
            if isinstance(act_result.get("result"), str) and "clamped from" in act_result["result"].lower():
                correction_needed = True
                reason_str = "Setpoint was clamped — LLM asked for out-of-range value"
                break

    # Cap self-correction iterations
    if state.get("correction_iteration", 0) >= config.MAX_CORRECTION_ITERATIONS:
        correction_needed = False

    node_history = state.get("node_history", []) + ["verify"]
    return {
        "correction_needed": correction_needed,
        "correction_reason": reason_str,
        "node_history": node_history,
    }


# ---------------------------------------------------------------------------
# Node: Correct
# ---------------------------------------------------------------------------
def correct(state: AgentState) -> AgentState:
    """Ask the LLM for a corrective action."""
    llm = get_llm_client()
    persistence = get_persistence()

    executed = state.get("executed_actions", [{}])
    previous_action = {
        "kind": executed[-1].get("name", "?"),
        "value_after": executed[-1].get("args", {}).get("temperature_c"),
        "reasoning": executed[-1].get("args", {}).get("reasoning", ""),
    }
    outcome = {
        "summary": state.get("correction_reason", ""),
        "metric": "action_validity",
        "before": "ok",
        "after": "flagged",
    }

    user_prompt = build_correct_prompt(previous_action, outcome)
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ]
    response = llm.invoke(messages)

    snapshot = state.get("pre_snapshot", {})
    persistence.log_llm_call(
        sim_time=snapshot.get("time", {}),
        node_name="correct",
        prompt={"user_len": len(user_prompt)},
        response=response.text,
        tool_calls=[{"name": tc.get("name"), "args": tc.get("args")}
                    for tc in response.tool_calls],
        latency_ms=response.latency_ms,
        tokens_in=response.tokens_in,
        tokens_out=response.tokens_out,
        error=response.error,
    )

    node_history = state.get("node_history", []) + ["correct"]
    return {
        "tool_calls": response.tool_calls,       # so Act can run again
        "llm_text": response.text,
        "correction_iteration": state.get("correction_iteration", 0) + 1,
        "node_history": node_history,
    }


# ---------------------------------------------------------------------------
# Conditional edge: after verify, either loop back or exit
# ---------------------------------------------------------------------------
def route_after_verify(state: AgentState) -> str:
    """Return the name of the next node."""
    if state.get("correction_needed"):
        return "correct"
    return "END"