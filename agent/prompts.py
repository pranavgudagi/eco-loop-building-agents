"""
System and user prompt templates for the closed-loop building control agent.

Design principles:
  1. System prompt teaches WHAT the agent optimizes and HOW to reason.
  2. Perceive/Reason prompts feed the LLM a COMPACT, structured snapshot
     rather than raw simulation logs (managing prompt length is a graded criterion).
  3. Correct prompt reuses the previous action + observed outcome, teaching
     self-correction without an extra system message.
  4. Every prompt asks the LLM to explain its reasoning as part of the tool
     call, so the action log tells the full story to judges.

Prompt engineering notes for docs/ARCHITECTURE.md:
  - Temperature 0.2: control tasks reward consistency over creativity.
  - Structured JSON state: LLM parses lists/dicts more reliably than prose.
  - Explicit reasoning field on every write: forces the LLM to justify actions,
    which both grounds its decisions and gives us auditable logs.
  - Bounded action space: system prompt names the safe range and reminds the
    LLM that outputs will be clamped, so the LLM learns to stay inside.
"""

import json
import config


# ---------------------------------------------------------------------------
# System prompt (shown once per invocation)
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = f"""You are an HVAC control agent for a small office in Bangalore, India.
Your job every simulated hour is to reduce cooling energy MODESTLY while keeping
occupied-hours comfort above 90%.

CRITICAL PHYSICS:
  - HIGHER cooling setpoint = LESS cooling energy consumed.
  - LOWER cooling setpoint = MORE cooling energy consumed.

TARGET:
  - Save 5-15% cooling energy vs the 24°C baseline.
  - Keep 90-95% of occupied zone-hours inside [{config.COMFORT_TEMP_MIN_C}, {config.COMFORT_TEMP_MAX_C}]°C.
  - AVOID setpoints above 25.5°C during occupied hours — they cause comfort violations.

CRITICAL RULE — OCCUPIED HOURS ARE SACRED:
  Between 7 AM and 6 PM on weekdays, occupants may be present.
  During these hours, cooling setpoint MUST stay between 23.5°C and 25.0°C.
  Do NOT apply aggressive setbacks during these hours even if occupancy sensor reads 0
  (the sensor may lag actual arrivals).

DECISION RULES (apply in order):

  RULE 1 — NIGHT SETBACK (hours 22-05):
    Set cooling to 26.0°C. Nobody is there, setpoint can drift.

  RULE 2 — WEEKEND SETBACK (Sat/Sun any hour):
    Set cooling to 26.0°C. Building unoccupied.

  RULE 3 — EARLY MORNING PREP (06:00):
    Set cooling to 24.5°C. Anticipate arrivals; gentle pre-cool.

  RULE 4 — OCCUPIED HOURS BASE (07:00-17:59 weekdays):
    Default to cooling = 24.5°C — a small +0.5°C nudge above baseline saves modest energy.
    If any zone is > 25.2°C, switch to 24.0°C.
    If ALL zones are < 24.0°C AND outdoor < 24°C, allow 25.0°C briefly.

  RULE 5 — EVENING WIND-DOWN (18:00-21:59):
    Set cooling to 25.5°C. Occupancy tapering, small setback OK.

  RULE 6 — DO NOTHING:
    If your last action already matches the rule for this hour, do NOT call any tool.

HARD BOUNDS:
  - Setpoints clamped to [{config.THERMOSTAT_MIN_C}, {config.THERMOSTAT_MAX_C}]°C.
  - Never call heating tool (Bangalore in July doesn't need heat).
  - 'reasoning' argument REQUIRED, one sentence naming the rule.

Example good reasoning: "Rule 4: occupied hours, holding 24.5°C for modest savings."
Example bad reasoning: "All zones empty, raising to 27°C." (WRONG during occupied hours)
"""
# ---------------------------------------------------------------------------
# Perceive/Reason prompt (per invocation)
# ---------------------------------------------------------------------------
REASON_TEMPLATE = """CURRENT STATE (sim time {sim_time}):

Outdoor: {outdoor_temp}°C
HVAC power draw: cooling={cooling_w}W, heating={heating_w}W

Zones:
{zones_block}

Recent history (last 3 hours):
{history_block}

Recent LLM decisions:
{recent_actions_block}

Decide whether to change any setpoint. If yes, call the appropriate tool
with a brief reasoning argument. If the current state is fine, respond
with a one-line note explaining why no change is needed."""


# ---------------------------------------------------------------------------
# Correction prompt (used only when verify detects degradation)
# ---------------------------------------------------------------------------
CORRECT_TEMPLATE = """CORRECTION NEEDED.

Your previous action:
  {previous_action}
  Reasoning: {previous_reasoning}

Observed outcome after {check_delay_min} minutes:
  {outcome_summary}

The outcome degraded {metric_name} (was {before_value}, now {after_value}).
Propose a corrective action, or revert to the previous setpoint if that is safer.
Explain what you now think went wrong and how the correction addresses it."""


# ---------------------------------------------------------------------------
# Formatting helpers (called by nodes.py)
# ---------------------------------------------------------------------------
def format_zones_block(zones: dict) -> str:
    """Turn the snapshot's zones dict into a compact LLM-readable block."""
    lines = []
    for zone_id, data in zones.items():
        temp = data.get("temp")
        occ = data.get("occupancy")
        cool = data.get("cooling_rate")
        heat = data.get("heating_rate")
        temp_str = f"{temp:.1f}°C" if temp is not None else "n/a"
        occ_str = f"{occ:.1f}" if occ is not None else "n/a"
        power = (cool or 0) + (heat or 0)
        lines.append(f"  {zone_id}: {temp_str}, occupants={occ_str}, hvac={power:.0f}W")
    return "\n".join(lines)


def format_history_block(history: list) -> str:
    """
    Format the rolling history (list of past snapshots) as terse trend lines.
    Each entry: {time, avg_temp, outdoor, total_power}
    """
    if not history:
        return "  (no history yet)"
    lines = []
    for h in history[-3:]:
        t = h.get("time", {})
        lines.append(
            f"  {t.get('day', '?'):02d} {t.get('hour', '?'):02d}h: "
            f"avg_indoor={h.get('avg_temp', 0):.1f}°C, "
            f"outdoor={h.get('outdoor', 0):.1f}°C, "
            f"hvac={h.get('total_power', 0):.0f}W"
        )
    return "\n".join(lines)


def format_recent_actions_block(actions: list) -> str:
    """
    Format the last few actions (from persistence) so the LLM remembers
    what it recently did. Prevents oscillation.
    """
    if not actions:
        return "  (none yet — this is the first decision cycle)"
    lines = []
    for a in actions[-3:]:
        lines.append(
            f"  {a.get('sim_day', '?'):02d} {a.get('sim_hour', '?'):02d}h: "
            f"{a.get('kind', '?')} -> {a.get('value_after', '?')}°C "
            f"({a.get('reasoning', '')[:60]})"
        )
    return "\n".join(lines)


def build_reason_prompt(snapshot: dict, history: list, recent_actions: list) -> str:
    """Assemble the Reason node's user message."""
    t = snapshot.get("time", {})
    sim_time_str = f"{t.get('month', '?'):02d}/{t.get('day', '?'):02d} " \
                   f"{t.get('hour', '?'):02d}:{t.get('minute', 0):02d}"

    outdoor = snapshot.get("outdoor_temp")
    outdoor_str = f"{outdoor:.1f}" if outdoor is not None else "n/a"

    hvac = snapshot.get("total_hvac_power", {})

    return REASON_TEMPLATE.format(
        sim_time=sim_time_str,
        outdoor_temp=outdoor_str,
        cooling_w=hvac.get("cooling_watts", 0),
        heating_w=hvac.get("heating_watts", 0),
        zones_block=format_zones_block(snapshot.get("zones", {})),
        history_block=format_history_block(history),
        recent_actions_block=format_recent_actions_block(recent_actions),
    )


def build_correct_prompt(previous_action: dict, outcome: dict) -> str:
    """Assemble the Correct node's user message."""
    return CORRECT_TEMPLATE.format(
        previous_action=f"{previous_action.get('kind')} -> "
                        f"{previous_action.get('value_after')}°C",
        previous_reasoning=previous_action.get("reasoning", "(none provided)"),
        check_delay_min=config.EP_TIMESTEP_MINUTES * config.LLM_DECISION_INTERVAL_STEPS,
        outcome_summary=outcome.get("summary", ""),
        metric_name=outcome.get("metric", "state"),
        before_value=outcome.get("before", "?"),
        after_value=outcome.get("after", "?"),
    )


# ---------------------------------------------------------------------------
# Standalone test: verify prompts render cleanly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sample_snapshot = {
        "time": {"month": 7, "day": 3, "hour": 14, "minute": 0},
        "zones": {
            "Core_ZN": {"temp": 24.2, "occupancy": 8, "cooling_rate": 1200, "heating_rate": 0},
            "Perimeter_ZN_1": {"temp": 25.5, "occupancy": 3, "cooling_rate": 900, "heating_rate": 0},
            "Perimeter_ZN_2": {"temp": 25.1, "occupancy": 3, "cooling_rate": 850, "heating_rate": 0},
            "Perimeter_ZN_3": {"temp": 24.8, "occupancy": 2, "cooling_rate": 700, "heating_rate": 0},
            "Perimeter_ZN_4": {"temp": 24.9, "occupancy": 4, "cooling_rate": 950, "heating_rate": 0},
        },
        "outdoor_temp": 27.8,
        "total_hvac_power": {"cooling_watts": 4600, "heating_watts": 0, "total_watts": 4600},
    }
    sample_history = [
        {"time": {"day": 3, "hour": 12}, "avg_temp": 24.4, "outdoor": 26.5, "total_power": 4200},
        {"time": {"day": 3, "hour": 13}, "avg_temp": 24.6, "outdoor": 27.2, "total_power": 4400},
    ]
    sample_actions = [
        {"sim_day": 3, "sim_hour": 12, "kind": "cooling_setpoint",
         "value_after": 24.0, "reasoning": "Occupied hours, holding standard cooling"},
    ]

    print("=" * 70)
    print("SYSTEM PROMPT")
    print("=" * 70)
    print(SYSTEM_PROMPT)
    print()
    print("=" * 70)
    print("REASON PROMPT (rendered)")
    print("=" * 70)
    print(build_reason_prompt(sample_snapshot, sample_history, sample_actions))
    print()
    print("=" * 70)
    print("CORRECT PROMPT (rendered)")
    print("=" * 70)
    print(build_correct_prompt(
        previous_action={
            "kind": "cooling_setpoint", "value_after": 22.0,
            "reasoning": "Anticipating heat spike",
        },
        outcome={
            "summary": "Total HVAC power increased sharply; occupancy is 0 in most zones",
            "metric": "hvac_power",
            "before": "3800W", "after": "5600W",
        },
    ))
    print()
    print("[OK] All prompt templates render.")