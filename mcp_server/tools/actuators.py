"""
Actuator tool functions exposed to the LLM.

All writes go through a safety wrapper that clamps values to configured
safe ranges, logs the pre/post state and any clamping event to persistence,
and never allows a raw LLM number to reach EnergyPlus.
"""

from typing import Optional

import config
from mcp_server import get_driver, get_persistence


def _clamp(value: float, lo: float, hi: float) -> tuple[float, bool]:
    """Clamp value to [lo, hi]. Returns (clamped_value, was_clamped)."""
    clamped = max(lo, min(hi, value))
    return clamped, clamped != value


def set_cooling_setpoint(temperature_c: float, reasoning: str = "") -> dict:
    """
    Override the building-wide cooling setpoint schedule.

    temperature_c: target temperature in Celsius (clamped to safe range)
    reasoning: brief explanation for the log (LLM should always fill this)

    Returns a dict describing what actually happened:
        {applied_value, was_clamped, original_request, ok}
    """
    driver = get_driver()
    persistence = get_persistence()

    clamped, was_clamped = _clamp(
        float(temperature_c),
        config.THERMOSTAT_MIN_C,
        config.THERMOSTAT_MAX_C,
    )

    ok = driver.write_setpoint("cooling", clamped)

    sim_time = driver.current_sim_time()
    persistence.log_action(
        sim_time=sim_time,
        kind="cooling_setpoint",
        target="global",
        value_before=temperature_c,
        value_after=clamped,
        reasoning=(reasoning + (" [CLAMPED]" if was_clamped else "")).strip(),
    )

    return {
        "applied_value": clamped,
        "was_clamped": was_clamped,
        "original_request": temperature_c,
        "safe_range": [config.THERMOSTAT_MIN_C, config.THERMOSTAT_MAX_C],
        "ok": ok,
    }


def set_heating_setpoint(temperature_c: float, reasoning: str = "") -> dict:
    """
    Override the building-wide heating setpoint schedule.

    Same interface and safety guarantees as set_cooling_setpoint.
    """
    driver = get_driver()
    persistence = get_persistence()

    clamped, was_clamped = _clamp(
        float(temperature_c),
        config.THERMOSTAT_MIN_C,
        config.THERMOSTAT_MAX_C,
    )

    ok = driver.write_setpoint("heating", clamped)

    sim_time = driver.current_sim_time()
    persistence.log_action(
        sim_time=sim_time,
        kind="heating_setpoint",
        target="global",
        value_before=temperature_c,
        value_after=clamped,
        reasoning=(reasoning + (" [CLAMPED]" if was_clamped else "")).strip(),
    )

    return {
        "applied_value": clamped,
        "was_clamped": was_clamped,
        "original_request": temperature_c,
        "safe_range": [config.THERMOSTAT_MIN_C, config.THERMOSTAT_MAX_C],
        "ok": ok,
    }