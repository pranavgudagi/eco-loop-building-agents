"""
Sensor tool functions exposed to the LLM.

Each function returns a compact, LLM-friendly value (float, dict, or list).
All functions read from the live EPDriver; if a handle isn't ready yet
(during warmup), they return None or a sentinel dict.
"""

from typing import Optional

import config
from mcp_server import get_driver


def get_zone_temperature(zone_id: str) -> Optional[float]:
    """
    Return the current mean air temperature of a zone in Celsius.

    zone_id: one of Core_ZN, Perimeter_ZN_1..4
    """
    driver = get_driver()
    return driver.read_variable("zone_temp", zone_id)


def get_all_zone_temperatures() -> dict:
    """Return {zone_id: temp_c} for all controllable zones."""
    driver = get_driver()
    return {
        zone: driver.read_variable("zone_temp", zone)
        for zone in config.ZONE_NAMES
    }


def get_zone_occupancy(zone_id: str) -> Optional[float]:
    """Return the current occupant count for a zone (may be fractional)."""
    driver = get_driver()
    return driver.read_variable("zone_occupancy", zone_id)


def get_all_zone_occupancy() -> dict:
    """Return {zone_id: occupant_count} for all controllable zones."""
    driver = get_driver()
    return {
        zone: driver.read_variable("zone_occupancy", zone)
        for zone in config.ZONE_NAMES
    }


def get_zone_cooling_rate(zone_id: str) -> Optional[float]:
    """Return the current sensible cooling rate for a zone in Watts."""
    driver = get_driver()
    return driver.read_variable("zone_cooling_rate", zone_id)


def get_zone_heating_rate(zone_id: str) -> Optional[float]:
    """Return the current sensible heating rate for a zone in Watts."""
    driver = get_driver()
    return driver.read_variable("zone_heating_rate", zone_id)


def get_total_hvac_power() -> dict:
    """Return {cooling_watts, heating_watts, total_watts} summed across zones."""
    driver = get_driver()
    cooling = 0.0
    heating = 0.0
    for zone in config.ZONE_NAMES:
        c = driver.read_variable("zone_cooling_rate", zone) or 0.0
        h = driver.read_variable("zone_heating_rate", zone) or 0.0
        cooling += c
        heating += h
    return {
        "cooling_watts": round(cooling, 1),
        "heating_watts": round(heating, 1),
        "total_watts": round(cooling + heating, 1),
    }


def get_outdoor_temperature() -> Optional[float]:
    """Return the current outdoor dry-bulb temperature in Celsius."""
    driver = get_driver()
    return driver.read_variable("outdoor_temp", "Environment")


def get_sim_time() -> dict:
    """Return the current simulated clock time."""
    driver = get_driver()
    return driver.current_sim_time()


def get_full_snapshot() -> dict:
    """
    One-call convenience: returns everything the LLM typically needs.

    Structure:
        {
          "time": {month, day, hour, minute},
          "zones": {zone_id: {temp, occupancy, cooling_rate, heating_rate}},
          "outdoor_temp": float,
          "total_hvac_power": {cooling_watts, heating_watts, total_watts}
        }
    """
    driver = get_driver()
    zones = {}
    for zone in config.ZONE_NAMES:
        zones[zone] = {
            "temp": driver.read_variable("zone_temp", zone),
            "occupancy": driver.read_variable("zone_occupancy", zone),
            "cooling_rate": driver.read_variable("zone_cooling_rate", zone),
            "heating_rate": driver.read_variable("zone_heating_rate", zone),
        }
    return {
        "time": driver.current_sim_time(),
        "zones": zones,
        "outdoor_temp": driver.read_variable("outdoor_temp", "Environment"),
        "total_hvac_power": get_total_hvac_power(),
    }