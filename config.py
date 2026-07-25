"""
Global configuration for the Eco-Loop Building Agents project.
All paths, model IDs, safety bounds, and timing constants live here
so no other module hardcodes them.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.resolve()
ENERGYPLUS_DIR = Path(r"D:\EnergyPlus\Energy_Plus")

IDF_DIR = PROJECT_ROOT / "idf"
BASELINE_IDF = IDF_DIR / "baseline_small_office.idf"
AI_MODIFIED_IDF = IDF_DIR / "ai_modified.idf"  # generated at runtime
WEATHER_EPW = IDF_DIR / "bangalore.epw"

LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(exist_ok=True)

BASELINE_OUTPUT_DIR = LOGS_DIR / "baseline_run"
AI_OUTPUT_DIR = LOGS_DIR / "ai_run"
EVENTS_DB = LOGS_DIR / "events.sqlite"

# -----------------------------------------------------------------------------
# LLM configuration
# -----------------------------------------------------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama-3.1-8b-instant"  # fast, free tier, good tool-calling

LLM_MAX_RETRIES = 3
LLM_TIMEOUT_SECONDS = 30
LLM_TEMPERATURE = 0.2  # low for consistent control decisions

# -----------------------------------------------------------------------------
# Simulation timing
# -----------------------------------------------------------------------------
SIMULATION_DAYS = 7             # one week of July simulated
SIMULATION_START_MONTH = 7      # July
SIMULATION_START_DAY = 1
SIMULATION_END_MONTH = 7
SIMULATION_END_DAY = 7

# EnergyPlus internal timestep (minutes). 15 is a good default.
EP_TIMESTEP_MINUTES = 15

# How often the LLM agent runs (in EnergyPlus timesteps).
# 4 timesteps × 15 min = 1 sim-hour between LLM decisions.
# This keeps Groq quota reasonable and simulations fast.
LLM_DECISION_INTERVAL_STEPS = 4

# Max self-correction iterations per LLM decision cycle
MAX_CORRECTION_ITERATIONS = 2

# -----------------------------------------------------------------------------
# Safety bounds (hard limits on LLM actuator writes)
# -----------------------------------------------------------------------------
THERMOSTAT_MIN_C = 20.0
THERMOSTAT_MAX_C = 28.0
LIGHTING_MIN_FRACTION = 0.0
LIGHTING_MAX_FRACTION = 1.0

# Thermal comfort target band (used by verify node + dashboard)
COMFORT_TEMP_MIN_C = 21.0
COMFORT_TEMP_MAX_C = 25.0
# -----------------------------------------------------------------------------
# EnergyPlus actuator + variable definitions
# -----------------------------------------------------------------------------
# Variables we read from EnergyPlus at each timestep.
# Format: (ep_variable_name, ep_key_value) → EnergyPlus API needs both.
SENSOR_VARIABLES = {
    "zone_temp": "Zone Mean Air Temperature",          # per zone
    "zone_occupancy": "Zone People Occupant Count",    # per zone
    "zone_cooling_rate": "Zone Air System Sensible Cooling Rate",  # per zone
    "zone_heating_rate": "Zone Air System Sensible Heating Rate",  # per zone
    "outdoor_temp": "Site Outdoor Air Drybulb Temperature",  # single value
}

# Actuators we write to. Small office uses ThermostatSetpoint:DualSetpoint.
# We override the heating and cooling setpoint schedules directly.
# Actual actuator handles resolved at runtime in handles.py.
ACTUATOR_COOLING_SETPOINT_SCHEDULE = "CLGSETP_SCH"
ACTUATOR_HEATING_SETPOINT_SCHEDULE = "HTGSETP_SCH"
# -----------------------------------------------------------------------------
# Zone mapping (populated after IDF inspection in Step 2 of ep_driver)
# -----------------------------------------------------------------------------
# For the DOE Small Office reference building, zones are typically named:
ZONE_NAMES = [
    "Core_ZN",
    "Perimeter_ZN_1",
    "Perimeter_ZN_2",
    "Perimeter_ZN_3",
    "Perimeter_ZN_4",
]

# -----------------------------------------------------------------------------
# Sanity check on startup
# -----------------------------------------------------------------------------
def validate_config():
    """Called at startup to catch config errors before the simulation begins."""
    errors = []
    if not ENERGYPLUS_DIR.exists():
        errors.append(f"EnergyPlus directory not found: {ENERGYPLUS_DIR}")
    if not BASELINE_IDF.exists():
        errors.append(f"Baseline IDF not found: {BASELINE_IDF}")
    if not WEATHER_EPW.exists():
        errors.append(f"Weather EPW not found: {WEATHER_EPW}")
    if not GROQ_API_KEY:
        errors.append("GROQ_API_KEY not set in .env file")
    if errors:
        raise RuntimeError("Config validation failed:\n  - " + "\n  - ".join(errors))
    return True


if __name__ == "__main__":
    # Run `python config.py` to sanity-check paths and env vars
    validate_config()
    print("[OK] Config valid.")
    print(f"  EnergyPlus: {ENERGYPLUS_DIR}")
    print(f"  IDF:        {BASELINE_IDF}")
    print(f"  EPW:        {WEATHER_EPW}")
    print(f"  Groq model: {GROQ_MODEL}")
    print(f"  Groq key:   {'set' if GROQ_API_KEY else 'MISSING'}")