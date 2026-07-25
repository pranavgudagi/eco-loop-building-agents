"""
EnergyPlus driver module.

Wraps pyenergyplus with a callback pattern so the LangGraph agent can
inspect state and inject setpoints while the simulation is running.

The rest of the system (MCP tools, agent, dashboard) never touches
pyenergyplus directly — everything goes through EPDriver.
"""

import sys
import os
from pathlib import Path
from typing import Callable, Optional, Any

import config

# Register EnergyPlus's pyenergyplus module before importing it
sys.path.insert(0, str(config.ENERGYPLUS_DIR))
from pyenergyplus.api import EnergyPlusAPI  # noqa: E402


class EPDriver:
    """
    Manages an EnergyPlus simulation with a user-provided control callback.

    Usage:
        def my_control(driver):
            temp = driver.read_variable("zone_temp", "Core_ZN")
            driver.write_setpoint("cooling", 24.0)

        driver = EPDriver(idf_path, epw_path, output_dir)
        driver.set_control_callback(my_control)
        exit_code = driver.run()
    """

    def __init__(
        self,
        idf_path: Path,
        epw_path: Path,
        output_dir: Path,
        control_callback: Optional[Callable] = None,
        step_interval: int = 4,  # invoke callback every N timesteps
    ):
        self.idf_path = Path(idf_path)
        self.epw_path = Path(epw_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # EnergyPlus API objects
        self.api = EnergyPlusAPI()
        self.state = self.api.state_manager.new_state()

        # Handles (cached after first callback invocation)
        self.handles_ready = False
        self.variable_handles: dict = {}   # {(sensor_name, key): handle}
        self.actuator_handles: dict = {}   # {actuator_name: handle}

        # Control
        self.control_callback = control_callback
        self.step_interval = step_interval
        self.timestep_counter = 0

        # Diagnostics
        self.callback_invocations = 0
        self.warnings_seen = 0

    # -------------------------------------------------------------------------
    # Public API for other modules
    # -------------------------------------------------------------------------
    def set_control_callback(self, callback: Callable):
        """Register the function invoked every `step_interval` timesteps."""
        self.control_callback = callback

    def read_variable(self, sensor_key: str, zone_or_key: str = "Environment") -> Optional[float]:
        """
        Read a sensor value from the running simulation.

        sensor_key: one of config.SENSOR_VARIABLES keys (e.g. "zone_temp").
        zone_or_key: the zone name for per-zone vars, or "Environment" for site vars.
        Returns None if the handle isn't ready yet (during warmup).
        """
        if not self.handles_ready:
            return None
        ep_var_name = config.SENSOR_VARIABLES.get(sensor_key)
        if ep_var_name is None:
            raise ValueError(f"Unknown sensor: {sensor_key}")
        handle = self.variable_handles.get((sensor_key, zone_or_key))
        if handle is None or handle < 0:
            return None
        return self.api.exchange.get_variable_value(self.state, handle)

    def write_setpoint(self, kind: str, value: float) -> bool:
        """
        Override a thermostat setpoint schedule.

        kind: "cooling" or "heating"
        value: temperature in Celsius
        Returns True on success, False if handle not ready.
        """
        if not self.handles_ready:
            return False
        actuator_key = {
            "cooling": config.ACTUATOR_COOLING_SETPOINT_SCHEDULE,
            "heating": config.ACTUATOR_HEATING_SETPOINT_SCHEDULE,
        }.get(kind)
        if actuator_key is None:
            raise ValueError(f"Unknown setpoint kind: {kind}")
        handle = self.actuator_handles.get(actuator_key)
        if handle is None or handle < 0:
            return False
        self.api.exchange.set_actuator_value(self.state, handle, float(value))
        return True

    def current_sim_time(self) -> dict:
        """Return current simulated time as a dict."""
        return {
            "month": self.api.exchange.month(self.state),
            "day": self.api.exchange.day_of_month(self.state),
            "hour": self.api.exchange.hour(self.state),
            "minute": self.api.exchange.minutes(self.state),
            "day_of_year": self.api.exchange.day_of_year(self.state),
        }

    def is_warmup(self) -> bool:
        """True while EnergyPlus is in warmup phase (skip control during warmup)."""
        return self.api.exchange.warmup_flag(self.state)

    # -------------------------------------------------------------------------
    # Internal: callback wiring
    # -------------------------------------------------------------------------
    def _resolve_handles(self):
        """Fetch and cache all variable/actuator handles once at start of run."""
        # Variables: iterate zones × sensor types
        for sensor_key, ep_var_name in config.SENSOR_VARIABLES.items():
            if sensor_key == "outdoor_temp":
                # Site-level variable, key = "Environment"
                handle = self.api.exchange.get_variable_handle(
                    self.state, ep_var_name, "Environment"
                )
                self.variable_handles[(sensor_key, "Environment")] = handle
            else:
                # Per-zone variable
                for zone in config.ZONE_NAMES:
                    handle = self.api.exchange.get_variable_handle(
                        self.state, ep_var_name, zone
                    )
                    self.variable_handles[(sensor_key, zone)] = handle

        # Actuators: cooling + heating setpoint schedules
        # Actuator signature: (component_type, control_type, actuator_key)
        for actuator_key in [
            config.ACTUATOR_COOLING_SETPOINT_SCHEDULE,
            config.ACTUATOR_HEATING_SETPOINT_SCHEDULE,
        ]:
            handle = self.api.exchange.get_actuator_handle(
                self.state, "Schedule:Compact", "Schedule Value", actuator_key
            )
            self.actuator_handles[actuator_key] = handle

        # Sanity report
        bad_vars = [k for k, v in self.variable_handles.items() if v < 0]
        bad_acts = [k for k, v in self.actuator_handles.items() if v < 0]
        if bad_vars:
            print(f"[WARN] Unresolved variable handles: {bad_vars}")
        if bad_acts:
            print(f"[WARN] Unresolved actuator handles: {bad_acts}")

        self.handles_ready = True
        print(f"[EPDriver] Handles resolved. "
              f"Vars: {len(self.variable_handles)}, "
              f"Actuators: {len(self.actuator_handles)}")

    def _timestep_callback(self, state):
        """
        Fired by EnergyPlus at the end of every zone timestep.
        We resolve handles on first call, then invoke user callback every N steps.
        """
        # Skip warmup phase entirely
        if self.api.exchange.warmup_flag(state):
            return

        # Skip until data-ready (handles can only be fetched after warmup starts)
        if not self.api.exchange.api_data_fully_ready(state):
            return

        # First real timestep after warmup: resolve handles
        if not self.handles_ready:
            self._resolve_handles()

        # Rate-limit the user callback
        self.timestep_counter += 1
        if self.timestep_counter % self.step_interval != 0:
            return

        if self.control_callback is None:
            return

        try:
            self.callback_invocations += 1
            self.control_callback(self)
        except Exception as e:
            # Never let a callback failure crash the simulation
            print(f"[EPDriver] Control callback error at step "
                  f"{self.timestep_counter}: {e}")

    # -------------------------------------------------------------------------
    # Run
    # -------------------------------------------------------------------------
    def run(self) -> int:
        """
        Execute the full simulation. Returns EnergyPlus exit code (0 = success).
        """
        # Register the callback BEFORE run_energyplus is called
        self.api.runtime.callback_end_zone_timestep_after_zone_reporting(
            self.state, self._timestep_callback
        )

        args = [
            "-w", str(self.epw_path),
            "-d", str(self.output_dir),
            str(self.idf_path),
        ]
        print(f"[EPDriver] Running simulation")
        print(f"  IDF:    {self.idf_path.name}")
        print(f"  EPW:    {self.epw_path.name}")
        print(f"  Output: {self.output_dir}")

        exit_code = self.api.runtime.run_energyplus(self.state, args)

        print(f"[EPDriver] Finished. Exit code: {exit_code}, "
              f"Callback invocations: {self.callback_invocations}")
        return exit_code


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    """
    Sanity test: run baseline IDF for one week, print zone temps every hour.
    No LLM, no MCP, no agent — just proving the callback + read/write plumbing works.
    """

    def simple_probe(driver: EPDriver):
        """Test callback: print current temps every invocation."""
        t = driver.current_sim_time()
        core_temp = driver.read_variable("zone_temp", "Core_ZN")
        outdoor = driver.read_variable("outdoor_temp", "Environment")
        if core_temp is not None:
            print(f"  [{t['month']:02d}/{t['day']:02d} {t['hour']:02d}:{t['minute']:02d}] "
                  f"Core: {core_temp:.2f}°C  Outdoor: {outdoor:.2f}°C")

    driver = EPDriver(
        idf_path=config.BASELINE_IDF,
        epw_path=config.WEATHER_EPW,
        output_dir=config.BASELINE_OUTPUT_DIR,
        control_callback=simple_probe,
        step_interval=4,  # every hour (4 × 15 min)
    )
    exit_code = driver.run()
    sys.exit(exit_code)