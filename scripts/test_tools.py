"""
End-to-end test of the tool layer.

Runs one simulation while a control callback:
  1. Calls sensor tools to read state
  2. Calls actuator tools to write setpoints based on a simple rule
  3. Every write flows through the safety wrapper to persistence

Success criterion: simulation completes, SQLite has actions logged,
CSV has snapshots for every hour.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from simulation.ep_driver import EPDriver
from data.persistence import Persistence
from mcp_server import set_runtime
from mcp_server.tools.sensors import (
    get_full_snapshot, get_outdoor_temperature, get_all_zone_temperatures
)
from mcp_server.tools.actuators import set_cooling_setpoint


call_count = [0]


def rule_based_controller(driver: EPDriver):
    """
    Dumb rule to prove the loop works.

    Rule: if outdoor > 28°C, cool aggressively (setpoint 23°C).
          if outdoor < 22°C, ease off (setpoint 26°C).
          else, hold moderate (setpoint 24°C).

    This is NOT the AI — it's a stand-in so we can verify the tool
    layer works before wiring in the LLM.
    """
    call_count[0] += 1

    persistence = _persistence  # captured from outer scope below
    snapshot = get_full_snapshot()

    # Log the full snapshot every invocation
    persistence.log_snapshot(snapshot["time"], snapshot)

    outdoor = snapshot["outdoor_temp"]
    if outdoor is None:
        return

    if outdoor > 28.0:
        target = 23.0
        why = f"Outdoor hot ({outdoor:.1f}°C), cooling harder"
    elif outdoor < 22.0:
        target = 26.0
        why = f"Outdoor cool ({outdoor:.1f}°C), easing off"
    else:
        target = 24.0
        why = f"Outdoor moderate ({outdoor:.1f}°C), holding"

    result = set_cooling_setpoint(target, reasoning=why)

    # Print every 24th call (once per sim day) so terminal isn't flooded
    if call_count[0] % 24 == 0:
        t = snapshot["time"]
        print(f"  Day {t['day']:02d} {t['hour']:02d}h: "
              f"outdoor={outdoor:.1f}°C -> setpoint={result['applied_value']}°C ({why})")


if __name__ == "__main__":
    # Clean prior test outputs
    test_output = config.LOGS_DIR / "tool_test_run"
    test_db = config.LOGS_DIR / "tool_test.sqlite"
    for p in [test_db, config.LOGS_DIR / "tool_test_run_timeseries.csv"]:
        if p.exists():
            p.unlink()

    persistence = Persistence(run_id="tool_test_run", db_path=test_db)
    _persistence = persistence  # captured by controller closure

    driver = EPDriver(
        idf_path=config.BASELINE_IDF,
        epw_path=config.WEATHER_EPW,
        output_dir=test_output,
        step_interval=4,  # one call per sim hour
    )

    set_runtime(driver, persistence, run_id="tool_test_run")

    driver.set_control_callback(rule_based_controller)
    exit_code = driver.run()

    summary = persistence.summary()
    print()
    print(f"[SUMMARY]")
    print(f"  Exit code:       {exit_code}")
    print(f"  Controller runs: {call_count[0]}")
    print(f"  Snapshots:       {summary['snapshots']}")
    print(f"  Actions:         {summary['actions']}")
    print(f"  CSV file:        {persistence.csv_path}")
    print(f"  DB file:         {persistence.db_path}")

    if exit_code == 0 and summary['snapshots'] > 0 and summary['actions'] > 0:
        print(f"[SUCCESS] Full tool layer works end-to-end.")
    else:
        print(f"[FAIL] Something is off. Review counts above.")
        sys.exit(1)