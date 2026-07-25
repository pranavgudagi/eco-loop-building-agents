"""
Entry point.

Runs the full experiment:
  1. Baseline simulation (no agent, IDF default schedules).
  2. AI-controlled simulation (LangGraph agent driving via LLM).

Both write to persistence tagged with their run_id, so the dashboard
can compare them side by side.

Usage:
    python main.py                 # run both baseline and AI
    python main.py --baseline-only # skip the AI run
    python main.py --ai-only       # skip the baseline run
"""

import sys
import argparse
import time
from pathlib import Path

import config
from simulation.ep_driver import EPDriver
from data.persistence import Persistence
from mcp_server import set_runtime
from mcp_server.tools.sensors import get_full_snapshot
from agent.graph import build_agent_graph


BASELINE_RUN_ID = "baseline"
AI_RUN_ID = "ai"


# ---------------------------------------------------------------------------
# Baseline controller: snapshot only, no writes.
# ---------------------------------------------------------------------------
def make_baseline_callback(persistence: Persistence):
    """
    Baseline callback: log every snapshot, make no control decisions.
    EnergyPlus uses the IDF's default schedules.
    """
    def callback(driver: EPDriver):
        snapshot = get_full_snapshot()
        persistence.log_snapshot(snapshot["time"], snapshot)
    return callback


# ---------------------------------------------------------------------------
# AI controller: snapshot + invoke the LangGraph agent.
# ---------------------------------------------------------------------------
def make_ai_callback(persistence: Persistence):
    """
    AI callback: log snapshot, then invoke the agent graph which
    reads sensors, decides, and writes actuators.
    """
    graph = build_agent_graph()

    invocation_count = [0]
    error_count = [0]

    def callback(driver: EPDriver):
        # Always log the snapshot (before the agent acts)
        snapshot = get_full_snapshot()
        persistence.log_snapshot(snapshot["time"], snapshot)

        # Invoke the agent
        invocation_count[0] += 1
        try:
            result = graph.invoke(
                {"correction_iteration": 0, "node_history": []},
                config={"recursion_limit": 15},
            )
            # Progress ping every 24 invocations (~ once per sim day)
            if invocation_count[0] % 24 == 0:
                t = snapshot.get("time", {})
                path = " -> ".join(result.get("node_history", []))
                n_calls = len(result.get("tool_calls", []) or [])
                print(f"  [Day {t.get('day', '?'):02d} "
                      f"{t.get('hour', '?'):02d}h] "
                      f"path={path} tool_calls={n_calls}")
        except Exception as e:
            # Agent failure never crashes the simulation
            error_count[0] += 1
            print(f"[main] Agent error (#{error_count[0]}) at "
                  f"{snapshot.get('time', {})}: {type(e).__name__}: {e}")

    return callback, invocation_count, error_count


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------
def run_baseline():
    print()
    print("=" * 70)
    print(f"BASELINE RUN")
    print("=" * 70)

    output_dir = config.LOGS_DIR / f"{BASELINE_RUN_ID}_run"
    persistence = Persistence(run_id=BASELINE_RUN_ID)

    driver = EPDriver(
        idf_path=config.BASELINE_IDF,
        epw_path=config.WEATHER_EPW,
        output_dir=output_dir,
        step_interval=config.LLM_DECISION_INTERVAL_STEPS,
    )
    set_runtime(driver, persistence, run_id=BASELINE_RUN_ID)
    driver.set_control_callback(make_baseline_callback(persistence))

    t0 = time.perf_counter()
    exit_code = driver.run()
    elapsed = time.perf_counter() - t0

    summary = persistence.summary()
    print()
    print(f"[BASELINE] Exit code: {exit_code}")
    print(f"[BASELINE] Wall time: {elapsed:.1f}s")
    print(f"[BASELINE] Snapshots: {summary['snapshots']}")
    print(f"[BASELINE] Actions:   {summary['actions']}  (expected 0)")
    return exit_code


def run_ai():
    print()
    print("=" * 70)
    print(f"AI-CONTROLLED RUN")
    print("=" * 70)

    output_dir = config.LOGS_DIR / f"{AI_RUN_ID}_run"
    persistence = Persistence(run_id=AI_RUN_ID)

    driver = EPDriver(
        idf_path=config.BASELINE_IDF,
        epw_path=config.WEATHER_EPW,
        output_dir=output_dir,
        step_interval=config.LLM_DECISION_INTERVAL_STEPS,
    )
    set_runtime(driver, persistence, run_id=AI_RUN_ID)

    callback, invocations, errors = make_ai_callback(persistence)
    driver.set_control_callback(callback)

    t0 = time.perf_counter()
    exit_code = driver.run()
    elapsed = time.perf_counter() - t0

    summary = persistence.summary()
    print()
    print(f"[AI] Exit code:        {exit_code}")
    print(f"[AI] Wall time:        {elapsed:.1f}s")
    print(f"[AI] Snapshots:        {summary['snapshots']}")
    print(f"[AI] LLM calls:        {summary['llm_calls']}")
    print(f"[AI] Actions logged:   {summary['actions']}")
    print(f"[AI] Agent invocations: {invocations[0]}")
    print(f"[AI] Agent errors:     {errors[0]}")
    return exit_code


def main():
    parser = argparse.ArgumentParser(description="Eco-Loop Building Agents")
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--ai-only", action="store_true")
    args = parser.parse_args()

    # Validate config first — bail early if anything is wrong
    config.validate_config()
    print(f"[OK] Config valid. Model: {config.GROQ_MODEL}, "
          f"days: {config.SIMULATION_DAYS}")

    exit_codes = []
    if not args.ai_only:
        exit_codes.append(run_baseline())
    if not args.baseline_only:
        exit_codes.append(run_ai())

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)
    print(f"CSV files: {config.LOGS_DIR / (BASELINE_RUN_ID + '_timeseries.csv')}")
    print(f"           {config.LOGS_DIR / (AI_RUN_ID + '_timeseries.csv')}")
    print(f"DB:        {config.EVENTS_DB}")
    print()
    print("Next step: streamlit run dashboard/app.py")

    if any(c != 0 for c in exit_codes):
        sys.exit(1)


if __name__ == "__main__":
    main()