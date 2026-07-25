"""
Persistence layer.

Writes three streams to SQLite so the dashboard and post-hoc analysis
have structured, queryable data:
  - snapshots:   sensor readings at each timestep
  - actions:     every actuator write (with pre/post state)
  - llm_calls:   every LLM invocation (prompt, response, latency, tokens)

Also mirrors timeseries data to CSV for easy pandas ingestion by Streamlit.
"""

import sqlite3
import json
import time
from pathlib import Path
from contextlib import contextmanager
from typing import Optional

import config


SCHEMA = """
CREATE TABLE IF NOT EXISTS snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    sim_month INTEGER,
    sim_day INTEGER,
    sim_hour INTEGER,
    sim_minute INTEGER,
    real_time REAL,
    payload_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS actions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    sim_month INTEGER,
    sim_day INTEGER,
    sim_hour INTEGER,
    kind TEXT NOT NULL,
    target TEXT NOT NULL,
    value_before REAL,
    value_after REAL,
    reasoning TEXT,
    real_time REAL
);

CREATE TABLE IF NOT EXISTS llm_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    sim_month INTEGER,
    sim_day INTEGER,
    sim_hour INTEGER,
    node_name TEXT,
    prompt_json TEXT,
    response_text TEXT,
    tool_calls_json TEXT,
    latency_ms REAL,
    tokens_in INTEGER,
    tokens_out INTEGER,
    error TEXT,
    real_time REAL
);

CREATE INDEX IF NOT EXISTS idx_snapshots_run ON snapshots(run_id, sim_month, sim_day, sim_hour);
CREATE INDEX IF NOT EXISTS idx_actions_run ON actions(run_id);
CREATE INDEX IF NOT EXISTS idx_llm_run ON llm_calls(run_id);
"""


class Persistence:
    """SQLite-backed event store. One instance per run."""

    def __init__(self, run_id: str, db_path: Optional[Path] = None, clear: bool = True):
        self.run_id = run_id
        self.db_path = Path(db_path) if db_path else config.EVENTS_DB
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()
        if clear:
            self._clear_run()

        # CSV mirror for time-series (Streamlit reads this directly)
        self.csv_path = self.db_path.parent / f"{run_id}_timeseries.csv"
        if clear and self.csv_path.exists():
            self.csv_path.unlink()
        self._init_csv()

    def _clear_run(self):
        """Delete any pre-existing rows for this run_id."""
        with self._conn() as c:
            c.execute("DELETE FROM snapshots WHERE run_id=?", (self.run_id,))
            c.execute("DELETE FROM actions WHERE run_id=?", (self.run_id,))
            c.execute("DELETE FROM llm_calls WHERE run_id=?", (self.run_id,))

    def _init_schema(self):
        with self._conn() as c:
            c.executescript(SCHEMA)

    def _init_csv(self):
        if not self.csv_path.exists():
            with open(self.csv_path, "w") as f:
                f.write("sim_month,sim_day,sim_hour,sim_minute,"
                        "core_temp,perimeter_1_temp,perimeter_2_temp,"
                        "perimeter_3_temp,perimeter_4_temp,outdoor_temp,"
                        "total_cooling_w,total_heating_w\n")

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # -------------------------------------------------------------------------
    # Public write API
    # -------------------------------------------------------------------------
    def log_snapshot(self, sim_time: dict, payload: dict):
        """Record a full sensor snapshot at the given sim time."""
        with self._conn() as c:
            c.execute(
                "INSERT INTO snapshots (run_id, sim_month, sim_day, sim_hour, "
                "sim_minute, real_time, payload_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (self.run_id, sim_time.get("month"), sim_time.get("day"),
                 sim_time.get("hour"), sim_time.get("minute"), time.time(),
                 json.dumps(payload)),
            )

        # Mirror the key numeric fields to CSV for fast plotting
        zones = payload.get("zones", {})
        cooling_total = sum(z.get("cooling_rate", 0) or 0 for z in zones.values())
        heating_total = sum(z.get("heating_rate", 0) or 0 for z in zones.values())
        row = [
            sim_time.get("month", 0), sim_time.get("day", 0),
            sim_time.get("hour", 0), sim_time.get("minute", 0),
            zones.get("Core_ZN", {}).get("temp", ""),
            zones.get("Perimeter_ZN_1", {}).get("temp", ""),
            zones.get("Perimeter_ZN_2", {}).get("temp", ""),
            zones.get("Perimeter_ZN_3", {}).get("temp", ""),
            zones.get("Perimeter_ZN_4", {}).get("temp", ""),
            payload.get("outdoor_temp", ""),
            cooling_total, heating_total,
        ]
        with open(self.csv_path, "a") as f:
            f.write(",".join(str(v) for v in row) + "\n")

    def log_action(self, sim_time: dict, kind: str, target: str,
                   value_before: float, value_after: float, reasoning: str = ""):
        """Record every actuator write."""
        with self._conn() as c:
            c.execute(
                "INSERT INTO actions (run_id, sim_month, sim_day, sim_hour, "
                "kind, target, value_before, value_after, reasoning, real_time) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (self.run_id, sim_time.get("month"), sim_time.get("day"),
                 sim_time.get("hour"), kind, target, value_before, value_after,
                 reasoning, time.time()),
            )

    def log_llm_call(self, sim_time: dict, node_name: str, prompt: dict,
                     response: str, tool_calls: list, latency_ms: float,
                     tokens_in: int = 0, tokens_out: int = 0,
                     error: Optional[str] = None):
        """Record every LLM invocation."""
        with self._conn() as c:
            c.execute(
                "INSERT INTO llm_calls (run_id, sim_month, sim_day, sim_hour, "
                "node_name, prompt_json, response_text, tool_calls_json, "
                "latency_ms, tokens_in, tokens_out, error, real_time) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (self.run_id, sim_time.get("month"), sim_time.get("day"),
                 sim_time.get("hour"), node_name, json.dumps(prompt),
                 response, json.dumps(tool_calls), latency_ms,
                 tokens_in, tokens_out, error, time.time()),
            )

    # -------------------------------------------------------------------------
    # Public read API (used by dashboard)
    # -------------------------------------------------------------------------
    def summary(self) -> dict:
        with self._conn() as c:
            cur = c.cursor()
            cur.execute("SELECT COUNT(*) FROM snapshots WHERE run_id=?", (self.run_id,))
            n_snap = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM actions WHERE run_id=?", (self.run_id,))
            n_act = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM llm_calls WHERE run_id=?", (self.run_id,))
            n_llm = cur.fetchone()[0]
        return {"snapshots": n_snap, "actions": n_act, "llm_calls": n_llm}


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    """Quick sanity: write a few dummy events, read summary, verify CSV."""
    import os

    # Use a temp DB so we don't pollute the real one
    test_db = config.LOGS_DIR / "test_persistence.sqlite"
    if test_db.exists():
        os.remove(test_db)
    test_csv = config.LOGS_DIR / "test_run_timeseries.csv"
    if test_csv.exists():
        os.remove(test_csv)

    p = Persistence(run_id="test_run", db_path=test_db)

    sim_time = {"month": 7, "day": 1, "hour": 12, "minute": 0}
    payload = {
        "zones": {
            "Core_ZN": {"temp": 24.5, "occupancy": 8, "cooling_rate": 1200},
            "Perimeter_ZN_1": {"temp": 25.1, "occupancy": 3, "cooling_rate": 800},
        },
        "outdoor_temp": 29.2,
    }
    p.log_snapshot(sim_time, payload)
    p.log_action(sim_time, "cooling_setpoint", "Core_ZN",
                 value_before=24.0, value_after=25.0,
                 reasoning="Reducing cooling load during low occupancy")
    p.log_llm_call(sim_time, "reason", prompt={"role": "test"},
                   response="test response", tool_calls=[],
                   latency_ms=345.6, tokens_in=120, tokens_out=45)

    summary = p.summary()
    print(f"[OK] Persistence sanity check:")
    print(f"  DB path:  {p.db_path}")
    print(f"  CSV path: {p.csv_path}")
    print(f"  Counts:   {summary}")

    # Verify CSV was written
    with open(p.csv_path) as f:
        lines = f.readlines()
    print(f"  CSV rows: {len(lines)} (should be 2: header + 1 data row)")

    # Clean up
    os.remove(test_db)
    os.remove(test_csv)
    print(f"[OK] Cleaned up test files.")