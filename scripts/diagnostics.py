"""Quick post-run diagnostics on the AI vs baseline results."""

import sys
import sqlite3
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import config


def action_stats():
    print("=" * 60)
    print("AI ACTION STATS")
    print("=" * 60)
    conn = sqlite3.connect(config.EVENTS_DB)
    cur = conn.cursor()

    cur.execute(
        "SELECT sim_hour, COUNT(*) FROM actions "
        "WHERE run_id='ai' GROUP BY sim_hour ORDER BY sim_hour"
    )
    rows = cur.fetchall()
    print("\nActions by hour of day:")
    if not rows:
        print("  (none)")
    for h, n in rows:
        print(f"  hour {h:02d}: {n} actions")

    cur.execute(
        "SELECT kind, COUNT(*) FROM actions WHERE run_id='ai' GROUP BY kind"
    )
    print("\nActions by kind:")
    for k, n in cur.fetchall():
        print(f"  {k}: {n}")

    cur.execute(
        "SELECT sim_day, sim_hour, kind, value_after, "
        "substr(reasoning, 1, 100) FROM actions "
        "WHERE run_id='ai' ORDER BY id LIMIT 5"
    )
    print("\nFirst 5 actions:")
    for r in cur.fetchall():
        print(f"  Day {r[0]:02d} {r[1]:02d}h: {r[2]} -> {r[3]}°C")
        print(f"    reason: {r[4]}")

    cur.execute(
        "SELECT sim_day, sim_hour, kind, value_after, "
        "substr(reasoning, 1, 100) FROM actions "
        "WHERE run_id='ai' ORDER BY id DESC LIMIT 5"
    )
    print("\nLast 5 actions:")
    for r in cur.fetchall():
        print(f"  Day {r[0]:02d} {r[1]:02d}h: {r[2]} -> {r[3]}°C")
        print(f"    reason: {r[4]}")

    conn.close()


def energy_comparison():
    print()
    print("=" * 60)
    print("ENERGY + COMFORT COMPARISON")
    print("=" * 60)

    b_csv = config.LOGS_DIR / "baseline_timeseries.csv"
    a_csv = config.LOGS_DIR / "ai_timeseries.csv"

    b = pd.read_csv(b_csv)
    a = pd.read_csv(a_csv)

    # Cooling energy (sum of watts across all timesteps)
    b_cool = b["total_cooling_w"].sum()
    a_cool = a["total_cooling_w"].sum()
    delta_cool = (a_cool - b_cool) / b_cool * 100 if b_cool else 0

    print(f"\nCooling energy (W-timesteps summed):")
    print(f"  Baseline: {b_cool:>15,.0f}")
    print(f"  AI:       {a_cool:>15,.0f}")
    print(f"  Delta:    {delta_cool:+.2f}%")

    b_heat = b["total_heating_w"].sum()
    a_heat = a["total_heating_w"].sum()
    print(f"\nHeating energy (W-timesteps summed):")
    print(f"  Baseline: {b_heat:>15,.0f}")
    print(f"  AI:       {a_heat:>15,.0f}")

    # Comfort: average indoor temp during occupied hours (7-18)
    zone_cols = ["core_temp", "perimeter_1_temp", "perimeter_2_temp",
                 "perimeter_3_temp", "perimeter_4_temp"]

    def occupied_stats(df, label):
        occ = df[(df["sim_hour"] >= 7) & (df["sim_hour"] <= 18)]
        avg = occ[zone_cols].values.mean()
        below = (occ[zone_cols] < config.COMFORT_TEMP_MIN_C).sum().sum()
        above = (occ[zone_cols] > config.COMFORT_TEMP_MAX_C).sum().sum()
        total = occ[zone_cols].size
        pct_in_band = (total - below - above) / total * 100 if total else 0
        print(f"\n{label} comfort (occupied hours 7-18):")
        print(f"  Avg indoor temp:       {avg:.2f}°C")
        print(f"  % time in comfort band: {pct_in_band:.1f}%")
        print(f"  Zone-timesteps below {config.COMFORT_TEMP_MIN_C}°C: {below}")
        print(f"  Zone-timesteps above {config.COMFORT_TEMP_MAX_C}°C: {above}")

    occupied_stats(b, "Baseline")
    occupied_stats(a, "AI")


if __name__ == "__main__":
    action_stats()
    energy_comparison()