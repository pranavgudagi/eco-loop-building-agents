"""
Streamlit dashboard: baseline vs AI comparison.

Reads from the CSV time-series files and SQLite events DB. Shows the
headline savings, comfort metrics, action timeline, and LLM stats.

Run with:  streamlit run dashboard/app.py
"""

import sys
import sqlite3
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import config


st.set_page_config(
    page_title="Eco-Loop Building Agents",
    page_icon="🏢",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Data loading (cached for snappy UI)
# ---------------------------------------------------------------------------
@st.cache_data
def load_timeseries():
    b_path = config.LOGS_DIR / "baseline_timeseries.csv"
    a_path = config.LOGS_DIR / "ai_timeseries.csv"
    if not b_path.exists() or not a_path.exists():
        return None, None
    b = pd.read_csv(b_path)
    a = pd.read_csv(a_path)
    for df in (b, a):
        df["step"] = range(len(df))
        df["hour_of_run"] = df["step"]
        df["avg_indoor"] = df[[
            "core_temp", "perimeter_1_temp", "perimeter_2_temp",
            "perimeter_3_temp", "perimeter_4_temp"
        ]].mean(axis=1)
    return b, a


@st.cache_data
def load_actions():
    if not config.EVENTS_DB.exists():
        return pd.DataFrame()
    conn = sqlite3.connect(config.EVENTS_DB)
    df = pd.read_sql_query(
        "SELECT sim_day, sim_hour, kind, value_before, value_after, reasoning "
        "FROM actions WHERE run_id='ai' ORDER BY id",
        conn,
    )
    conn.close()
    return df


@st.cache_data
def load_llm_stats():
    if not config.EVENTS_DB.exists():
        return pd.DataFrame()
    conn = sqlite3.connect(config.EVENTS_DB)
    df = pd.read_sql_query(
        "SELECT node_name, latency_ms, tokens_in, tokens_out, error "
        "FROM llm_calls WHERE run_id='ai'",
        conn,
    )
    conn.close()
    return df


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------
def energy_totals(df):
    return {
        "cooling_wh": df["total_cooling_w"].sum() * (config.EP_TIMESTEP_MINUTES / 60),
        "heating_wh": df["total_heating_w"].sum() * (config.EP_TIMESTEP_MINUTES / 60),
    }


def comfort_stats(df):
    zone_cols = ["core_temp", "perimeter_1_temp", "perimeter_2_temp",
                 "perimeter_3_temp", "perimeter_4_temp"]
    occ = df[(df["sim_hour"] >= 7) & (df["sim_hour"] <= 18)]
    if occ.empty:
        return {"pct_in_band": 0, "violations_hot": 0, "violations_cold": 0, "avg_temp": 0}
    vals = occ[zone_cols].values
    below = (vals < config.COMFORT_TEMP_MIN_C).sum()
    above = (vals > config.COMFORT_TEMP_MAX_C).sum()
    total = vals.size
    return {
        "pct_in_band": (total - below - above) / total * 100,
        "violations_hot": int(above),
        "violations_cold": int(below),
        "avg_temp": float(vals.mean()),
    }


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("🏢 Eco-Loop Building Agents")
st.caption("Autonomous LLM-driven smart building control • EnergyPlus + MCP + LangGraph + Groq")

baseline, ai = load_timeseries()
if baseline is None or ai is None:
    st.error("No run data found. Execute `python main.py` first.")
    st.stop()

# ---------------------------------------------------------------------------
# Headline metrics
# ---------------------------------------------------------------------------
b_energy = energy_totals(baseline)
a_energy = energy_totals(ai)
b_comfort = comfort_stats(baseline)
a_comfort = comfort_stats(ai)

energy_delta_pct = (
    (a_energy["cooling_wh"] - b_energy["cooling_wh"])
    / b_energy["cooling_wh"] * 100
    if b_energy["cooling_wh"] > 0 else 0
)
comfort_delta_pts = a_comfort["pct_in_band"] - b_comfort["pct_in_band"]

st.header("Headline results")
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric(
        "Cooling energy (AI)",
        f"{a_energy['cooling_wh'] / 1000:.1f} kWh",
        delta=f"{energy_delta_pct:+.1f}% vs baseline",
        delta_color="inverse",  # negative = green (less energy is good)
    )
with c2:
    st.metric(
        "Baseline cooling",
        f"{b_energy['cooling_wh'] / 1000:.1f} kWh",
    )
with c3:
    st.metric(
        "Comfort (AI)",
        f"{a_comfort['pct_in_band']:.1f}%",
        delta=f"{comfort_delta_pts:+.1f} pts vs baseline",
    )
with c4:
    st.metric(
        "Baseline comfort",
        f"{b_comfort['pct_in_band']:.1f}%",
    )

# ---------------------------------------------------------------------------
# Time-series comparison
# ---------------------------------------------------------------------------
st.header("Time-series comparison")

fig = make_subplots(
    rows=3, cols=1, shared_xaxes=True,
    subplot_titles=(
        "Cooling power draw (W)",
        "Average indoor temperature (°C)",
        "Outdoor temperature (°C)",
    ),
    vertical_spacing=0.08,
)

fig.add_trace(go.Scatter(
    x=baseline["step"], y=baseline["total_cooling_w"],
    name="Baseline cooling", line=dict(color="#888", width=1),
), row=1, col=1)
fig.add_trace(go.Scatter(
    x=ai["step"], y=ai["total_cooling_w"],
    name="AI cooling", line=dict(color="#2ca02c", width=1.5),
), row=1, col=1)

fig.add_trace(go.Scatter(
    x=baseline["step"], y=baseline["avg_indoor"],
    name="Baseline indoor", line=dict(color="#888", width=1),
    showlegend=False,
), row=2, col=1)
fig.add_trace(go.Scatter(
    x=ai["step"], y=ai["avg_indoor"],
    name="AI indoor", line=dict(color="#2ca02c", width=1.5),
    showlegend=False,
), row=2, col=1)

# Comfort band overlay
fig.add_hrect(
    y0=config.COMFORT_TEMP_MIN_C, y1=config.COMFORT_TEMP_MAX_C,
    fillcolor="green", opacity=0.06, layer="below", line_width=0,
    row=2, col=1,
)

fig.add_trace(go.Scatter(
    x=ai["step"], y=ai["outdoor_temp"],
    name="Outdoor", line=dict(color="#ff7f0e", width=1),
), row=3, col=1)

fig.update_layout(height=680, hovermode="x unified", margin=dict(t=40, b=20))
fig.update_xaxes(title_text="Simulation step (1 = 15 min)", row=3, col=1)
st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Comfort breakdown
# ---------------------------------------------------------------------------
st.header("Comfort breakdown (occupied hours only)")
c1, c2 = st.columns(2)
with c1:
    st.subheader("Baseline")
    st.write(f"• Avg indoor temp: **{b_comfort['avg_temp']:.2f}°C**")
    st.write(f"• Time in {config.COMFORT_TEMP_MIN_C}-{config.COMFORT_TEMP_MAX_C}°C band: **{b_comfort['pct_in_band']:.1f}%**")
    st.write(f"• Zone-timesteps too hot: **{b_comfort['violations_hot']}**")
    st.write(f"• Zone-timesteps too cold: **{b_comfort['violations_cold']}**")
with c2:
    st.subheader("AI-controlled")
    st.write(f"• Avg indoor temp: **{a_comfort['avg_temp']:.2f}°C**")
    st.write(f"• Time in {config.COMFORT_TEMP_MIN_C}-{config.COMFORT_TEMP_MAX_C}°C band: **{a_comfort['pct_in_band']:.1f}%**")
    st.write(f"• Zone-timesteps too hot: **{a_comfort['violations_hot']}**")
    st.write(f"• Zone-timesteps too cold: **{a_comfort['violations_cold']}**")

# ---------------------------------------------------------------------------
# Agent action timeline
# ---------------------------------------------------------------------------
st.header("Agent decisions")
actions = load_actions()
if actions.empty:
    st.info("No actions logged. Did you run `python main.py`?")
else:
    st.caption(f"{len(actions)} setpoint changes over the simulation.")
    display = actions.copy()
    display["When"] = display.apply(
        lambda r: f"Day {int(r['sim_day']):02d}  {int(r['sim_hour']):02d}:00", axis=1
    )
    display = display[["When", "kind", "value_after", "reasoning"]].rename(
        columns={"kind": "Action", "value_after": "Setpoint (°C)", "reasoning": "LLM reasoning"}
    )
    st.dataframe(display, use_container_width=True, height=400)

# ---------------------------------------------------------------------------
# LLM statistics
# ---------------------------------------------------------------------------
st.header("LLM performance")
llm = load_llm_stats()
if not llm.empty:
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total LLM calls", f"{len(llm)}")
    with c2:
        st.metric("Median latency", f"{llm['latency_ms'].median():.0f} ms")
    with c3:
        st.metric("Total tokens in", f"{int(llm['tokens_in'].sum()):,}")
    with c4:
        n_errors = llm["error"].notna().sum()
        st.metric("Errors", f"{n_errors}", delta=None if n_errors == 0 else f"⚠️ {n_errors}")

    st.subheader("Latency distribution")
    fig2 = go.Figure()
    fig2.add_trace(go.Histogram(x=llm["latency_ms"], nbinsx=40, marker_color="#2ca02c"))
    fig2.update_layout(
        xaxis_title="Latency (ms)",
        yaxis_title="Count",
        height=280,
        margin=dict(t=20, b=40),
    )
    st.plotly_chart(fig2, use_container_width=True)
else:
    st.info("No LLM stats yet.")

st.caption("Built with EnergyPlus • MCP • LangGraph • Groq (Llama 3.1) • Streamlit")