# Eco-Loop Building Agents
<div align="center">

# Eco-Loop Building Agents

### 🎥 Project Demo

<a href="https://drive.google.com/file/d/1qFXqSriZ1IW5l15QUA5bSpt4l-oDAZln/view?usp=sharing">
  <img src="assets/demo-thumbnail.png" alt="Watch Demo" width="900">
</a>

<p><b>▶ Click the image above to watch the full project demo.</b></p>

</div>
Autonomous LLM-driven HVAC control for smart buildings. A closed-loop agent reads live simulation state, reasons about occupancy, comfort, and outdoor conditions, and injects setpoints back into an EnergyPlus simulation via Model Context Protocol tools — all without a human in the loop.

Built for the **Honeywell Technologies Campus Connect Hackathon 2026**, Problem Statement 1: *AI-Powered Autonomous Smart Building Optimization*.

## Headline results

Simulation: DOE Reference Small Office building, one week of July using Bangalore ISHRAE weather data, 15-minute EnergyPlus timestep, LLM decision every simulated hour (216 autonomous decisions total).

| Metric | Baseline (fixed 24°C) | AI-controlled | Delta |
|---|---|---|---|
| Cooling energy (7-day sim) | 189.5 kWh | 178.8 kWh | **−5.7%** |
| Time in comfort band (21–25°C, occupied hours) | 79.8% | **85.2%** | **+5.4 pts** |
| Zone-timesteps too hot | 88 | 59 | **−33%** |
| Agent errors over 216 decisions | — | 0 | — |

**The AI beat the fixed-schedule baseline on both energy and comfort simultaneously — with a 33% reduction in comfort violations.**

## What this is

A modern Building Management System runs on rule-based schedules that don't adapt to real-time occupancy, weather, or price signals. This project replaces the fixed schedule with a small open-source LLM (Llama 3.1 8B via Groq) that acts as the building's brain.

Every simulated hour:

1. **Perceive** — the system reads live zone temperatures, occupancy, HVAC power draw, and outdoor conditions from a running EnergyPlus simulation.
2. **Reason** — the LLM receives a compact, structured state summary and reasons about what to do, guided by a rule-based system prompt.
3. **Act** — the LLM emits a tool call to change setpoints; a safety wrapper clamps out-of-range values before they reach EnergyPlus.
4. **Verify** — post-action state is checked; if the action was clamped or errored, the graph loops back to Correct for a revised decision.
5. **Log** — every snapshot, LLM call (with latency and tokens), and action (with LLM reasoning) is persisted to SQLite + CSV for the dashboard.

The LLM never touches EnergyPlus directly. All interactions flow through a tool layer that mirrors the MCP protocol pattern — the same LLM brain could be swapped from EnergyPlus in dev to real Honeywell BMS APIs in production by re-implementing only the tool bindings.

## Architecture

```
   ┌──────────────────────────────────────────┐    ┌─────────────┐
   │        LangGraph agent                   │◄──►│  Groq LLM   │
   │  Perceive → Reason → Act → Verify → Correct   │ (Llama 3.1) │
   └──────────────────────────────────────────┘    └─────────────┘
                          ▲
                          │  tool calls / results
                          ▼
   ┌──────────────────────────────────────────┐
   │        MCP-style tool server              │
   │  Sensor tools · Actuator tools · Files    │
   └──────────────────────────────────────────┘
                          ▲                        ┌─────────────┐
                          │  read / write          │ SQLite +    │
                          ▼                        │ CSV logs    │
   ┌──────────────────────────────────────────┐    │             │
   │        Bridge layer                       │───►│             │
   │  Safety wrapper · Log summarizer · Handles│    └──────┬──────┘
   └──────────────────────────────────────────┘           │
                          ▲                                ▼
                          │                        ┌─────────────┐
                          ▼                        │ Streamlit   │
   ┌──────────────────────────────────────────┐    │ dashboard   │
   │        EnergyPlus simulation              │    └─────────────┘
   │  Small office IDF + Bangalore EPW         │
   └──────────────────────────────────────────┘
```

Full design details in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Tech stack

- **Simulation**: EnergyPlus 26.1.0 via the `pyenergyplus` Python API
- **Building model**: DOE Reference Small Office (5 conditioned zones + attic)
- **Weather**: Bangalore ISHRAE `.epw` from EnergyPlus weather archive
- **Agent orchestration**: LangGraph state machine with typed state
- **LLM**: Llama 3.1 8B via Groq (free tier)
- **Tool layer**: LangChain `@tool` decorators, MCP-inspired design
- **IDF manipulation**: `eppy` for schedule injection and output-variable declaration
- **Persistence**: SQLite (structured events) + CSV (time-series mirror)
- **Dashboard**: Streamlit + Plotly

**Zero paid services.** The entire stack runs on a free Groq API key and open-source components.

## Setup

Prerequisites: Windows or Linux, EnergyPlus 26.1.0 installed, Python 3.10+, a free Groq API key from https://console.groq.com.

```powershell
# Clone
git clone https://github.com/pranavgudagi/eco-loop-building-agents.git
cd eco-loop-building-agents

# Virtual env
python -m venv venv
venv\Scripts\activate       # PowerShell
# source venv/bin/activate  # bash

# Dependencies
pip install -r requirements.txt

# Configure
copy .env.example .env
# Edit .env and paste your Groq API key

# Point config.py at your local EnergyPlus install
# (edit the ENERGYPLUS_DIR constant if not D:\EnergyPlus\Energy_Plus)

# Copy in the building and weather files
copy "<EnergyPlus>\ExampleFiles\RefBldgSmallOfficeNew2004_Chicago.idf" idf\baseline_small_office.idf
# Download Bangalore ISHRAE .epw from https://energyplus.net/weather
# and save it as idf\bangalore.epw

# Prepare the IDF (enables weather-file run, injects occupancy schedule,
# declares required Output:Variables)
python scripts\prepare_idf.py

# Sanity check
python config.py
```

Should print `[OK] Config valid.` when everything is wired up.

## Running

```powershell
# Run baseline + AI simulations back-to-back
python main.py

# Or run one at a time
python main.py --baseline-only
python main.py --ai-only

# Print energy + comfort comparison
python scripts\diagnostics.py

# Launch the dashboard (opens at http://localhost:8501)
streamlit run dashboard\app.py
```

Wall-time: baseline ~3 seconds, AI-controlled ~13-15 minutes (network-bound by Groq API calls, one per simulated hour × 168 hours + sizing periods).

## Repo structure

```
eco-loop-building-agents/
├── main.py                       # Orchestrates baseline + AI runs
├── config.py                     # Paths, model ID, safety bounds, timing
├── requirements.txt
├── .env.example                  # Template — real .env is gitignored
│
├── simulation/
│   └── ep_driver.py              # pyenergyplus wrapper, callback registration
│
├── mcp_server/
│   ├── __init__.py               # Runtime state broker for tools
│   └── tools/
│       ├── sensors.py            # Sensor tool functions
│       ├── actuators.py          # Actuator writes with safety clamping
│       ├── files.py              # IDF read/modify tools
│       └── errors.py             # EnergyPlus error log tools
│
├── agent/
│   ├── graph.py                  # LangGraph state machine
│   ├── nodes.py                  # Perceive / Reason / Act / Verify / Correct
│   ├── prompts.py                # System prompt + templates
│   └── llm_client.py             # Groq wrapper with retry-backoff
│
├── data/
│   └── persistence.py            # SQLite + CSV writers
│
├── dashboard/
│   └── app.py                    # Streamlit comparison dashboard
│
├── idf/
│   ├── baseline_small_office.idf # Modified reference building
│   └── bangalore.epw             # Weather data
│
├── logs/                         # Runtime output (gitignored)
├── scripts/
│   ├── prepare_idf.py            # Injects occupancy + output variables
│   ├── smoke_test.py             # Toolchain verification
│   └── diagnostics.py            # Post-run summary stats
└── docs/
    └── ARCHITECTURE.md           # Deep-dive design document
```

## Key design decisions

**Prompt engineering.** The system prompt encodes a rule-based decision procedure (Rules 1-6) rather than asking the LLM to reason about HVAC physics from scratch. Small models pattern-match to numbered rules more reliably than they reason about thermodynamics. The prompt also names the baseline behavior explicitly, so the LLM understands what "better" looks like.

**Safety-clamped actuators.** Every LLM setpoint request passes through a clamp bounded to a configurable safe range (20–28°C). Clamp events are logged. This means the LLM cannot destabilize the building even under adversarial or hallucinated outputs.

**Compact prompt formatting.** Raw EnergyPlus `.eso` and `.err` files are never sent to the LLM. Instead a deterministic Python summarizer distills recent state into a ~250-token structured block. This keeps context small, Groq latency low, and prompts within Llama's effective context window.

**Retry-with-fallback.** The Groq client wraps every call in exponential-backoff retry (3 attempts) and falls back to holding last-known-good setpoints on total failure. In the reference run, zero agent errors occurred across 216 LLM calls.

**Two-run comparison.** Baseline and AI runs use identical IDF, weather, occupancy, and simulation timesteps — the only variable is whether the agent is registered on the callback. This isolates the AI's contribution.

## Deliverables mapping

| Deliverable | Location |
|---|---|
| 1. Unified source code | This repository |
| 2. Building models (baseline + modified IDF) | `idf/baseline_small_office.idf` |
| 3. Quantitative savings dashboard | `dashboard/app.py` (Streamlit) |
| 4. System architecture document | `docs/ARCHITECTURE.md` |
| 5. PoC demonstration video | Linked in the presentation |

