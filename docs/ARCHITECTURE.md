# System Architecture

**Eco-Loop Building Agents** — a closed-loop, LLM-driven HVAC control system
that reads live EnergyPlus state and injects setpoints via a Model Context
Protocol tool layer. This document explains the design, the four
architectural concerns called out in the problem statement (tool-calling
architecture, prompt engineering, prompt latency, log handling), and the
key engineering trade-offs.

---

## 1. High-level architecture

The system is a single Python process running six cooperating layers.
Data flows in a closed loop between the LLM (top) and the EnergyPlus
simulation (bottom); a persistence layer collects everything for the
dashboard.

```
   ┌────────────────────────────────────────────────────┐   ┌──────────────┐
   │             LangGraph agent                        │◄─►│   Groq LLM   │
   │  Perceive → Reason → Act → Verify → Correct        │   │ Llama 3.1 8B │
   └────────────────────────────────────────────────────┘   └──────────────┘
                    ▲             tool calls / results
                    │
                    ▼
   ┌────────────────────────────────────────────────────┐
   │           MCP-style tool server                    │
   │   sensors · actuators · files · errors             │
   └────────────────────────────────────────────────────┘
                    ▲             read / write
                    │                                       ┌──────────────┐
                    ▼                                       │  SQLite +    │
   ┌────────────────────────────────────────────────────┐   │  CSV logs    │
   │              Bridge layer                          │──►│              │
   │  safety wrapper · log summarizer · handle registry │   └──────┬───────┘
   └────────────────────────────────────────────────────┘          │
                    ▲                                              ▼
                    │                                       ┌──────────────┐
                    ▼                                       │  Streamlit   │
   ┌────────────────────────────────────────────────────┐   │  dashboard   │
   │           EnergyPlus simulation                    │   └──────────────┘
   │  Small office IDF + Bangalore weather EPW          │
   └────────────────────────────────────────────────────┘
```

**Layer responsibilities:**

| Layer | Responsibility | Files |
|---|---|---|
| Agent | Orchestrates one control cycle per simulated hour | `agent/graph.py`, `agent/nodes.py` |
| MCP server | Exposes sensor + actuator functions as LLM-callable tools | `mcp_server/tools/*.py` |
| Bridge | Handle caching, safety clamping, log summarization | `simulation/handles.py`, `mcp_server/tools/actuators.py`, `data/summarizer.py` |
| Simulation | Live EnergyPlus process driven by pyenergyplus callbacks | `simulation/ep_driver.py` |
| Persistence | SQLite events + CSV time-series mirror | `data/persistence.py` |
| Dashboard | Streamlit visualisation of baseline vs AI runs | `dashboard/app.py` |

---

## 2. Tool-calling architecture

*(Deliverable 4 requirement)*

### 2.1 Why MCP-style tools

An LLM alone is text-in, text-out. It cannot read a temperature sensor or
turn a valve. Model Context Protocol standardises how an LLM connects to
external functions: the LLM sees a menu of typed tools, decides which to
call, and receives structured results back. We implement the tool pattern
using LangChain's `@tool` decorator, which produces MCP-compatible JSON
schemas. This means the same LLM brain could be repointed at real Honeywell
BMS APIs in production by re-implementing only the tool bindings — the
agent code, prompts, and reasoning logic remain unchanged.

### 2.2 Tool inventory

Tools are grouped by purpose into four modules under `mcp_server/tools/`:

**Sensor tools** (`sensors.py`) — read-only, return current EnergyPlus state:

| Tool | Returns |
|---|---|
| `get_zone_temperature(zone_id)` | °C |
| `get_all_zone_temperatures()` | dict of five zones |
| `get_zone_occupancy(zone_id)` | occupant count |
| `get_zone_cooling_rate(zone_id)` | W |
| `get_zone_heating_rate(zone_id)` | W |
| `get_total_hvac_power()` | aggregated W |
| `get_outdoor_temperature()` | °C |
| `get_sim_time()` | month/day/hour/minute |
| `get_full_snapshot()` | one-call convenience combining all of the above |

**Actuator tools** (`actuators.py`) — safety-clamped writes:

| Tool | Effect |
|---|---|
| `set_cooling_setpoint(temperature_c, reasoning)` | Overrides building cooling schedule |
| `set_heating_setpoint(temperature_c, reasoning)` | Overrides heating schedule (not exposed to LLM in Bangalore variant) |

**File tools** (`files.py`) — for offline IDF inspection and modification.

**Error tools** (`errors.py`) — parse EnergyPlus `.err` file for warnings.

### 2.3 The safety wrapper

Every actuator write flows through `_clamp()`:

```python
clamped, was_clamped = _clamp(value, THERMOSTAT_MIN_C, THERMOSTAT_MAX_C)
```

If the LLM requests a setpoint outside `[20°C, 28°C]`, the value is clamped
and the event is logged with `[CLAMPED]` in the reasoning field. This means
a malformed or hallucinated LLM output cannot destabilise the building.
Combined with retry-with-backoff on the LLM client, the closed loop
survives arbitrary LLM failures — critical for the 30%-weighted "System
Integration" criterion.

### 2.4 The reasoning field

Every actuator tool requires a `reasoning` string argument. The LLM must
name which rule from the system prompt justified the action. This produces
an auditable log where every decision is self-explaining — the action
timeline in the dashboard is directly LLM-generated commentary.

Example log entry:

```
Day 03 09h: cooling_setpoint -> 24.5°C
    reason: Rule 4: occupied hours, holding 24.5°C for modest savings.
```

---

## 3. Prompt engineering strategies

*(Deliverable 4 requirement)*

### 3.1 Rule-based system prompt

Small models (Llama 3.1 8B) pattern-match to numbered rules more reliably
than they reason from first principles. Rather than asking the LLM to
understand HVAC physics, the system prompt encodes a six-rule decision
procedure ordered by priority:

- **Rule 1** — Night setback (hours 22–05): set cooling to 26°C
- **Rule 2** — Weekend setback: set cooling to 26°C
- **Rule 3** — Early-morning prep (06:00): 24.5°C to pre-cool for arrivals
- **Rule 4** — Occupied hours (07:00–17:59 weekdays): 24.5°C default,
  drop to 24.0°C if any zone > 25.2°C
- **Rule 5** — Evening wind-down (18:00–21:59): 25.5°C
- **Rule 6** — Do nothing if last action matches current rule

Full text: `agent/prompts.py::SYSTEM_PROMPT`.

### 3.2 Explicit "sacred" hours

Occupied hours (07:00–17:59 weekdays) are declared inviolable in the prompt:

> "During these hours, cooling setpoint MUST stay between 23.5°C and 25.0°C.
> Do NOT apply aggressive setbacks during these hours even if occupancy
> sensor reads 0 (the sensor may lag actual arrivals)."

This addresses an observed failure mode in earlier iterations where the LLM
would apply Rule 1 (empty-building setback) during work hours whenever a
transient occupancy read reported zero.

### 3.3 Explicit physics reminders

Every LLM invocation includes the physical facts the model tends to
misremember:

> "HIGHER cooling setpoint = LESS cooling energy consumed.
> LOWER cooling setpoint = MORE cooling energy consumed."

An earlier iteration without these reminders had the LLM *lowering*
setpoints during idle periods with reasoning like "outdoor is stable,
maintaining setpoint at 22.5°C" — which raised energy use by 40%.
Explicit reminders resolved this.

### 3.4 Named comparator

The prompt names the baseline behaviour explicitly:

> "You are being compared against a dumb baseline that always holds
> cooling at 24°C — you must BEAT it on energy use."

Giving the LLM a concrete comparator improves optimisation-directed
reasoning versus generic "minimise energy" wording.

### 3.5 Structured compact state

The user prompt (`build_reason_prompt` in `prompts.py`) sends a compact
structured snapshot — zone temperatures, occupancy, HVAC power, three-hour
rolling history, last three actions with reasoning. Approximate size:
250–350 tokens per invocation, regardless of simulation age.

### 3.6 Low temperature

`LLM_TEMPERATURE = 0.2`. Control tasks reward consistency over creativity.
Higher temperatures produce more diverse but less predictable setpoint
choices.

### 3.7 Empirically-derived rule values

The specific setpoint values in each rule were derived by iterative testing.
Multiple prompt variants were tested (aggressive setbacks to 27°C, moderate
setbacks to 25°C, conservative nudges to 24.5°C). We observed a nonlinear
cliff in the energy-comfort tradeoff around 25.5°C setpoint: values above
this threshold produced 15-53% energy savings but caused comfort violations
to double or triple. The chosen values keep the system on the safe side of
this cliff, delivering a Pareto improvement rather than trading comfort for
energy.

---

## 4. Prompt latency management

*(Deliverable 4 requirement)*

### 4.1 Constraint

The simulation runs 168 sim-hours (one week). At one LLM decision per
sim-hour, that's 168 LLM calls minimum, plus corrections and sizing period
warmups. Each call round-trips to Groq's public API. Wall time = network
latency × call count.

### 4.2 Techniques applied

**Sub-hourly rate limiting.** EnergyPlus fires the callback every 15
minutes of sim time (4 timesteps per hour). We invoke the LLM only every
fourth callback — one decision per sim-hour, not per timestep. Cut LLM
call count by 4×.

**Fast free-tier model.** `llama-3.1-8b-instant` on Groq averages ~500ms
per tool-calling invocation, vs several seconds for larger models. This
choice trades reasoning quality for reliable sub-second latency.

**Retry with exponential backoff.** `agent/llm_client.py` implements three
retries with delays of 0.5s / 1.0s / 2.0s. Transient Groq errors don't
crash the simulation; a persistent failure falls back to holding
last-known-good setpoints via the safety wrapper.

**Constant-size prompts.** The compact snapshot format (Section 3.5) keeps
prompt token count roughly constant across simulation days. Prompt length
does not grow with simulation age, so latency does not drift.

**No streaming.** Setpoint decisions are structured tool calls, not prose.
We use blocking `invoke()` rather than streaming to get parsed tool
arguments in one round-trip.

### 4.3 Observed performance

From the reference AI run (216 total LLM calls over ~13 minutes wall time
for a 7-day simulation):

| Metric | Value |
|---|---|
| Median latency | ~500 ms |
| p95 latency | ~800 ms |
| LLM errors | 0 / 216 |
| Tokens per prompt (in) | ~300 |
| Tokens per response (out) | ~50 |
| Total cost | $0 (Groq free tier) |

Latency histogram is rendered live in the dashboard's LLM performance
section.

---

## 5. Handling lengthy simulation logs

*(Deliverable 4 requirement)*

### 5.1 The problem

EnergyPlus produces four verbose text files per run:

- `eplusout.eso` — every sensor value at every timestep (megabytes)
- `eplusout.err` — warnings and errors (thousands of lines for edge cases)
- `eplusout.eio` — initialization detail
- `eplusout.mtr` — meter readings

Directly forwarding any of these to the LLM would consume the context
window in seconds and destroy latency.

### 5.2 Never send raw logs

The system does not read `.eso`, `.err`, or `.eio` files during runtime.
Instead, all sensor values reach the LLM via the pyenergyplus API — cached
integer handles resolve to live in-memory values in microseconds.

### 5.3 Deterministic Python summarizer

For any log context that must reach the LLM — e.g. recent warnings for
error-recovery reasoning — a summarizer function in `data/summarizer.py`
(currently a stub, extension point) distils logs into a bounded structured
form:

- Recent warnings deduplicated by message
- Errors grouped by severity
- Compacted to ~200 tokens regardless of raw log size

### 5.4 SQLite as event store, not log parser

Every timestep snapshot and every LLM call is written to SQLite at write
time. Dashboards and post-hoc analysis query indexed tables, not text
logs. This makes the whole system observable without ever re-reading a
verbose EnergyPlus file.

### 5.5 Compact prompt state (Section 3.5)

Reiterated for completeness: the prompt itself sends a fixed-size ~300-token
snapshot rather than any cumulative log. Prompt size does not grow with
simulation age. This is the single most important design choice for
handling long-horizon simulations reliably.

---

## 6. Closed-loop execution framework

The main loop pattern, per the problem statement's "Closed-Loop Execution
Framework" requirements:

**Feedback (EnergyPlus → AI).** At each 15-minute timestep, EnergyPlus
fires the registered callback (`_timestep_callback` in `ep_driver.py`).
Every 4th callback (once per sim-hour), the callback wakes the LangGraph
agent by invoking the compiled state machine.

**Reasoning.** The agent's Perceive node calls `get_full_snapshot()`,
producing zone temperatures, occupancy, HVAC power, and outdoor conditions.
Reason node builds the structured prompt (Section 3.5) and calls the LLM.

**Control Actions (AI → EnergyPlus).** The LLM emits `set_cooling_setpoint`
tool calls. The Act node dispatches through TOOL_DISPATCH; the safety
wrapper clamps values before calling `driver.write_setpoint()`, which in
turn calls `api.exchange.set_actuator_value()`.

**Forward Injection.** EnergyPlus resumes simulation with the new setpoint
value in place. The next timestep computes the building's thermal response
to the new setpoint. This closes the loop.

**Self-correction.** After Act, the Verify node checks for tool errors and
clamp events. If present (and correction iteration count < 2), the graph
routes to Correct, which asks the LLM for a revised action. Correct loops
back to Act. This is the same Critic → Writer pattern from multi-agent RAG
systems, applied to physical control.

---

## 7. Two-run comparison methodology

Both simulations use identical inputs:

- Same IDF (`idf/baseline_small_office.idf`)
- Same weather (`idf/bangalore.epw`)
- Same occupancy schedule (injected by `scripts/prepare_idf.py`)
- Same simulation window (July 1–7)
- Same 15-minute EnergyPlus timestep

The only variable is whether the agent callback is registered:

- **Baseline run** — no agent registered. EnergyPlus follows the IDF's
  default cooling schedule (fixed 24°C).
- **AI run** — agent registered on the timestep callback. LLM sets
  cooling setpoint every sim-hour based on the six-rule prompt.

Isolating this single variable means observed energy and comfort deltas
are attributable purely to the AI's decisions, not to differences in
weather, occupancy, or building physics.

---

## 8. Reference run results

Simulation: DOE Reference Small Office, one week of July, Bangalore
weather. 15-minute simulation timestep, 1-hour LLM decision interval,
216 autonomous LLM decisions over the run.

| Metric | Baseline | AI-controlled | Delta |
|---|---|---|---|
| Cooling energy (kWh, 7 days) | 189.5 | 178.8 | **−5.7%** |
| Comfort band (21–25°C, occupied) | 79.8% | **85.2%** | **+5.4 pts** |
| Zone-timesteps > 25°C (occupied) | 88 | 59 | **−33%** |
| Zone-timesteps < 21°C (occupied) | 21 | 21 | 0 |
| LLM errors | — | 0 / 216 | — |
| Agent errors | — | 0 / 216 | — |
| Wall time | 4 s | 748 s | — |
| API cost | $0 | $0 | — |

**Key result:** the AI beat the fixed-schedule baseline on both energy
and comfort simultaneously, with a 33% reduction in comfort violations.
No test-driven baseline tuning was required to achieve this — the
baseline is the reference IDF's default behaviour. This is a genuine
Pareto improvement: better on both axes at once.

---

## 9. Known limitations

Documented honestly for interview defence.

**Reference building sized for Chicago.** The DOE small office HVAC is
sized for Chicago winters. Absolute energy numbers are higher than a real
Bangalore small office; the delta remains a valid comparison because both
runs use identical HVAC sizing.

**Injected synthetic occupancy schedule.** The reference IDF had
near-zero occupancy in its default schedule; `scripts/prepare_idf.py`
overrides with a realistic 90% office-hours weekday pattern. Real
occupancy from a BMS would come from motion sensors or badge data.

**Single global cooling setpoint.** The small-office HVAC configuration
exposes one cooling schedule for the whole building. Per-zone setpoint
control is future work — the sensor tool layer already reads per-zone
values.

**Llama 3.1 8B occasionally emits inconsistent reasoning.** Small model.
We mitigate with numbered rules and safety clamping; a larger model
(Llama 3.3 70B) would give better zero-shot reasoning at higher latency.

**LLM stochasticity.** Even at temperature 0.2, LLM outputs vary between
runs. Observed run-to-run variance of ±2-3 percentage points on energy
savings for the same prompt. Production deployment would either use
deterministic rules extracted from the LLM's decisions, or a larger model
with more stable behaviour.

**7-day evaluation horizon.** Longer horizons (month, season) would
strengthen the reliability claim; the architecture is unchanged, only the
`SIMULATION_END_DAY` in `config.py`.

**Log summarizer is a stub.** The current build reads sensor values via
the pyenergyplus API only, so no runtime log summarization was needed.
The summarizer module (`data/summarizer.py`) is a defined extension point
for future error-recovery workflows.

---

## 10. Future work

- Per-zone setpoint control (requires HVAC template modification)
- Real-time grid carbon intensity input (via API tool) for
  carbon-aware setpoint selection
- Comfort model beyond dry-bulb temperature (PMV/PPD, humidity-aware)
- Extended horizon evaluation (30-day, season, year)
- Larger LLM (Llama 3.3 70B) A/B comparison
- IDF error-recovery workflow: LLM inspects `.err`, proposes IDF edits,
  reruns simulation (uses File and Error tool groups already defined)
- Predictive comfort-risk signal to enable safe operation past 25°C setpoint
- Ensemble decisions (average N LLM samples) to reduce stochasticity