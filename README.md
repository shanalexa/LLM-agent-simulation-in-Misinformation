# MGS — LLM Agent Misinformation Simulation

Multi-agent simulation (AIS/DOA): 35 LLM-driven agents, Day 0 rumor broadcast, Days 1–7 peer diffusion, Day 8 five refutation conditions, Days 9–15 follow-up. The program runs the experiment and exports **raw per-agent data**; aggregate metrics (PI, IBG, RSR, AAS) and plots are computed **offline** in your analysis workflow.

## Experiment timeline

| Days | Phase | What happens |
|------|--------|----------------|
| 0 | Rumor | All agents receive misinformation; initial belief & SIR |
| 1–7 | Baseline | Daily peer interaction (shared across conditions) |
| 7 | Checkpoint | State frozen → `checkpoint_day7.json` |
| 8 | Intervention | One refutation text per branch (NYT / WSJ / AI / Neutral / Control) |
| 9–15 | Follow-up | Peer diffusion under that branch |

Five branches share the **same** Day 0–7 history; they diverge from Day 8.

## Repository layout

```
MGS/
├── main.py              # CLI entry: phases, runs, exports
├── config.py            # Load & validate YAML; get_config()
├── config.yaml          # Experiment, paths, LLM, rumor/refutation texts
├── personas.yaml        # 5×7 personas + system_prompt_template
├── prompts.yaml         # LLM task templates (day0 / daily / intervention)
├── citizen.py           # DOA: belief, SIR, STM/LTM, LLM calls
├── world.py             # AIS: schedule, days, checkpoint, DataCollector
├── utils.py             # LLM client, agent config builder
├── run_manifest.py      # run_manifest.json for reproducibility
├── scripts/
│   ├── vllm_serve_qwen25.sh
│   └── run_phase1_trial.sh
├── output/run-1/        # Created at runtime
└── checkpoint/run-1/    # Optional .pkl snapshots
```

**Design principle:** simulation code only reads config via `init_config()` → `get_config()`. No duplicate experiment defaults in Python (missing required YAML keys fail at startup).

## Configuration

| File | Purpose |
|------|---------|
| `config.yaml` | `experiment.*`, `paths.*`, `llm.*`, `texts.*` (misinformation + refutation per branch), `run.*`, `world.*` |
| `personas.yaml` | `system_prompt_template`, `archetypes`, `ideology_profiles` → 35 agents |
| `prompts.yaml` | Cognitive / STM / LTM prompt templates |

Paths in `config.yaml`:

```yaml
paths:
  personas: personas.yaml
  prompts: prompts.yaml
```

Optional: `export MGS_CONFIG=/path/to/config.yaml`

### Parameters used only in offline analysis

`experiment.ibg_liberal_max` and `experiment.ibg_conservative_min` (default 3 and 5) define ideology groups for **Inter-group Belief Gap (IBG)** when you analyze exports. The simulator does not compute IBG during a run.

### LLM settings (priority)

1. CLI: `--llm_model`, `--vllm_api_base`, `--llm_api_key`, `--llm_workers`, …
2. Environment: `LLM_MODEL`, `VLLM_API_BASE`, `OPENAI_API_KEY`, `LLM_MAX_WORKERS`, `LLM_MAX_TOKENS`
3. `config.yaml` → `llm.model` (default `Qwen2.5-32B-Instruct`)

**Required before run:** `VLLM_API_BASE` and `OPENAI_API_KEY` must be set (no silent fallback). Local vLLM typically uses `OPENAI_API_KEY=EMPTY`.

## Setup

```bash
cd "/path/to/MGS"
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then export variables or use direnv
```

Use a **separate** environment for the vLLM server; this `venv` is only for `python main.py`.

## Run (recommended: vLLM + Qwen2.5-32B-Instruct)

### Terminal A — start vLLM

```bash
chmod +x scripts/vllm_serve_qwen25.sh
./scripts/vllm_serve_qwen25.sh
```

Hub model: **`Qwen/Qwen2.5-32B-Instruct`**. Client `LLM_MODEL` / `llm.model` must match **`--served-model-name`** (default **`Qwen2.5-32B-Instruct`**).

| Env (server) | Default | Notes |
|--------------|---------|--------|
| `VLLM_PORT` | `8000` | Match `VLLM_API_BASE` |
| `VLLM_MAX_NUM_SEQS` | `16` | Should be ≥ `LLM_MAX_WORKERS` |
| `VLLM_TENSOR_PARALLEL` | `1` | Set to GPU count if multi-GPU |

### Terminal B — simulation

```bash
source venv/bin/activate

export VLLM_API_BASE="http://127.0.0.1:8000/v1"
export OPENAI_API_KEY="EMPTY"
export LLM_MODEL="Qwen2.5-32B-Instruct"
export LLM_MAX_WORKERS=8
export LLM_MAX_TOKENS=2048
```

**Phase 1 only (trial, ~Day 0–7):**

```bash
./scripts/run_phase1_trial.sh
# or:
python main.py --phase phase1 --name trial1 --llm_workers 8 --no_json_mode --no_daily_pickle
```

**Full experiment (recommended for thesis runs):**

```bash
python main.py --phase full --name MGS --llm_workers 8 --no_json_mode --no_daily_pickle
```

**After Phase 1, five branches only:**

```bash
python main.py --phase phase2 \
  --checkpoint_day7 output/run-1/checkpoint_day7.json \
  --llm_workers 8 --no_json_mode --no_daily_pickle
```

**Single branch:**

```bash
python main.py --phase branch --branch NYT \
  --checkpoint_day7 output/run-1/checkpoint_day7.json \
  --no_json_mode --no_daily_pickle
```

### Run modes

| `--phase` | Behavior |
|-----------|----------|
| `phase1` | Day 0–7 → `checkpoint_day7.json` |
| `phase2` | All five branches from existing Day 7 checkpoint |
| `branch` | One branch (`--branch NYT`) Day 8–15 |
| `full` | Phase 1 (stops at Day 7), then five branches (one command) |

`--no_of_runs N` creates `output/run-1` … `output/run-N`.

### Disable large pickle snapshots

```bash
python main.py --phase full --no_daily_pickle
```

Or in `config.yaml`: `world.save_daily_pickle: false`. JSON/CSV outputs are enough for analysis; daily `.pkl` files are only for mid-run resume.

## Outputs (`output/run-1/`)

| Artifact | Description |
|----------|-------------|
| `run_manifest.json` | Git commit, `random_seed`, LLM config (key redacted), CLI args, snapshots of `config.yaml` / `personas.yaml` / `prompts.yaml`, start/end time |
| `checkpoint_day7.json` | All agents at end of Day 7 (branch fork point) |
| `MGS-Phase1-agents-panel.csv` | Per-agent daily rows, Days 0–7 |
| `MGS_{Branch}-agents-panel.csv` | Per-agent daily rows, Days 8–15 (e.g. `MGS_NYT-…`) |
| `results_{Branch}_Day15.json` | Full `history_log` per agent at Day 15 |

**Agents panel columns** include: `simulation_day`, `experiment_branch`, `BeliefIntensity`, `RumorBeliefState`, `PoliticalIdeology`, demographics, `AgentConfigID`. Mesa index `Step` is the collection order within that file (for branch files, Day 8 → `Step` 0).

**Building 16-day series for five conditions:** concatenate Phase1 CSV (Days 0–7) with each branch CSV (Days 8–15), keyed on `simulation_day` and `experiment_branch`.

`agents_config.json` is regenerated at startup (35 agents from `personas.yaml`).

Optional: `checkpoint/run-1/*.pkl` if `save_daily_pickle` is enabled.

## Offline analysis (not in this repo)

Compute in Jupyter/R/Python from panel + JSON:

- **PI** — std of `BeliefIntensity` across 35 agents per day  
- **IBG** — \|mean(belief, ideology ≤ `ibg_liberal_max`) − mean(belief, ideology ≥ `ibg_conservative_min`)\|  
- **RSR** — share of population in `Recovered` (completed Infected→Recovered)  
- **AAS** — mean \|belief(t) − belief(0)\| (or your thesis-specified windows, e.g. Day8−Day0, Day15−Day7)

Plots and statistical tests belong in your analysis project, not in `main.py`.

## Runtime notes

- **Parallel LLM:** `--llm_workers` / `LLM_MAX_WORKERS` (default 8). Server `VLLM_MAX_NUM_SEQS` ≥ workers.
- **JSON mode:** use `--no_json_mode` for local Qwen/vLLM unless you have verified JSON `response_format` support.
- **Reproducibility:** `experiment.random_seed` sets Python/NumPy RNG (contact sampling). LLM outputs remain stochastic unless the server temperature is 0.
- **Rough runtime:** full run ~5000 LLM calls; ~3–8 h on one GPU with 8 workers (model-dependent).

## Cloud OpenAI

```bash
export VLLM_API_BASE="https://api.openai.com/v1"
export OPENAI_API_KEY="sk-..."
export LLM_MODEL="gpt-4o"   # or your deployment name
```

Both `VLLM_API_BASE` and `OPENAI_API_KEY` are mandatory.
