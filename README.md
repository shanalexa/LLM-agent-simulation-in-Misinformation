# MGS — LLM Agent Misinformation Simulation

35 agents (AIS/DOA), Day 0 rumor broadcast, Days 1–7 peer interaction, Day 8 refutation branches, Days 9–15 follow-up.

## Setup

```bash
cd "/path/to/MGS"
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # VLLM_API_BASE and API keys
```

## Configuration (`config.yaml`)

Most experiment parameters (days, agent count, SIR threshold, rumor/refutation texts, LLM temperatures, output paths) live in **`config.yaml`**. Agent personas (5×7 archetypes, ideology backgrounds, `system_prompt_template`) live in **`personas.yaml`**. LLM task prompt templates live in **`prompts.yaml`**. Paths are set under `paths:` in config. Required keys must be present; missing values raise an error at startup (no silent Python fallbacks for experiment semantics).

```bash
# optional: custom config path
export MGS_CONFIG=/path/to/my_config.yaml
python main.py --config config.yaml --phase phase1
```

CLI still overrides single runs: `--belief_threshold`, `--rumor_topic`, `--no_daily_pickle`, `--vllm_api_base`, etc.

## Recommended: vLLM + Qwen3.6-35B-A3B (trial / main runs)

Use a **separate** environment for vLLM (GPU stack), and this repo’s `venv` only for `python main.py`.

### 1) Start vLLM (terminal A)

```bash
pip install -U vllm   # or follow https://docs.vllm.ai

chmod +x scripts/vllm_serve_qwen36.sh
./scripts/vllm_serve_qwen36.sh
```

Served model name: **`Qwen3.6-35B-A3B`**. Client API root comes from **`VLLM_API_BASE`** (not hardcoded in code).

Tune vLLM server via env before `vllm_serve_qwen36.sh`:

| Variable | Default | Notes |
|----------|---------|--------|
| `VLLM_PORT` | `8000` | Server listen port; set `VLLM_API_BASE` to match |
| `VLLM_TENSOR_PARALLEL` | `1` | Multi-GPU: set to GPU count |
| `VLLM_MAX_NUM_SEQS` | `16` | Should be ≥ `LLM_MAX_WORKERS` |
| `VLLM_MAX_MODEL_LEN` | `16384` | Enough for persona + LTM |
| `VLLM_GPU_MEMORY_UTILIZATION` | `0.92` | |

**Important:** Do **not** pass `--reasoning-parser qwen3` for MGS.

Set the client endpoint (example — use your host/port):

```bash
export VLLM_API_BASE="http://127.0.0.1:8000/v1"
```

Verify:

```bash
curl "${VLLM_API_BASE}/models"
```

### 2) Run Phase 1 trial (terminal B)

```bash
source venv/bin/activate

export VLLM_API_BASE="http://127.0.0.1:8000/v1"   # same host/port as vLLM
export OPENAI_API_KEY="EMPTY"
export LLM_MODEL="Qwen3.6-35B-A3B"
export LLM_MAX_TOKENS=2048

chmod +x scripts/run_phase1_trial.sh
./scripts/run_phase1_trial.sh
```

Or explicitly (reads `VLLM_API_BASE` from the environment):

```bash
python main.py --phase phase1 --name vllm_qwen36_trial \
  --llm_api_key EMPTY \
  --llm_model Qwen3.6-35B-A3B \
  --llm_workers 8 \
  --no_json_mode
```

CLI override: `--vllm_api_base "$VLLM_API_BASE"`

Success: `output/run-1/checkpoint_day7.json` and `LLM parallel: Day N` in the log.

### 3) After Phase 1 — branches

```bash
export VLLM_API_BASE="..."   # must still be set

python main.py --phase phase2 \
  --checkpoint_day7 output/run-1/checkpoint_day7.json \
  --llm_api_key EMPTY \
  --llm_model Qwen3.6-35B-A3B \
  --llm_workers 8 \
  --no_json_mode
```

Single branch: `--phase branch --branch NYT`.

## Ollama (alternative)

Point `VLLM_API_BASE` at Ollama’s OpenAI-compatible path:

```bash
export VLLM_API_BASE="http://127.0.0.1:11434/v1"
export OPENAI_API_KEY="ollama"
export LLM_MODEL="qwen2.5:7b"

python main.py --phase phase1 --name trial1 --no_json_mode
```

## Parallel LLM

```bash
export LLM_MAX_WORKERS=8
python main.py --phase phase1 --llm_workers 8 --no_json_mode
```

`VLLM_MAX_NUM_SEQS` on the server should be ≥ `LLM_MAX_WORKERS`.

## Run modes

| Command | What it runs |
|---------|----------------|
| `--phase phase1` | Day 0–7 → `checkpoint_day7.json` |
| `--phase branch --branch Control` | One branch Day 8–15 |
| `--phase phase2` | All five branches from checkpoint |
| `--phase full` | Day 0–7 (baseline), then five branches Day 8–15 (one command) |

## Outputs

- `output/run-1/run_manifest.json` — reproducibility record (config snapshot, git commit, seed, LLM, timestamps)
- `output/run-1/MGS-Phase1-agents-panel.csv` — daily per-agent belief & SIR (Day 0–7)
- `output/run-1/MGS_{Branch}-agents-panel.csv` — same for each intervention branch (Day 8–15)
- `output/run-1/checkpoint_day7.json` — Day 7 state (~1–3 MB)
- `output/run-1/results_{Branch}_Day15.json` — final panels with full `history_log` (~2–4 MB each)
- `checkpoint/run-1/*.pkl` — optional daily snapshots (~0.5–2 GB total if kept)

PI, IBG, RSR, AAS and plots are computed offline from the panel/JSON exports.

## Notes

- `agents_config.json` is regenerated on each run.
- Local Qwen/vLLM: use **`--no_json_mode`** unless JSON mode is verified.
- Cloud OpenAI: set **`VLLM_API_BASE=https://api.openai.com/v1`** and **`OPENAI_API_KEY`** (both required).
