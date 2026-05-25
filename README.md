# MGS — LLM Agent Misinformation Simulation

35 agents (AIS/DOA), Day 0 rumor broadcast, Days 1–7 peer interaction, Day 8 refutation branches, Days 9–15 follow-up.

## Setup

```bash
cd "/path/to/MGS"
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Recommended: vLLM + Qwen3.6-35B-A3B (trial / main runs)

Use a **separate** environment for vLLM (GPU stack), and this repo’s `venv` only for `python main.py`.

### 1) Start vLLM (terminal A)

```bash
pip install -U vllm   # or follow https://docs.vllm.ai

# Optional: FP8 weights on 96GB VRAM
# export VLLM_MODEL=Qwen/Qwen3.6-35B-A3B-FP8

chmod +x scripts/vllm_serve_qwen36.sh
./scripts/vllm_serve_qwen36.sh
```

Default listens on `http://127.0.0.1:8000/v1`, served name **`Qwen3.6-35B-A3B`**.

Tune via env before the script:

| Variable | Default | Notes |
|----------|---------|--------|
| `VLLM_TENSOR_PARALLEL` | `1` | Multi-GPU: set to GPU count |
| `VLLM_MAX_NUM_SEQS` | `16` | Should be ≥ `--llm_workers` |
| `VLLM_MAX_MODEL_LEN` | `16384` | Enough for persona + LTM; lower saves VRAM |
| `VLLM_GPU_MEMORY_UTILIZATION` | `0.92` | |

**Important:** Do **not** pass `--reasoning-parser qwen3` for MGS. Reasoning tags break JSON/tweet parsing.

Verify the server:

```bash
curl http://127.0.0.1:8000/v1/models
```

### 2) Run Phase 1 trial (terminal B)

```bash
source venv/bin/activate

export OPENAI_BASE_URL="http://127.0.0.1:8000/v1"
export OPENAI_API_KEY="EMPTY"
export LLM_MODEL="Qwen3.6-35B-A3B"
export LLM_MAX_TOKENS=2048

chmod +x scripts/run_phase1_trial.sh
./scripts/run_phase1_trial.sh
```

Or explicitly:

```bash
python main.py --phase phase1 --name vllm_qwen36_trial \
  --llm_base_url http://127.0.0.1:8000/v1 \
  --llm_api_key EMPTY \
  --llm_model Qwen3.6-35B-A3B \
  --llm_workers 8 \
  --no_json_mode
```

Success: `output/run-1/checkpoint_day7.json` and console lines `LLM parallel: Day N (...)`.

### 3) After Phase 1 — branches

```bash
python main.py --phase phase2 \
  --checkpoint_day7 output/run-1/checkpoint_day7.json \
  --llm_base_url http://127.0.0.1:8000/v1 \
  --llm_api_key EMPTY \
  --llm_model Qwen3.6-35B-A3B \
  --llm_workers 8 \
  --no_json_mode
```

Single branch only: `--phase branch --branch NYT`.

## Ollama (alternative)

```bash
ollama serve
ollama pull qwen2.5:7b

export OPENAI_BASE_URL="http://127.0.0.1:11434/v1"
export OPENAI_API_KEY="ollama"
export LLM_MODEL="qwen2.5:7b"

python main.py --phase phase1 --name trial1 --no_json_mode
```

## Parallel LLM

After the daily snapshot, each agent’s LLM calls run in a thread pool (default **8** workers). Match vLLM concurrency:

```bash
# VLLM_MAX_NUM_SEQS >= LLM_WORKERS
export LLM_MAX_WORKERS=8
python main.py --phase phase1 --llm_workers 8 --no_json_mode ...
```

Use `--llm_workers 1` for debugging. End-of-day barrier unchanged: snapshot → parallel compute → serial apply.

## Run modes

| Command | What it runs |
|---------|----------------|
| `--phase phase1` | Day 0–7 → `checkpoint_day7.json` |
| `--phase branch --branch Control` | One branch Day 8–15 |
| `--phase phase2` | All five branches from checkpoint |
| `--phase full` | Phase 1 + five branches |

## Outputs

- `output/run-1/checkpoint_day7.json` — Day 7 state (~1–3 MB)
- `output/run-1/results_{Branch}_Day15.json` — final panels (~2–4 MB each)
- `checkpoint/run-1/*.pkl` — optional daily snapshots (~0.5–2 GB total if kept)

## Notes

- `agents_config.json` is regenerated on each run.
- Local Qwen: always use **`--no_json_mode`** unless you confirmed vLLM JSON mode works.
- Cloud OpenAI: set `OPENAI_API_KEY`, omit `--llm_base_url`, omit `--no_json_mode` if desired.
