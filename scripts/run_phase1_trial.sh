#!/usr/bin/env bash
# Phase-1 smoke / trial run against local vLLM (after scripts/vllm_serve_qwen36.sh is up).

set -euo pipefail
cd "$(dirname "$0")/.."

export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://127.0.0.1:8000/v1}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export LLM_MODEL="${LLM_MODEL:-Qwen3.6-35B-A3B}"
export LLM_MAX_TOKENS="${LLM_MAX_TOKENS:-2048}"

python main.py \
  --phase phase1 \
  --name "${RUN_NAME:-vllm_qwen36_trial}" \
  --llm_base_url "${OPENAI_BASE_URL}" \
  --llm_api_key "${OPENAI_API_KEY}" \
  --llm_model "${LLM_MODEL}" \
  --llm_workers "${LLM_WORKERS:-8}" \
  --no_json_mode
