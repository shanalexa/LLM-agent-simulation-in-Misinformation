#!/usr/bin/env bash
# Phase-1 smoke / trial run against vLLM (set VLLM_API_BASE before running).

set -euo pipefail
cd "$(dirname "$0")/.."

: "${VLLM_API_BASE:?Set VLLM_API_BASE (OpenAI-compatible root, must end with /v1)}"

export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export LLM_MODEL="${LLM_MODEL:-Qwen3.6-35B-A3B}"
export LLM_MAX_TOKENS="${LLM_MAX_TOKENS:-2048}"

python main.py \
  --phase phase1 \
  --name "${RUN_NAME:-vllm_qwen36_trial}" \
  --llm_api_key "${OPENAI_API_KEY}" \
  --llm_model "${LLM_MODEL}" \
  --llm_workers "${LLM_WORKERS:-8}" \
  --no_json_mode
