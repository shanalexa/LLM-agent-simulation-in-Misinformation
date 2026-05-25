#!/usr/bin/env bash
# vLLM OpenAI-compatible server for MGS trial runs (Qwen3.6-35B-A3B).
# Do NOT use --reasoning-parser: simulation needs plain JSON/text, not chain-of-thought blocks.

set -euo pipefail

MODEL="${VLLM_MODEL:-Qwen/Qwen3.6-35B-A3B}"
SERVED_NAME="${VLLM_SERVED_NAME:-Qwen3.6-35B-A3B}"
PORT="${VLLM_PORT:-8000}"
TP="${VLLM_TENSOR_PARALLEL:-1}"
MAX_LEN="${VLLM_MAX_MODEL_LEN:-16384}"
MAX_SEQS="${VLLM_MAX_NUM_SEQS:-16}"
GPU_UTIL="${VLLM_GPU_MEMORY_UTILIZATION:-0.92}"

echo "Starting vLLM: ${MODEL}"
echo "  served-model-name: ${SERVED_NAME}"
echo "  port: ${PORT}  tensor-parallel: ${TP}  max-num-seqs: ${MAX_SEQS}"
echo "  Then set: export VLLM_API_BASE=\"http://<host>:${PORT}/v1\""

exec vllm serve "${MODEL}" \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --served-model-name "${SERVED_NAME}" \
  --tensor-parallel-size "${TP}" \
  --max-model-len "${MAX_LEN}" \
  --max-num-seqs "${MAX_SEQS}" \
  --gpu-memory-utilization "${GPU_UTIL}" \
  --trust-remote-code
