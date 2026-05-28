#!/usr/bin/env bash
# vLLM OpenAI-compatible server for MGS (Qwen/Qwen2.5-32B-Instruct).

set -euo pipefail

MODEL="${VLLM_MODEL:-Qwen/Qwen2.5-32B-Instruct}"
SERVED_NAME="${VLLM_SERVED_NAME:-Qwen2.5-32B-Instruct}"
PORT="${VLLM_PORT:-8000}"
TP="${VLLM_TENSOR_PARALLEL:-1}"
MAX_LEN="${VLLM_MAX_MODEL_LEN:-16384}"
MAX_SEQS="${VLLM_MAX_NUM_SEQS:-16}"
GPU_UTIL="${VLLM_GPU_MEMORY_UTILIZATION:-0.92}"

echo "Starting vLLM: ${MODEL}"
echo "  served-model-name: ${SERVED_NAME}"
echo "  port: ${PORT}  tensor-parallel: ${TP}  max-num-seqs: ${MAX_SEQS}"
echo "  Then set: export VLLM_API_BASE=\"http://<host>:${PORT}/v1\""
echo "            export LLM_MODEL=\"${SERVED_NAME}\""

exec vllm serve "${MODEL}" \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --served-model-name "${SERVED_NAME}" \
  --tensor-parallel-size "${TP}" \
  --max-model-len "${MAX_LEN}" \
  --max-num-seqs "${MAX_SEQS}" \
  --gpu-memory-utilization "${GPU_UTIL}" \
  --trust-remote-code
