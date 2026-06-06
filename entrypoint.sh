#!/usr/bin/env bash
#
# entrypoint.sh - Arma los argumentos de `vllm serve` desde variables de
# entorno y arranca el servidor OpenAI-compatible de vLLM para Gemma 4 12B
# (encoder-free: vision + audio nativos).
#
# Contexto del 12B: 256K (262144) segun model card. Algunas configs de
# vLLM lo fijan a 131072. NO existe 1M en este modelo.
#
set -euo pipefail

# ----- Modelo -----
# Default: checkpoint w4a16 (QAT) de Unsloth, NO gated (sin token), cabe en
# 16-24GB. vLLM detecta compressed-tensors solo. Para bf16 (40GB+) usa
# unsloth/gemma-4-12b-it; oficial gated: google/gemma-4-12B-it.
MODEL="${MODEL:-unsloth/gemma-4-12B-it-qat-w4a16}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-gemma-4-12b}"
DTYPE="${DTYPE:-bfloat16}"

# ----- Contexto y concurrencia -----
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"   # default consumo 128K; max 262144
MAX_NUM_SEQS="${MAX_NUM_SEQS:-3}"          # clientes concurrentes

# ----- Memoria GPU / paralelismo -----
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.92}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"

# ----- Multimodal (vision + audio, encoder-free) -----
LIMIT_MM_PER_PROMPT="${LIMIT_MM_PER_PROMPT:-}"
[ -z "$LIMIT_MM_PER_PROMPT" ] && LIMIT_MM_PER_PROMPT='{"image": 4, "audio": 1}'
VISION_MAX_SOFT_TOKENS="${VISION_MAX_SOFT_TOKENS:-280}"  # 70|140|280|560|1120

# ----- Agentes de codigo: tool-calling + thinking -----
ENABLE_TOOLS="${ENABLE_TOOLS:-1}"          # 1 = activa tool-calling/reasoning
CHAT_TEMPLATE="${CHAT_TEMPLATE:-}"         # opcional: ruta a plantilla jinja

# ----- Red -----
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

# ----- Opcionales -----
ASYNC_SCHEDULING="${ASYNC_SCHEDULING:-1}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"    # default consumo; fp8 -> ~50% menos KV
ROPE_SCALING="${ROPE_SCALING:-}"           # solo si superas el ctx nativo
QUANTIZATION="${QUANTIZATION:-}"           # p.ej. compressed-tensors / awq / fp8
VLLM_API_KEY="${VLLM_API_KEY:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

ARGS=(
  --model "$MODEL"
  --served-model-name "$SERVED_MODEL_NAME"
  --dtype "$DTYPE"
  --max-model-len "$MAX_MODEL_LEN"
  --max-num-seqs "$MAX_NUM_SEQS"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
  --limit-mm-per-prompt "$LIMIT_MM_PER_PROMPT"
  --mm-processor-kwargs "{\"max_soft_tokens\": ${VISION_MAX_SOFT_TOKENS}}"
  --trust-remote-code
  --host "$HOST"
  --port "$PORT"
)

if [ "$ENABLE_TOOLS" = "1" ]; then
  ARGS+=(--enable-auto-tool-choice --reasoning-parser gemma4 --tool-call-parser gemma4)
  [ -n "$CHAT_TEMPLATE" ] && ARGS+=(--chat-template "$CHAT_TEMPLATE")
fi

[ "$ASYNC_SCHEDULING" = "1" ] && ARGS+=(--async-scheduling)
[ -n "$KV_CACHE_DTYPE" ]      && ARGS+=(--kv-cache-dtype "$KV_CACHE_DTYPE")
[ -n "$ROPE_SCALING" ]        && ARGS+=(--rope-scaling "$ROPE_SCALING")
[ -n "$QUANTIZATION" ]        && ARGS+=(--quantization "$QUANTIZATION")
[ -n "$VLLM_API_KEY" ]        && ARGS+=(--api-key "$VLLM_API_KEY")
# shellcheck disable=SC2206
[ -n "$EXTRA_ARGS" ]          && ARGS+=($EXTRA_ARGS)

echo "============================================================"
echo " Gemma 4 12B  ->  vLLM (OpenAI-compatible)"
echo "  modelo:        $MODEL  (served as: $SERVED_MODEL_NAME)"
echo "  contexto:      $MAX_MODEL_LEN tokens (max 262144)"
echo "  concurrencia:  $MAX_NUM_SEQS clientes"
echo "  multimodal:    $LIMIT_MM_PER_PROMPT  | vision tokens: $VISION_MAX_SOFT_TOKENS"
echo "  tool-calling:  $ENABLE_TOOLS"
echo "  gpu_mem_util:  $GPU_MEMORY_UTILIZATION  | TP: $TENSOR_PARALLEL_SIZE  | dtype: $DTYPE"
[ -n "$QUANTIZATION" ]   && echo "  quantization:  $QUANTIZATION"
[ -n "$KV_CACHE_DTYPE" ] && echo "  kv_cache:      $KV_CACHE_DTYPE"
[ -n "$ROPE_SCALING" ]   && echo "  rope_scaling:  $ROPE_SCALING"
echo "  endpoint:      http://$HOST:$PORT/v1"
echo "============================================================"
echo "vllm serve ${ARGS[*]}"
echo "============================================================"

exec vllm serve "${ARGS[@]}"
