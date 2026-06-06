# Dockerfile - Gemma 4 12B @ vLLM con la config horneada.
# Imagen base FIJADA (el 12B encoder-free no esta en :latest estable).
#   CUDA 13:   vllm/vllm-openai:gemma4-unified  (default)
#   CUDA 12.9: vllm/vllm-openai:gemma4-unified-cu129
ARG BASE_IMAGE=vllm/vllm-openai:gemma4-unified
FROM ${BASE_IMAGE}

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENV MODEL=unsloth/gemma-4-12B-it-qat-w4a16 \
    SERVED_MODEL_NAME=gemma-4-12b \
    DTYPE=bfloat16 \
    MAX_MODEL_LEN=131072 \
    MAX_NUM_SEQS=3 \
    GPU_MEMORY_UTILIZATION=0.92 \
    KV_CACHE_DTYPE=fp8 \
    TENSOR_PARALLEL_SIZE=1 \
    LIMIT_MM_PER_PROMPT='{"image": 4, "audio": 1}' \
    VISION_MAX_SOFT_TOKENS=280 \
    ENABLE_TOOLS=1 \
    ASYNC_SCHEDULING=1 \
    HOST=0.0.0.0 \
    PORT=8000

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --retries=30 --start-period=900s \
  CMD curl -fsS http://localhost:8000/health || exit 1
ENTRYPOINT ["/bin/bash", "/entrypoint.sh"]
