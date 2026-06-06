# Gemma 4 12B — despliegue local (vLLM + llama.cpp)

**Gemma 4 12B Unified**: modelo multimodal *encoder-free* (texto + visión +
audio), Apache 2.0, **contexto 256K (262144)**, fuerte en coding/agentic
(LiveCodeBench v6 72%, Codeforces ELO 1659). Servido con API **compatible con
OpenAI** para agentes de código y uso multimodal.

Dos caminos según el hardware:

- **vLLM** (Docker) para la **RTX 5070 Ti** — máximo throughput.
- **llama.cpp** (GGUF de Unsloth) para todo: 5070 Ti (CUDA), **Ryzen 8700G**
  (iGPU 780M, Vulkan) y **MacBook Pro M4 Pro** (Metal, nativo). Más simple,
  multicliente, visión por `mmproj` y speculative decoding con el draft MTP.

> El contexto real del 12B es **256K**. (La receta de vLLM fija `config.json` a
> 131072 de forma conservadora; el model card y el GGUF confirman 262144.) **No
> existe 1M** en este modelo. Tus 200K caben bien.

> **GGUF por defecto: QAT.** Las rutas llama.cpp usan
> `unsloth/gemma-4-12B-it-qat-GGUF:UD-Q4_K_XL` (~6.7GB). El QAT
> (*Quantization-Aware Training*) da calidad **casi de bf16 a 4-bit**, mejor que
> un Q4 normal del mismo tamaño, y al pesar menos **deja más memoria para el KV
> cache** (= más contexto/clientes). Solo hay 4-bit; para Q5/Q6/Q8/BF16 o el
> draft MTP usa el repo no-QAT `unsloth/gemma-4-12b-it-GGUF`.

---

## Realidad por equipo

| Equipo | Camino recomendado | Pesos | Contexto realista | Clientes |
|---|---|---|---|---|
| **RTX 5070 Ti 16GB** | vLLM (`cuda`) **w4a16** · o llama.cpp (`llamacpp-cuda`) **GGUF QAT** | bf16 (~24GB) NO cabe; usa w4a16 o GGUF QAT (~6.7GB) | 128K cómodo; 200K con `fp8` KV / ctx ajustado | 3 |
| **Ryzen 8700G 64GB (780M)** | llama.cpp Vulkan (`llamacpp`) | GGUF Q4 (~7GB), usa RAM | 256K total (≈87K×3) holgado | 3 vía `--parallel` |
| **MBP M4 Pro 24GB** | llama.cpp Metal / MLX (nativo) | GGUF Q4 (~7GB) | ≈64K×3, o 200K para 1 | 3 vía `--parallel` |

**¿llama.cpp es multicliente?** Sí: `llama-server --parallel N` con *continuous
batching*. El `-c` es el contexto **total** y se reparte entre las N ranuras, así
que "3 clientes a X" = `-c` de `3·X`.

---

## 1) RTX 5070 Ti

### Opción A — vLLM (Docker, máximo throughput)

Imagen **fijada** `vllm/vllm-openai:gemma4-unified` (el `:latest` aún no trae el
12B encoder-free). En 16GB **debes** usar pesos **w4a16**.

```bash
cp .env.example .env
# .env:
#   HF_TOKEN=hf_xxx
#   MODEL=<repo-w4a16-de-gemma-4-12B-it>   QUANTIZATION=compressed-tensors
#   MAX_MODEL_LEN=200000   MAX_NUM_SEQS=3   KV_CACHE_DTYPE=fp8
docker compose --profile cuda up -d
docker compose --profile cuda logs -f
```

CUDA 12.9: `VLLM_CUDA_IMAGE=vllm/vllm-openai:gemma4-unified-cu129`. El KV cache no
se cuantiza con w4a16; a 200K es lo que más pesa, por eso `KV_CACHE_DTYPE=fp8`. Si
OOM, baja `MAX_MODEL_LEN` o `MAX_NUM_SEQS`. (Blackwell sm_120 va en la imagen CUDA 13.)

### Opción B — llama.cpp CUDA (más simple, sin buscar checkpoint w4a16)

Usa el GGUF de Unsloth (se autodescarga). Más fácil de encajar en 16GB.

```bash
cp .env.example .env
# .env:  (QAT por defecto)  LLAMACPP_CTX=131072   MAX_NUM_SEQS=3
docker compose --profile llamacpp-cuda up -d
```

---

## 2) Ryzen 8700G — llama.cpp Vulkan (Docker)

vLLM-ROCm **solo** soporta GPUs Instinct (MI300X+), no el iGPU 780M. El camino
real es **llama.cpp Vulkan**, que corre en la 780M y tira de tus 64GB de RAM.

```bash
cp .env.example .env
# .env:  (QAT por defecto)  LLAMACPP_CTX=262144   MAX_NUM_SEQS=3
docker compose --profile llamacpp up -d
curl http://localhost:8000/v1/models
```

---

## 3) MacBook Pro M4 Pro — nativo (sin Docker)

Docker en Mac no ve la GPU; se corre nativo para usar Metal.

```bash
brew install llama.cpp
llama-server \
  -hf unsloth/gemma-4-12B-it-qat-GGUF:UD-Q4_K_XL \
  --alias gemma-4-12b \
  -c 196608 --parallel 3 \      # ≈64K x 3 (200K x 3 no cabe en 24GB)
  -ngl 999 --jinja \
  --temp 1.0 --top-p 0.95 --top-k 64 \
  --host 0.0.0.0 --port 8000
# 1 cliente a 200K:  -c 200000 --parallel 1
```

Alternativa MLX: `pip install mlx-lm && mlx_lm.server --model mlx-community/gemma-4-12B-it --port 8000`.

---

## Multimodal y velocidad (llama.cpp)

**Visión:** el `mmproj` se autodescarga con `-hf`. Si quieres forzar uno:
`LLAMACPP_EXTRA=--mmproj <ruta>` (en el repo: `mmproj-F16.gguf`, `mmproj-F32.gguf`).

**Audio:** el modelo soporta audio, pero el soporte en llama.cpp es parcial según
versión. Si el audio es imprescindible, usa la ruta **vLLM**, que sí lo expone.

**Speculative decoding (MTP):** el repo trae draft `MTP/...`. Acelera la
generación:
```
LLAMACPP_EXTRA=-md unsloth/gemma-4-12b-it-GGUF:MTP-Q8_0 --draft-max 8 --draft-min 1
```
En vLLM: `EXTRA_ARGS=--speculative-config '{"model":"google/gemma-4-12B-it-assistant","num_speculative_tokens":6}'`.

---

## Usar el modelo

Endpoint común: `http://localhost:8000/v1`, modelo `gemma-4-12b`.
Sampling recomendado por Google: **temp 1.0, top_p 0.95, top_k 64**.

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
r = client.chat.completions.create(
    model="gemma-4-12b",
    messages=[{"role": "user", "content": "Refactoriza esta función..."}],
    temperature=1.0, top_p=0.95, extra_body={"top_k": 64},
)
print(r.choices[0].message.content)
```

Prueba rápida (texto/imagen/audio): `python test_api.py --image URL` / `--audio clip.wav`.

**Agentes de código:** vLLM con `ENABLE_TOOLS=1` activa function-calling +
thinking (`--tool-call-parser gemma4`, `--reasoning-parser gemma4`); llama.cpp lo
hace con `--jinja` (plantilla de tool-call embebida en el GGUF). Apunta tu agente
(Continue, Aider, OpenCode) a `OPENAI_BASE_URL=http://localhost:8000/v1`,
`OPENAI_MODEL=gemma-4-12b`. Tip multimodal: imagen **antes** del texto, audio
**después** (lo indica el model card).

---

## Configuración (vLLM, vía `.env`)

| Variable | Default | Qué hace |
|---|---|---|
| `MAX_MODEL_LEN` | `200000` | Contexto (máx 262144 / 256K). |
| `MAX_NUM_SEQS` | `3` | Clientes concurrentes. |
| `MODEL` / `QUANTIZATION` | `gemma-4-12B-it` / — | w4a16 + `compressed-tensors` en 16GB. |
| `KV_CACHE_DTYPE` | — | `fp8` ≈ −50% KV cache. |
| `LIMIT_MM_PER_PROMPT` | `{"image":4,"audio":1}` | Máx. imágenes/audios. `{"image":0,"audio":0}` = solo texto. |
| `VISION_MAX_SOFT_TOKENS` | `280` | Detalle visual: 70/140/280/560/1120. |
| `ENABLE_TOOLS` | `1` | Tool-calling + reasoning. |
| `GGUF_QUANT` / `LLAMACPP_CTX` | `Q4_K_M` / `262144` | (perfiles llama.cpp) quant y contexto total. |

---

## Fuentes

- GGUF + mmproj + MTP (Unsloth): <https://huggingface.co/unsloth/gemma-4-12b-it-GGUF> · guía: <https://docs.unsloth.ai/models/gemma-4>
- Receta vLLM (12B, w4a16): <https://recipes.vllm.ai/Google/gemma-4-12B-it?variant=w4a16>
- Receta familia Gemma 4: <https://docs.vllm.ai/projects/recipes/en/latest/Google/Gemma4.html>
- Model card / pesos: <https://huggingface.co/google/gemma-4-12B-it>
- Anuncio: <https://blog.google/innovation-and-ai/technology/developers-tools/introducing-gemma-4-12b/>
