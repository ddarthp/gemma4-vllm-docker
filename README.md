# Gemma 4 12B — despliegue local (vLLM + llama.cpp)

Despliegue OpenAI-compatible de **Gemma 4 12B Unified**, el modelo multimodal
*encoder-free* de Google (texto + imagen + audio), Apache 2.0, **contexto 256K
(262144)**, con buen rendimiento en coding/agentic (LiveCodeBench v6 72%,
Codeforces ELO 1659). Pensado para agentes de código y uso multimodal local.

Incluye dos motores según tu hardware:

- **vLLM** (Docker) — máximo throughput en GPU NVIDIA (y AMD Instinct).
- **llama.cpp** (GGUF) — el camino portable: NVIDIA (CUDA), GPUs/iGPUs AMD y
  Intel (Vulkan), Apple Silicon (Metal, nativo) y CPU. Multicliente, visión por
  `mmproj` y *speculative decoding* opcional.

Todo se controla por `.env`. Los GGUF se descargan solos con `-hf`.

---

## ¿Qué motor elijo?

| Tu hardware | Motor recomendado | Perfil |
|---|---|---|
| GPU NVIDIA de centro de datos (≥40 GB: A100/H100/L40S/RTX 6000 Ada…) | vLLM bf16 | `cuda` |
| GPU NVIDIA de consumo (16–24 GB) | **llama.cpp CUDA (GGUF)** — vLLM aún no tiene w4a16 funcional | `llamacpp-cuda` |
| GPU AMD Instinct (MI300X/MI325X/MI350X/MI355X) | vLLM-ROCm | (ver nota AMD) |
| GPU/iGPU AMD o Intel de consumo | llama.cpp Vulkan | `llamacpp` |
| Apple Silicon (Mac M-series) | llama.cpp Metal / MLX (**nativo, sin Docker**) | — |
| Solo CPU | vLLM CPU o llama.cpp | `cpu` / `llamacpp` |

> **AMD:** vLLM-ROCm **solo** soporta GPUs Instinct (CDNA, MI300X+). Las GPU/iGPU
> de consumo (RDNA, p.ej. Radeon integradas o de escritorio) **no** están
> soportadas por vLLM; para esas usa el perfil `llamacpp` (Vulkan).
>
> **Apple Silicon:** vLLM no tiene backend Metal y Docker en macOS no accede a la
> GPU. En Mac se corre **nativo** (sección al final).

---

## Requisitos mínimos (memoria)

Dos cosas consumen memoria: los **pesos** del modelo (fijo) y el **KV cache**
(crece con el contexto y con el nº de clientes). Para **un solo cliente**:

| Configuración | Pesos | Contexto | VRAM / RAM mínima* |
|---|---|---|---|
| 4-bit (QAT/GGUF Q4) · texto · contexto corto (8–16K) | ~6.7 GB | 8–16K | **~10–12 GB** |
| 4-bit · contexto medio (32–64K) | ~6.7 GB | 32–64K | ~14–18 GB |
| 4-bit · contexto largo (128K) | ~6.7 GB | 128K | ~20–28 GB (usa KV `fp8`) |
| 4-bit · contexto máximo (256K) | ~6.7 GB | 256K | ~32–40 GB (KV `fp8`) |
| 8-bit (Q8) · contexto medio | ~12.5 GB | 32–64K | ~20–24 GB |
| **bf16 sin cuantizar** (vLLM) | ~24 GB | 32K | **1 GPU de 40 GB+** |

\* Estimaciones orientativas; el KV cache real depende de la arquitectura (Gemma
usa atención por ventana deslizante que reduce mucho el cache en contextos
largos). Si te quedas sin memoria (OOM): baja el contexto, activa KV cache `fp8`,
reduce clientes, o usa un quant más pequeño.

**Reglas prácticas:**

- En **16 GB** entra cómodo en **4-bit** con contexto moderado y 1–3 clientes.
- Para **bf16** necesitas una GPU de **40 GB+** (lo indica la receta de vLLM).
- Cada cliente concurrente necesita su propia porción de KV cache: más clientes
  = menos contexto por cliente para la misma memoria.

---

## Consideraciones clave

**Cuantización — QAT por defecto.** Las rutas llama.cpp usan
`unsloth/gemma-4-12B-it-qat-GGUF:UD-Q4_K_XL` (~6.7 GB). El **QAT**
(*Quantization-Aware Training*) da calidad **casi de bf16 a 4-bit**, mejor que un
Q4 normal del mismo tamaño, y al pesar menos **deja más memoria para el KV cache**
(= más contexto / más clientes). Solo hay 4-bit; si quieres Q5/Q6/Q8/BF16 o el
draft MTP, usa el repo no-QAT `unsloth/gemma-4-12b-it-GGUF`. En **vLLM** sobre GPU
de 16–24 GB necesitas pesos **w4a16** (formato *compressed-tensors*); en GPU de
40 GB+ puedes usar bf16 directamente.

**Contexto.** El 12B soporta hasta **256K (262144)**. (Algunas configs de vLLM
fijan `config.json` a 131072 de forma conservadora; súbelo con `MAX_MODEL_LEN`
si tu build lo acepta.) **No existe contexto de 1M** en este modelo. El KV cache
domina la memoria en contextos largos: `KV_CACHE_DTYPE=fp8` lo reduce ~50%.

**Concurrencia.** `MAX_NUM_SEQS` (vLLM) o `--parallel` (llama.cpp) fija los
clientes simultáneos. En llama.cpp el `-c` es el contexto **total** del servidor
y se reparte entre las ranuras: "N clientes a X tokens" ⇒ `-c = N·X`.

**Multimodal.** Visión: en vLLM es nativa; en llama.cpp el `mmproj` se
autodescarga con `-hf` (o fuérzalo con `--mmproj`). Audio: el modelo lo soporta y
en vLLM se expone; en llama.cpp el soporte de audio es parcial según versión — si
el audio es imprescindible, usa vLLM. Tip del model card: pon la **imagen antes**
del texto y el **audio después**.

**Agentes / tool-calling.** vLLM con `ENABLE_TOOLS=1` activa function-calling +
thinking (`--tool-call-parser gemma4`, `--reasoning-parser gemma4`); llama.cpp lo
hace con `--jinja` (plantilla embebida en el GGUF).

**Sampling recomendado por Google:** `temperature=1.0`, `top_p=0.95`, `top_k=64`.

---

## Arranque rápido

```bash
cp .env.example .env      # edita HF_TOKEN y ajusta a tu hardware
```

**vLLM en GPU NVIDIA** (imagen fijada `gemma4-unified`). El default ya es un
checkpoint **w4a16 (QAT) de Unsloth, no gated** (`unsloth/gemma-4-12B-it-qat-w4a16`,
~10 GB) que arranca **sin token** en 16–24 GB; vLLM detecta la cuantización solo:

```bash
docker compose --profile cuda up -d
docker compose --profile cuda logs -f
curl http://localhost:8000/v1/models
```

CUDA 12.9 en vez de 13: `VLLM_CUDA_IMAGE=vllm/vllm-openai:gemma4-unified-cu129`.

**llama.cpp en GPU NVIDIA** (más simple, GGUF QAT):

```bash
docker compose --profile llamacpp-cuda up -d
```

**llama.cpp en GPU/iGPU AMD o Intel** (Vulkan):

```bash
docker compose --profile llamacpp up -d
```

**CPU** (pruebas, lento):

```bash
docker compose --profile cpu up -d
```

**Apple Silicon (nativo, sin Docker):**

```bash
brew install llama.cpp
llama-server \
  -hf unsloth/gemma-4-12B-it-qat-GGUF:UD-Q4_K_XL \
  --alias gemma-4-12b \
  -c 65536 --parallel 1 \      # sube/ajusta según tu RAM unificada
  -ngl 999 --jinja \
  --temp 1.0 --top-p 0.95 --top-k 64 \
  --host 0.0.0.0 --port 8000
# Alternativa MLX:
# pip install mlx-lm && mlx_lm.server --model mlx-community/gemma-4-12B-it --port 8000
```

> **Token:** con los **defaults (Unsloth, no gated) no necesitas `HF_TOKEN`**.
> Solo hace falta si cambias `MODEL` al repo **oficial de Google** (gated):
> acepta la licencia en <https://huggingface.co/google/gemma-4-12B-it> y pon el
> token en `.env`. Variantes vLLM: bf16 (40 GB+) `unsloth/gemma-4-12b-it`;
> oficial gated `google/gemma-4-12B-it`.

---

## Usar el modelo

Endpoint común: `http://localhost:8000/v1`, modelo `gemma-4-12b`.

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

Prueba texto/imagen/audio: `python test_api.py --image URL` / `--audio clip.wav`.

Para conectar un agente (Continue, Aider, OpenCode, etc.):
`OPENAI_BASE_URL=http://localhost:8000/v1`, `OPENAI_MODEL=gemma-4-12b`,
`OPENAI_API_KEY=` (cualquier valor, o el de `VLLM_API_KEY` si lo defines).

---

## Configuración (`.env`)

**vLLM:**

| Variable | Default | Qué hace |
|---|---|---|
| `MODEL` / `QUANTIZATION` | `unsloth/gemma-4-12B-it-qat-w4a16` / — | w4a16 QAT no gated (sin token); vLLM auto-detecta la cuantización. bf16: `unsloth/gemma-4-12b-it`. |
| `MAX_MODEL_LEN` | `200000` | Contexto (máx 262144 / 256K). |
| `MAX_NUM_SEQS` | `3` | Clientes concurrentes. |
| `KV_CACHE_DTYPE` | — | `fp8` ≈ −50% KV cache. |
| `GPU_MEMORY_UTILIZATION` | `0.90` | VRAM para modelo + KV (0.85–0.95). |
| `TENSOR_PARALLEL_SIZE` | `1` | GPUs para repartir el modelo. |
| `LIMIT_MM_PER_PROMPT` | `{"image":4,"audio":1}` | Máx. imágenes/audios. `{"image":0,"audio":0}` = solo texto. |
| `VISION_MAX_SOFT_TOKENS` | `280` | Detalle visual: 70/140/280/560/1120. |
| `ENABLE_TOOLS` | `1` | Tool-calling + reasoning. |
| `VLLM_API_KEY` | — | Protege el endpoint (Bearer). |

**llama.cpp:**

| Variable | Default | Qué hace |
|---|---|---|
| `GGUF_HF_REPO` / `GGUF_QUANT` | `unsloth/gemma-4-12B-it-qat-GGUF` / `UD-Q4_K_XL` | Modelo GGUF (QAT). |
| `LLAMACPP_CTX` | `262144` | Contexto **total** (se reparte entre `--parallel`). |
| `MAX_NUM_SEQS` | `3` | Ranuras `--parallel`. |
| `LLAMACPP_EXTRA` | — | Flags extra (p.ej. `--mmproj`, draft MTP). |

---

---

## Troubleshooting

**`set: pipefail: invalid option name` (bucle de reinicios).** Fin de línea
CRLF en `entrypoint.sh` (típico en Windows). Ya está mitigado (el compose limpia
el CR en runtime y `.gitattributes` fuerza LF). Haz `git pull` y
`docker compose --profile <perfil> up -d --force-recreate`.

**`'Gemma4UnifiedVisionConfig' object has no attribute 'num_soft_tokens'`
(vLLM, perfil `cuda`).** El checkpoint **w4a16** de Unsloth omite ese campo en su
`config.json`. El entrypoint lo **parchea automáticamente** (`PATCH_VISION_SOFT_TOKENS=auto`,
descarga el snapshot y añade `num_soft_tokens=280`). Si prefieres, ponlo en
`off` y corre **text-only** (`LIMIT_MM_PER_PROMPT={"image":0,"audio":0}`), o usa
el perfil **`llamacpp-cuda`** (GGUF QAT + mmproj), que no tiene este problema y
encaja mejor en 16 GB para multimodal.

**`There is no module or parameter named 'vision_embedder.patch_dense.weight'`
(vLLM, perfil `cuda`).** Segundo bug de empaquetado del checkpoint **w4a16** de
Unsloth: la capa del embebedor de visión quedó sin cuantizar (en la lista
`ignore`) pero vLLM la trata como cuantizada. **No se puede arreglar desde el
entrypoint** (es a nivel de pesos). Hoy por hoy **no hay checkpoint w4a16
funcional para vLLM**, y el bf16 (~24 GB) no entra en 16 GB. **En GPU de consumo
(16–24 GB) usa el perfil `llamacpp-cuda`** (GGUF QAT, sin estos bugs); reserva
vLLM para GPU de 40 GB+ con bf16 (`unsloth/gemma-4-12b-it`) o hasta que Unsloth
corrija el w4a16.

**OOM al cargar / KV cache.** Baja `MAX_MODEL_LEN` (p.ej. 65536), reduce
`MAX_NUM_SEQS`, mantén `KV_CACHE_DTYPE=fp8`, o usa un quant más pequeño.

**Lento / "WSL detected, pin_memory=False".** Es normal en Docker Desktop sobre
WSL2; para máximo rendimiento conviene Linux nativo con NVIDIA Container Toolkit.

## Fuentes

- GGUF QAT (Unsloth): <https://huggingface.co/unsloth/gemma-4-12B-it-qat-GGUF> · GGUF estándar + MTP: <https://huggingface.co/unsloth/gemma-4-12b-it-GGUF>
- Receta vLLM (12B): <https://recipes.vllm.ai/Google/gemma-4-12B-it> · familia Gemma 4: <https://docs.vllm.ai/projects/recipes/en/latest/Google/Gemma4.html>
- Model card / pesos: <https://huggingface.co/google/gemma-4-12B-it>
- Anuncio: <https://blog.google/innovation-and-ai/technology/developers-tools/introducing-gemma-4-12b/>
