# llama.cpp — VITRIOL Fork

> **Branch note:** This branch contains the README and licensing information.
> VITRIOL-specific code modifications are maintained on the
> [`vitriol`](https://github.com/Randozart/llama.cpp/tree/vitriol) branch.
> See below for a full inventory of changed files.


# llama.cpp — VITRIOL Branch

A fork of [llama.cpp](https://github.com/ggml-org/llama.cpp) with VITRIOL's VRAM extension layer for running large MoE models on VRAM-limited GPUs, plus TurboQuant (`TQ3_1S` / `TQ3_4S`) runtime support.

---

## Why This Fork Exists

Standard llama.cpp requires weights to fit in VRAM for GPU inference. On older GPUs with 4-8 GB VRAM (GTX 1070 Ti, GTX 960, etc.), modern MoE models like Qwen3.6-35B don't fit at all — `cudaMalloc` fails with out-of-memory at `-ngl 99`.

This fork adds an alternative path: **keep expert weights in page-locked system RAM, stream them to the GPU over PCIe DMA on demand.** The GPU only needs VRAM for the base model weights, KV cache, and compute buffers — typically 1.5-2.5 GiB vs the ~12 GiB the full model would need.

The approach is documented in full at [github.com/Randozart/VITRIOL](https://github.com/Randozart/VITRIOL).

---

## Performance

**Qwen3.6-35B-A3B-UD-IQ2_M** on GTX 1070 Ti (8 GB VRAM, PCIe 3.0 x16, 15 GB DDR3):

| Config | Gen (tok/s) | vs baseline | VRAM used |
|--------|------------|-------------|-----------|
| x8 PCIe (no VITRIOL) | 5.7 | — | OOM |
| x16 PCIe (no VITRIOL) | 8.9 | +56% | OOM |
| + VITRIOL DMA + pin 8 | 12.8 | +125% | ~1.6 GiB |
| **+ Chimera dual-backend + MTP** | **25.2** | **+342%** | ~2.5 GiB |

Native context: **262,144 tokens** at 17.3 tok/s (with pin pool disabled).

---

## Key Features

### RAM Shot (Page-Locked Host DMA)

The core VITRIOL trick: expert weights stay in page-locked system RAM (`mmap` + `mlock` + `cudaHostRegister`). The GPU reads them over PCIe DMA when the MoE router selects them. Zero VRAM consumed for weight storage.

**Files:** `ggml/src/ggml-cuda/vitriol-buffer.{cpp,h}`, `vitriol-cuda-integration.{cpp,h}`

### Chimera Dual-Backend (CUDA + Vulkan)

Routes each operation to the optimal GPU backend:
- **MoE expert matmuls** → CUDA VITRIOL DMA (page-locked host RAM + pin pool + predictor)
- **SSM scan, attention, norms** → Vulkan (pre-baked command buffers, `VK_EXT_external_memory_host`)
- Cross-backend activation copies are automatic via the scheduler (~0.13% overhead, thread-local staging buffer eliminates `malloc`/`free`)

Auto-detected via `VITRIOL_CHIMERA_MODE=auto` — `dlsym` checks for `vitriol_get_vk_buffer_type` in `libggml-vulkan.so`. If found, Chimera activates. No user config needed.

**Config modes:** `auto` (default) | `cuda` | `vulkan` | `off`

**Files:** `ggml/src/ggml-vulkan/vitriol-vk-buffer.{cpp,h}`, `ggml/src/ggml-vulkan/ggml-vulkan.cpp`

### Mamba-1 Vulkan SSM Shader

The upstream Vulkan backend only supported Mamba-2 SSM scan (d_state 128/256). This fork adds a Mamba-1 GLSL compute shader (d_state=16, head_dim=1, 128 threads/workgroup) for Qwen3's Gated Delta Net and Jamba2.

Previously, the Vulkan backend `supports_op` rejected Mamba-1 entirely (`if (!is_mamba2) return false`). Now both Mamba-1 and Mamba-2 are supported on Vulkan.

**File:** `ggml/src/ggml-vulkan/vulkan-shaders/ssm_scan_mamba1.comp`
**Upstream PR reference:** [#16463](https://github.com/ggerganov/llama.cpp/pull/16463) (giuseppe) — added Mamba-2 Vulkan support; our shader extends this to Mamba-1.

### Expert Pinning

Static VRAM pool for caching the first N layers' expert tensors in VRAM. Configured via `VITRIOL_PIN_FIRST_N_LAYERS=N` (default 8 for Qwen3.6). Tensors beyond the pool are read from host RAM over DMA.

**File:** `ggml/src/ggml-cuda/vitriol-cuda-integration.cpp`

### Predictive Prefetch

Cross-layer expert prefetching: while the current token computes, the system predicts which experts the next token will need and begins their DMA transfer in the background. Overlaps PCIe transfer with GPU compute.

**File:** `ggml/src/ggml-cuda/vitriol-cuda-integration.cpp`

### Auto-Detect Backend Routing

`VITRIOL_CHIMERA_MODE` env var controls routing:
| Mode | Behavior |
|---|---|
| `auto` | If `dlsym` finds Vulkan buffer type → Chimera; else CUDA-only |
| `cuda` | CUDA-only (standard VITRIOL) |
| `vulkan` | Vulkan-only (all tensors via `VK_EXT_external_memory_host`) |
| `off` | CUDA-only (same as `cuda`) |

### K/V Cache Quantization

Separate quantization for K and V caches. K can be `q4_0` (recommended) or `q8_0`. **V must stay `f16`** — quantizing V produces garbage output with VITRIOL's expert buffer type (see EXPERIMENT_LOG.md Experiment 17).

---

## What's Different From Upstream

### New Files (VITRIOL-specific)

| File | Purpose |
|---|---|
| `ggml/src/ggml-cuda/vitriol-buffer.cpp` | RAM Shot buffer type — page-locked host RAM for expert weights |
| `ggml/src/ggml-cuda/vitriol-buffer.h` | Buffer type interface + context struct |
| `ggml/src/ggml-cuda/vitriol-cuda-integration.cpp` | Pin pool, predictor, config, LRU cache, Chimera VK buffer dispatch |
| `ggml/src/ggml-cuda/vitriol-cuda-integration.h` | Public API for CUDA integration |
| `ggml/src/ggml-cuda/vitriol_copy_engine.cpp` | Copy Engine DMA (standalone) |
| `ggml/src/ggml-cuda/vitriol_copy_engine.h` | Copy Engine interface |
| `ggml/src/ggml-vulkan/vitriol-vk-buffer.cpp` | VITRIOL VK buffer type — `VK_EXT_external_memory_host` import |
| `ggml/src/ggml-vulkan/vitriol-vk-buffer.h` | VK buffer interface + context struct |
| `ggml/src/ggml-vulkan/vulkan-shaders/ssm_scan_mamba1.comp` | Mamba-1 SSM scan GLSL shader |

### Modified Files

| File | Changes |
|---|---|
| `ggml/src/ggml-cuda/ggml-cuda.cu` | `supports_buft` for VITRIOL type, pin pool hooks in `ggml_cuda_mul_mat_id` |
| `ggml/src/ggml-vulkan/ggml-vulkan.cpp` | Pipeline + dispatch + `supports_op` for Mamba-1; `supports_buft` for VITRIOL VK type; `tensor_subbuffer` VITRIOL VK context handling |
| `ggml/src/ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp` | Register Mamba-1 shader for SPIR-V compilation |
| `src/llama-model-loader.cpp` | VITRIOL tensor interception (expert → CUDA type, dense → VK type when Chimera active) |
| `ggml/src/ggml-backend.cpp` | Thread-local staging buffer for cross-backend tensor copies (eliminates `malloc`/`free` in hot path) |
| `ggml/src/ggml-backend-meta.cpp` | `try_emplace` → `emplace`, `explicit` constructors, null checks |
| `ggml/src/ggml.c` | Null checks in `is_contiguous_rows` and `are_same_shape` |
| `ggml/src/gguf.cpp` | `explicit` constructors, `string_view`, range-based loops, virtual destructor |
| `ggml/src/ggml-quants.c` | Flattened nested ternaries, `f` suffix optimization, arithmetic replacement |

---

## Personal Modifications

This fork reflects a specific design philosophy: **the GPU should not be fed; it should eat.**

Rather than fitting weights into VRAM (the conventional approach), we let the GPU pull weights from system RAM over PCIe on demand. This inverts the traditional memory hierarchy for inference — system RAM becomes primary storage, VRAM becomes cache.

Key design decisions:
- **Buffer type trick:** VITRIOL's custom `ggml_backend_buffer_type` reports `is_host=true` but is accepted by the CUDA backend via a stolen `get_name` function pointer. The scheduler routes `MUL_MAT_ID` to CUDA, which reads `src0->data` from page-locked host RAM over PCIe DMA transparently.
- **Chimera hybrid:** Rather than choosing one backend, route each op to its optimal backend — CUDA for MoE (sparse expert access), Vulkan for dense ops (pre-baked command buffers). Cross-backend copies are automatic via CPU staging (~0.13% overhead).
- **Thread-local over thread-global:** The staging buffer for cross-backend copies uses `thread_local std::vector` instead of `malloc`/`free` per copy — eliminating ~1300 heap alloc/free calls per second at 25 tok/s.
- **No speculative draft model:** Unlike most speculative decoding implementations, this fork uses MTP (Multi-Token Prediction) heads built into the model itself — no second model needed, no VRAM overhead for a draft model.

---

## TurboQuant Runtime

This fork also retains the upstream TurboQuant runtime path for `TQ3_1S` and `TQ3_4S` quantized models. See the relevant sections below for building and running TQ3 models.

### Building From Source

```bash
git clone --branch vitriol https://github.com/Randozart/llama.cpp.git
cd llama.cpp

# With VITRIOL + CUDA:
cmake -B build -DGGML_CUDA=ON -DGGML_NATIVE=ON -DGGML_VULKAN=ON
cmake --build build -j$(nproc)

# TQ3-only (no Vulkan):
cmake -B build -DGGML_CUDA=ON -DGGML_NATIVE=ON
cmake --build build -j$(nproc)
```

### Download A Model

VITRIOL-supported models (IQ2_M, MTP-capable):

```bash
huggingface-cli download unsloth/Qwen3.6-35B-A3B-MTP-GGUF \
  Qwen3.6-35B-A3B-UD-IQ2_M.gguf \
  --local-dir ./models
```

TQ3 models:

```bash
huggingface-cli download YTan2000/Qwen3.5-27B-TQ3_4S \
  Qwen_Qwen3.5-27B-TQ3_4S.gguf \
  --local-dir ./models/tq3_4s
```

### Running (VITRIOL Chimera)

```bash
env \
  CUDA_VISIBLE_DEVICES=0 \
  VITRIOL_MODE=stream \
  VITRIOL_ENGINE_MODE=vitriol-dma \
  VITRIOL_PIN_FIRST_N_LAYERS=8 \
  ./build/bin/llama-server \
    -m ./models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf \
    -ngl 99 -c 8192 --host 0.0.0.0 --port 8279 \
    --parallel 1 -t 4 -fa on \
    --cache-type-k q4_0 --no-mmap \
    --checkpoint-every-n-tokens 4096 \
    --spec-type mtp --spec-draft-n-max 2
```

### Running (TQ3)

```bash
./build/bin/llama-server \
  -m ./models/tq3_4s/Qwen_Qwen3.5-27B-TQ3_4S.gguf \
  -ngl 99 -fa on -c 8192 --port 8090 \
  -ctk tq3_0 -ctv tq3_0
```

---

## Supported Hardware

| GPU | VRAM | Status | Notes |
|-----|------|--------|-------|
| GTX 1070 Ti | 8 GB | ✅ Verified | 25.2 tok/s (Chimera, IQ2_M, MTP N=2) |
| RTX 3060 | 12 GB | ✅ Supported | Larger KV cache |
| RTX 4090 | 24 GB | ✅ Supported | PCIe 4.0 → higher bandwidth |
| AMD RX 7000 | varies | ✅ Chimera (Vulkan) | Dense ops on Vulkan; MoE via CUDA not available |
| Intel Arc | varies | ✅ Chimera (Vulkan) | Via `VK_EXT_external_memory_host` |

---

## Acknowledgements

- **[ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)** — The core inference engine. GGUF format, CUDA backend, tensor loading pipeline.
- **[slaren](https://github.com/ggerganov/llama.cpp/pull/11397)** — `--override-tensor` flag, the breakthrough that enabled expert streaming.
- **[giuseppe](https://github.com/ggerganov/llama.cpp/pull/16463)** — Vulkan SSM operations (Mamba-2).
- **[fairydreaming](https://github.com/ggerganov/llama.cpp/pull/11571)** — Load-all-experts-during-warmup API.
- **llama.cpp PR [#22673](https://github.com/ggml-org/llama.cpp/pull/22673)** — MTP (Multi-Token Prediction) speculative decoding implementation.
- **[TheTom/turboquant_plus](https://github.com/TheTom/turboquant_plus)** — TurboQuant KV-cache engineering, benchmarking, and implementation.
- **[flamme-demon/llama.cpp-hip-turboquant-tq3](https://github.com/flamme-demon/llama.cpp-hip-turboquant-tq3)** — HIP/ROCm compatibility port of TQ3 native kernels.
- **Google Research** — [TurboQuant](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/) compression algorithm.
- **[KTransformers](https://github.com/kvcache-ai/KTransformers)** — YAML-based layer placement, double-buffer prefetch pattern.
- **[PowerInfer](https://github.com/SJTU-IPADS/PowerInfer)** — Neuron-level offloading with predictor.
- **[LLM in a Flash](https://arxiv.org/abs/2312.11514)** (Apple, 2023) — Proved windowing + zero-copy streaming from host memory enables inference on memory-limited hardware. Foundation of the RAM Shot approach.
- **[Unsloth](https://huggingface.co/unsloth)** — Dynamic quantization formats (UD-Q2_K_XL, IQ2_M). The Qwen3.6 models we target were quantized and distributed by them.

---

## License

This fork inherits the MIT license from upstream llama.cpp. See `LICENSE` for details.
