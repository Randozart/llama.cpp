#include "vitriol-cuda-integration.h"
#include "vitriol-buffer.h"
#include "ggml-cuda.h"
#include "ggml-backend.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unordered_map>
#include <list>
#include <mutex>
#include <cuda.h>
#include <cuda_runtime.h>
#include <sys/mman.h>

vitriol_config_t g_vitriol_config;

static struct vitriol_config_init {
    vitriol_config_init() {
        memset(&g_vitriol_config, 0, sizeof(g_vitriol_config));
        g_vitriol_config.mode = VITRIOL_MODE_DISABLED;
        g_vitriol_config.prefetch_ahead = 2;
        g_vitriol_config.static_layers = 15;
        g_vitriol_config.window_size_mb = 2048;
        g_vitriol_config.use_double_buffer = true;
        g_vitriol_config.buffer_count = 2;
        g_vitriol_config.pin_tensors_per_layer = 2;
    }
} s_vitriol_config_init;

/* ── LRU Cache ──────────────────────────────────────────────────── */

#define VITRIOL_LRU_POOL_SIZE  (2048ULL * 1024 * 1024)  // default 2 GB VRAM pool (env VITRIOL_LRU_MB overrides)
#define VITRIOL_LRU_MAX_SLOTS  65536

/* Everything below is per-CUDA-device. With a layer split across two GPUs
 * (e.g. GTX 1070 Ti + RTX 3060), each device owns the expert tensors of its
 * layers, so each device needs its own LRU pool, DMA stream and slot events.
 * A single global pool/stream would route GPU1's expert DMA into GPU0's VRAM
 * and hand GPU1 kernels a device-0 pointer → invalid memory access. */
static CUdeviceptr g_lru_pool[GGML_CUDA_MAX_DEVICES];
static size_t      g_lru_pool_size[GGML_CUDA_MAX_DEVICES];
static size_t      g_lru_slot_size[GGML_CUDA_MAX_DEVICES];
static int         g_lru_num_slots[GGML_CUDA_MAX_DEVICES];

/* Composite key: (device, tensor_base_address, expert_idx) prevents
 * cross-layer collisions where expert 0 of layer 1 != expert 0 of layer 2,
 * and keeps cache entries from different GPUs fully disjoint. */
struct LRUKey {
    int       device;
    uintptr_t tensor_base;
    int       expert_idx;
    bool operator==(const LRUKey &o) const {
        return device == o.device && tensor_base == o.tensor_base && expert_idx == o.expert_idx;
    }
};

struct LRUKeyHash {
    size_t operator()(const LRUKey &k) const {
        return (size_t)(k.tensor_base * 2654435761U) ^ (size_t)k.expert_idx ^ (size_t)k.device;
    }
};

static std::unordered_map<LRUKey, int, LRUKeyHash> g_lru_map;
static std::list<LRUKey> g_lru_order;
static std::mutex        g_lru_mtx;

/* Dedicated stream for async DMA; per-slot events for sync.
 * One stream + event array per device. */
static CUstream  g_lru_stream[GGML_CUDA_MAX_DEVICES];
static CUevent *g_lru_slot_events[GGML_CUDA_MAX_DEVICES];
static std::mutex g_lru_init_mtx;

static struct LRUStats {
    unsigned long long hits;
    unsigned long long misses;
    unsigned long long evictions;
} g_lru_stats;

/* ── Expert Pinning Table ──────────────────────────────────────────
 * Maps tensor_base address → deduplicated pinned VRAM buffer.
 * Each entry covers ALL experts of that tensor (contiguous, entire tensor). */
static std::unordered_map<uintptr_t, CUdeviceptr> g_pin_map;
static std::mutex          g_pin_mtx;
static size_t              g_pinned_bytes = 0;
static bool                g_pin_init_done = false;

/* Monolithic VRAM pool per device for all pinned tensors.
 * Allocated once on first pin for a given device; subdivided per tensor. */
static CUdeviceptr g_pin_pool[GGML_CUDA_MAX_DEVICES];
static size_t      g_pin_pool_offset[GGML_CUDA_MAX_DEVICES];
static size_t      g_pin_pool_total[GGML_CUDA_MAX_DEVICES];
static bool        g_pin_pool_failed[GGML_CUDA_MAX_DEVICES];

/* ── Expert Output Cache (approximate) ───────────────────────────── */

#define VITRIOL_MAX_CACHE_EXPERTS_PER_LAYER 256
#define VITRIOL_MAX_CACHE_LAYERS            128

static struct {
    uintptr_t tensor_base;
    int       expert_id;
    float    *data_dev;   // GPU buffer, n_embd floats
    bool      valid;
} g_output_cache[VITRIOL_MAX_CACHE_LAYERS][VITRIOL_MAX_CACHE_EXPERTS_PER_LAYER];

static bool   g_output_cache_initialized = false;
static size_t g_output_cache_n_embd      = 0;
static int    g_output_cache_n_layers    = 0;
static CUdeviceptr g_output_cache_pool[GGML_CUDA_MAX_DEVICES];
static size_t g_output_cache_pool_size[GGML_CUDA_MAX_DEVICES];

static struct {
    unsigned long long hits;
    unsigned long long misses;
} g_output_cache_stats;

static int vitriol_current_device(void);

void vitriol_output_cache_init(int n_layers, size_t n_embd) {
    if (g_output_cache_initialized) return;

    if (n_layers > VITRIOL_MAX_CACHE_LAYERS) n_layers = VITRIOL_MAX_CACHE_LAYERS;

    // Allocate a per-device VRAM pool: n_layers * n_experts * n_embd * sizeof(float)
    size_t entry_size = n_embd * sizeof(float);
    size_t pool_size  = (size_t)n_layers * VITRIOL_MAX_CACHE_EXPERTS_PER_LAYER * entry_size;

    int dev = vitriol_current_device();
    cudaSetDevice(dev);
    CUresult err = cuMemAlloc(&g_output_cache_pool[dev], pool_size);
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "VITRIOL: output cache pool alloc %zu MB failed (%d)\n",
                pool_size / 1024 / 1024, (int)err);
        return;
    }

    // Initialize all entries
    memset(g_output_cache, 0, sizeof(g_output_cache));

    g_output_cache_pool_size[dev] = pool_size;
    g_output_cache_n_embd    = n_embd;
    g_output_cache_n_layers  = n_layers;
    g_output_cache_initialized = true;

    // Device pointers are assigned per entry at store time (the pool is
    // per-device, so a fixed bulk assignment can only describe one GPU).

    if (g_vitriol_config.verbose)
        printf("VITRIOL: output cache initialized: %d layers x %d experts x %zu bytes = %zu MB (GPU %d)\n",
               n_layers, VITRIOL_MAX_CACHE_EXPERTS_PER_LAYER, entry_size,
               pool_size / 1024 / 1024, dev);
}

// Forward declarations
static int get_layer_index(uintptr_t tensor_base);
static void detect_token_boundary(int layer_idx);

const float * vitriol_output_cache_lookup(
    const void *tensor_base,
    int         expert_id)
{
    if (!g_output_cache_initialized) return NULL;

    // Find layer index using existing predictor infrastructure
    int layer_idx = get_layer_index((uintptr_t)tensor_base);
    if (layer_idx < 0 || layer_idx >= g_output_cache_n_layers) return NULL;

    auto & entry = g_output_cache[layer_idx][expert_id];
    if (entry.valid &&
        entry.tensor_base == (uintptr_t)tensor_base &&
        entry.expert_id == expert_id) {
        g_output_cache_stats.hits++;
        return entry.data_dev;
    }

    g_output_cache_stats.misses++;
    return NULL;
}

void vitriol_output_cache_store(
    const void    *tensor_base,
    int            expert_id,
    const float   *output_data,
    size_t         n_embd,
    CUstream       stream)
{
    if (!g_output_cache_initialized) {
        // Lazy init: assume up to 128 layers, infer n_embd from data
        vitriol_output_cache_init(128, n_embd);
        if (!g_output_cache_initialized) return;
    }

    int layer_idx = get_layer_index((uintptr_t)tensor_base);
    if (layer_idx < 0 || layer_idx >= g_output_cache_n_layers) return;

    if (n_embd != g_output_cache_n_embd) return;

    int dev = vitriol_current_device();
    if (g_output_cache_pool[dev] == 0) {
        vitriol_output_cache_init(g_output_cache_n_layers, n_embd);
        if (g_output_cache_pool[dev] == 0) return;
    }

    auto & entry = g_output_cache[layer_idx][expert_id];

    size_t entry_size = n_embd * sizeof(float);
    entry.data_dev = (float *)(g_output_cache_pool[dev] +
        ((size_t)layer_idx * VITRIOL_MAX_CACHE_EXPERTS_PER_LAYER + (size_t)expert_id) * entry_size);
    entry.tensor_base = (uintptr_t)tensor_base;
    entry.expert_id   = expert_id;
    entry.valid       = true;

    // D2D copy from compute buffer to cache slot (same device)
    size_t copy_size = n_embd * sizeof(float);
    CUresult r = cuMemcpyDtoDAsync(
        (CUdeviceptr)entry.data_dev,
        (CUdeviceptr)output_data,
        copy_size,
        stream);
    if (r != CUDA_SUCCESS) {
        entry.valid = false;
    }
}

void vitriol_output_cache_advance_token(void) {
    if (!g_output_cache_initialized) return;

    // On token boundary, invalidate ALL entries from the previous token
    // because outputs are token-dependent
    for (int l = 0; l < g_output_cache_n_layers; l++) {
        for (int e = 0; e < VITRIOL_MAX_CACHE_EXPERTS_PER_LAYER; e++) {
            g_output_cache[l][e].valid = false;
        }
    }
}

void vitriol_output_cache_print_stats(void) {
    uint64_t total = g_output_cache_stats.hits + g_output_cache_stats.misses;
    float hr = (total > 0) ? 100.0f * (float)g_output_cache_stats.hits / (float)total : 0.0f;
    printf("=== VITRIOL Output Cache ===\n");
    printf("Enabled: %s\n", g_vitriol_config.output_cache ? "yes" : "no");
    printf("Initialized: %s\n", g_output_cache_initialized ? "yes" : "no");
    printf("Layers: %d, n_embd: %zu\n", g_output_cache_n_layers, g_output_cache_n_embd);
    printf("Hits: %llu\n", g_output_cache_stats.hits);
    printf("Misses: %llu\n", g_output_cache_stats.misses);
    printf("Hit Rate: %.2f%%\n", hr);
    printf("==============================\n");
}

static bool lru_init_pool(size_t min_expert_size, int device);
static bool lru_ensure_stream(int device);
/* Forward decl so the output cache (early in the file) can resolve the
 * current device before the helper's definition below. */
static int vitriol_current_device(void);

/* Current CUDA device index for this thread, clamped to a sane range.
 * The VITRIOL paths are entered from ggml-cuda kernels running on whatever
 * device owns the layer being computed — exactly the device whose LRU pool /
 * streams / events must be used. Self-contained (cudaGetDevice), no dependency
 * on ggml internals. */
static int vitriol_current_device(void) {
    int dev = 0;
    cudaGetDevice(&dev);
    if (dev < 0) dev = 0;
    if (dev >= GGML_CUDA_MAX_DEVICES) dev = GGML_CUDA_MAX_DEVICES - 1;
    return dev;
}

/* ── Initialization ──────────────────────────────────────────────── */

static void vitriol_cuda_cleanup_vram(void);

void vitriol_cuda_init(void) {
    const char* mode_env = getenv("VITRIOL_MODE");
    if (mode_env) {
        if (strcmp(mode_env, "off") == 0 || strcmp(mode_env, "disabled") == 0)
            g_vitriol_config.mode = VITRIOL_MODE_DISABLED;
        else if (strcmp(mode_env, "sync") == 0)
            g_vitriol_config.mode = VITRIOL_MODE_SYNC;
        else if (strcmp(mode_env, "async") == 0)
            g_vitriol_config.mode = VITRIOL_MODE_ASYNC;
        else if (strcmp(mode_env, "stream") == 0)
            g_vitriol_config.mode = VITRIOL_MODE_STREAM;
    }

    const char* verbose_env = getenv("VITRIOL_VERBOSE");
    if (verbose_env && strcmp(verbose_env, "1") == 0)
        g_vitriol_config.verbose = true;

    const char* pf_env = getenv("VITRIOL_PREDICTIVE_PREFETCH");
    if (pf_env && strcmp(pf_env, "1") == 0)
        g_vitriol_config.async_prefetch = true;

    const char* disk_env = getenv("VITRIOL_DISK_OFFLOAD");
    if (disk_env && strcmp(disk_env, "1") == 0)
        g_vitriol_config.disk_offload = true;

    const char* oc_env = getenv("VITRIOL_OUTPUT_CACHE");
    if (oc_env && strcmp(oc_env, "1") == 0)
        g_vitriol_config.output_cache = true;

    const char* pin_env = getenv("VITRIOL_PIN_FIRST_N_LAYERS");
    if (pin_env) {
        int val = atoi(pin_env);
        if (val > 0) g_vitriol_config.pin_first_n_layers = val;
    }

    const char* prune_env = getenv("VITRIOL_PRUNE_EXPERTS");
    if (prune_env) {
        int val = atoi(prune_env);
        if (val > 0 && val <= 7) g_vitriol_config.prune_experts = val;
    }

    const char* locked_env = getenv("VITRIOL_MAX_LOCKED_MB");
    if (locked_env) {
        int val = atoi(locked_env);
        if (val > 0) g_vitriol_config.max_locked_mb = val;
    }

    if (g_vitriol_config.mode == VITRIOL_MODE_STREAM) {
        if (g_vitriol_config.verbose)
            printf("VITRIOL: stream mode — page-locked host RAM + LRU VRAM cache\n");
        if (g_vitriol_config.max_locked_mb > 0) {
            if (g_vitriol_config.verbose)
                printf("VITRIOL: lazy page locking — max %d MB, per-expert on demand\n",
                       g_vitriol_config.max_locked_mb);
        }
    }

    if (g_vitriol_config.async_prefetch) {
        if (g_vitriol_config.verbose)
            printf("VITRIOL: predictive prefetching enabled (cross-layer + temporal)\n");
    }

    if (g_vitriol_config.pin_first_n_layers > 0) {
        if (g_vitriol_config.verbose)
            printf("VITRIOL: expert pinning enabled — first %d layers will be loaded into VRAM\n",
                   g_vitriol_config.pin_first_n_layers);
        /* Pinning targets the fast path; disable output cache to avoid confusion */
        if (g_vitriol_config.output_cache) {
            g_vitriol_config.output_cache = false;
            if (g_vitriol_config.verbose)
                printf("VITRIOL: output cache disabled (pinning + output cache target different decode paths)\n");
        }
    }

    if (g_vitriol_config.disk_offload) {
        if (g_vitriol_config.verbose)
            printf("VITRIOL: disk offload mode — file-backed mmap, no page-lock\n");
    }

    static bool first_init = true;
    if (first_init) {
        memset(&g_lru_stats, 0, sizeof(g_lru_stats));
        first_init = false;

        /* Register atexit handlers */
        atexit(vitriol_cuda_cleanup_vram);
        atexit(vitriol_cuda_print_stats);
    }
}

/* ── Locked-Slice LRU (Lazy/Chunked Page Locking) ──────────────
 * Tracks which (tensor_base, expert_idx) pairs are currently
 * page-locked via mlock + cudaHostRegister. Keeps total locked
 * bytes below g_vitriol_config.max_locked_mb. Evicts by LRU.
 * ───────────────────────────────────────────────────────────────*/

#define VITRIOL_MAX_LOCKED_SLICES 1024

struct locked_slice {
    const void * tensor_base;
    int          expert_idx;
    size_t       size;
    uint64_t     last_used;  // timestamp for LRU eviction
};

static std::mutex               g_locked_mtx;
static locked_slice             g_locked_slices[VITRIOL_MAX_LOCKED_SLICES];
static int                      g_n_locked_slices = 0;
static uint64_t                 g_locked_bytes    = 0;
static uint64_t                 g_locked_epoch    = 1;

void vitriol_ensure_expert_locked(const void * tensor_base, int expert_idx, size_t expert_size) {
    if (!vitriol_lazy_lock_active()) return;
    if (expert_size == 0) return;

    std::lock_guard<std::mutex> lock(g_locked_mtx);

    /* Check if already locked */
    for (int i = 0; i < g_n_locked_slices; i++) {
        if (g_locked_slices[i].tensor_base == tensor_base &&
            g_locked_slices[i].expert_idx == expert_idx) {
            g_locked_slices[i].last_used = g_locked_epoch++;
            return;
        }
    }

    /* Evict if at budget limit */
    const uint64_t max_bytes = (uint64_t)g_vitriol_config.max_locked_mb * 1024 * 1024;
    while (g_locked_bytes + expert_size > max_bytes && g_n_locked_slices > 0) {
        /* Find LRU victim */
        int victim = 0;
        uint64_t oldest = g_locked_slices[0].last_used;
        for (int i = 1; i < g_n_locked_slices; i++) {
            if (g_locked_slices[i].last_used < oldest) {
                oldest = g_locked_slices[i].last_used;
                victim = i;
            }
        }

        const locked_slice & s = g_locked_slices[victim];
        const void * addr = (const char *)s.tensor_base + (ptrdiff_t)s.expert_idx * (ptrdiff_t)s.size;

        cudaError_t err = cudaHostUnregister((void *)addr);
        if (err != cudaSuccess && g_vitriol_config.verbose) {
            fprintf(stderr, "VITRIOL: cudaHostUnregister evict failed: %s\n", cudaGetErrorString(err));
        }
        munlock(addr, s.size);

        g_locked_bytes -= s.size;
        g_n_locked_slices--;

        /* Swap last slot into victim position */
        if (victim < g_n_locked_slices) {
            g_locked_slices[victim] = g_locked_slices[g_n_locked_slices];
        }
    }

    /* Lock this slice */
    const void * addr = (const char *)tensor_base + (ptrdiff_t)expert_idx * (ptrdiff_t)expert_size;

    if (mlock(addr, expert_size) != 0) {
        static bool warned = false;
        if (!warned) {
            fprintf(stderr, "VITRIOL: mlock slice(%zu) failed: %m — subsequent failures suppressed\n", expert_size);
            warned = true;
        }
        return;
    }

    cudaError_t err = cudaHostRegister((void *)addr, expert_size, 0);
    if (err != cudaSuccess) {
        static bool warned = false;
        if (!warned) {
            fprintf(stderr, "VITRIOL: cudaHostRegister slice(%zu) failed: %s — subsequent failures suppressed\n",
                    expert_size, cudaGetErrorString(err));
            warned = true;
        }
        munlock(addr, expert_size);
        return;
    }

    /* Record */
    int slot = g_n_locked_slices++;
    g_locked_slices[slot].tensor_base = tensor_base;
    g_locked_slices[slot].expert_idx  = expert_idx;
    g_locked_slices[slot].size        = expert_size;
    g_locked_slices[slot].last_used   = g_locked_epoch++;
    g_locked_bytes += expert_size;

    if (g_vitriol_config.verbose) {
        printf("VITRIOL: locked slice (tensor=%p, expert=%d, size=%zu, total_locked=%lu MiB)\n",
               (const void *)tensor_base, expert_idx, expert_size,
               (unsigned long)(g_locked_bytes / 1024 / 1024));
    }
}

/* Fallback predictor state (used when per-layer tracking hasn't stabilized) */
static std::mutex            g_pred_mtx;
static int                   g_pred_experts[256];
static int                    g_pred_count;

/* ── Temporal + Cross-Layer Tracking ─────────────────────────────────
 * Per-layer expert selection history across tokens.
 * Combined predictor: prefetch union of
 *   (a) cross-layer:  cur[this_layer - 1]  (experts from previous layer, same token)
 *   (b) temporal:     prev[this_layer]      (experts from same layer, previous token)
 *
 * Token boundary is detected when a previously-seen tensor_base reappears
 * at an index lower than the highest index seen so far (i.e., sequential
 * layer processing wraps back to layer 0).
 */
#define VITRIOL_MAX_LAYERS 128

/* Map tensor_base address → sequential layer index (0, 1, 2, ...) */
static uintptr_t g_layer_bases[VITRIOL_MAX_LAYERS];
static int       g_n_layers = 0;   // total distinct layers seen
static int       g_last_layer = -1; // last layer index seen

/* Current token */
static int  g_cur_exp[VITRIOL_MAX_LAYERS][256];
static int  g_cur_cnt[VITRIOL_MAX_LAYERS];

/* Previous token (boundary: wrap detected) */
static int  g_prev_exp[VITRIOL_MAX_LAYERS][256];
static int  g_prev_cnt[VITRIOL_MAX_LAYERS];

/* ── Layer Index Map ──────────────────────────────────────────────── */

/* Resolve tensor_base → sequential layer index (or -1 if at capacity).
 * This is NOT a direct hardware-layer mapping — it's an ordinal index
 * in order of first appearance (layer 0 → 0, layer 1 → 1, ...), which
 * matches sequential layer execution in autoregressive decode. */
static int get_layer_index(uintptr_t tensor_base) {
    for (int i = 0; i < g_n_layers && i < VITRIOL_MAX_LAYERS; i++) {
        if (g_layer_bases[i] == tensor_base) return i;
    }
    if (g_n_layers >= VITRIOL_MAX_LAYERS) return -1;
    g_layer_bases[g_n_layers] = tensor_base;
    return g_n_layers++;
}

/* Check for token boundary: when layer index wrapped back to an earlier
 * value (e.g., 0 after 39), the current token's per-layer data becomes
 * the "previous token" for the next call. */
static void detect_token_boundary(int layer_idx) {
    if (g_last_layer >= 0 && layer_idx <= g_last_layer) {
        // Swap cur → prev for all layers
        for (int i = 0; i < VITRIOL_MAX_LAYERS; i++) {
            g_prev_cnt[i] = g_cur_cnt[i];
            memcpy(g_prev_exp[i], g_cur_exp[i], g_cur_cnt[i] * sizeof(int));
            g_cur_cnt[i] = 0;
        }
    }
    g_last_layer = layer_idx;
}

void vitriol_predictor_prefetch(
    const void    *tensor_base,
    size_t         expert_size,
    CUstream       compute_stream)
{
    if (!g_vitriol_config.async_prefetch)
        return;

    if (!lru_ensure_stream(vitriol_current_device()))
        return;

    int layer_idx = get_layer_index((uintptr_t)tensor_base);
    if (layer_idx < 0) return;

    detect_token_boundary(layer_idx);

    /* Collect predicted expert set (union of cross-layer + temporal) */
    int predicted[256];
    int n_predicted = 0;

    auto add_prediction = [&](int e) {
        if (e < 0) return;
        for (int i = 0; i < n_predicted; i++)
            if (predicted[i] == e) return;
        predicted[n_predicted++] = e;
    };

    /* (a) Cross-layer: experts from the previous layer of current token */
    if (layer_idx > 0) {
        for (int i = 0; i < g_cur_cnt[layer_idx - 1]; i++)
            add_prediction(g_cur_exp[layer_idx - 1][i]);
    }

    /* (b) Temporal: experts from the same layer of previous token */
    for (int i = 0; i < g_prev_cnt[layer_idx]; i++)
        add_prediction(g_prev_exp[layer_idx][i]);

    if (n_predicted == 0)
        return;

    /* Also fall through to the old g_pred_experts from the last call
     * (for pre-existing behavior on the very first few layers before
     *  per-layer tracking builds up). */
    {
        std::lock_guard<std::mutex> lock(g_pred_mtx);
        for (int i = 0; i < g_pred_count && n_predicted < 256; i++)
            add_prediction(g_pred_experts[i]);
    }

    /* Submit async prefetches for the combined prediction set */
    for (int i = 0; i < n_predicted; i++) {
        int e = predicted[i];
        const void *expert_data = (const char *)tensor_base + (size_t)e * expert_size;
        vitriol_lru_prefetch(tensor_base, e, expert_data, expert_size, compute_stream);
    }
}

void vitriol_predictor_update(
    const void    *tensor_base,
    size_t         expert_size,
    const int32_t *expert_ids,
    int            n_experts)
{
    (void)tensor_base;
    (void)expert_size;

    if (!g_vitriol_config.async_prefetch)
        return;

    int layer_idx = get_layer_index((uintptr_t)tensor_base);
    if (layer_idx < 0) {
        /* Fallback: store in old flat buffer */
        std::lock_guard<std::mutex> lock(g_pred_mtx);
        g_pred_count = 0;
        for (int i = 0; i < n_experts; i++) {
            int e = expert_ids[i];
            if (e < 0) continue;
            bool dup = false;
            for (int j = 0; j < g_pred_count; j++)
                if (g_pred_experts[j] == e) { dup = true; break; }
            if (dup) continue;
            g_pred_experts[g_pred_count++] = e;
        }
        return;
    }

    detect_token_boundary(layer_idx);

    /* Store current layer's expert indices (deduplicated) */
    int count = 0;
    for (int i = 0; i < n_experts && count < 256; i++) {
        int e = expert_ids[i];
        if (e < 0) continue;
        bool dup = false;
        for (int j = 0; j < count; j++)
            if (g_cur_exp[layer_idx][j] == e) { dup = true; break; }
        if (dup) continue;
        g_cur_exp[layer_idx][count++] = e;
    }
    g_cur_cnt[layer_idx] = count;

    /* Also update the old flat buffer as a fallback predictor
     * (used during the first pass before per-layer tracking stabilizes) */
    {
        std::lock_guard<std::mutex> lock(g_pred_mtx);
        g_pred_count = count;
        memcpy(g_pred_experts, g_cur_exp[layer_idx], count * sizeof(int));
    }
}

/* ── Expert Pinning Operations ────────────────────────────────────── */

/* Per-device pin range (model-layer count of pinned layers for this GPU).
 * Env VITRIOL_PIN_FIRST_N_LAYERS_GPU<d> overrides the global
 * VITRIOL_PIN_FIRST_N_LAYERS for that device — e.g. pin 16 layers on the
 * 12 GB card, only 6 on the 8 GB card. */
static int vitriol_pin_range(int device) {
    char env_buf[64];
    snprintf(env_buf, sizeof(env_buf), "VITRIOL_PIN_FIRST_N_LAYERS_GPU%d", device);
    const char * dev_env = getenv(env_buf);
    if (dev_env && dev_env[0]) {
        int v = atoi(dev_env);
        return v < 0 ? 0 : v;
    }
    return g_vitriol_config.pin_first_n_layers;
}

bool vitriol_pin_enabled(void) {
    if (g_vitriol_config.pin_first_n_layers > 0)
        return true;
    for (int d = 0; d < GGML_CUDA_MAX_DEVICES; d++) {
        if (vitriol_pin_range(d) > 0)
            return true;
    }
    return false;
}

CUdeviceptr vitriol_pin_ensure(
    const void    *tensor_base,
    size_t         tensor_nb02,
    int64_t        n_experts,
    CUstream       stream)
{
    if (!tensor_base || tensor_nb02 == 0 || n_experts <= 0)
        return 0;

    /* Only pin layers within the configured range.
     * Each model layer has multiple tensor ops (e.g., gate_up + down for fused MoE).
     * Divide by pin_tensors_per_layer to count in model-layer units. */
    int layer_idx = get_layer_index((uintptr_t)tensor_base);
    if (layer_idx < 0)
        return 0;

    int tensors_per_layer = g_vitriol_config.pin_tensors_per_layer;
    if (tensors_per_layer <= 0) tensors_per_layer = 2;  // safe default

    int model_layer = layer_idx / tensors_per_layer;

    int dev = vitriol_current_device();
    if (model_layer >= vitriol_pin_range(dev))
        return 0;

    /* Check if already pinned */
    {
        std::lock_guard<std::mutex> lock(g_pin_mtx);
        auto it = g_pin_map.find((uintptr_t)tensor_base);
        if (it != g_pin_map.end()) {
            return it->second;
        }
    }

    /* Get total size for this tensor */
    size_t total_size = tensor_nb02 * (size_t)n_experts;

    if (g_pin_pool_failed[dev])
        return 0;

    /* Per-device monolithic pool: allocate once on first pin for this GPU.
     * (Layer split → each pinned tensor lives on exactly one GPU.) */
    if (g_pin_pool[dev] == 0) {
        int expected_tensors = vitriol_pin_range(dev) *
                               g_vitriol_config.pin_tensors_per_layer;
        if (expected_tensors < 1) expected_tensors = 1;
        /* Use 1.5× the first tensor's size as per-slot estimate.
         * Different tensor types vary (e.g., gate_up ~66 MB, down ~98 MB),
         * so we leave margin. If a tensor exceeds remaining pool space,
         * it falls through to host RAM gracefully. */
        size_t pool_size = total_size * expected_tensors * 3 / 2;
        cudaSetDevice(dev);
        CUresult err = cuMemAlloc(&g_pin_pool[dev], pool_size);
        if (err != CUDA_SUCCESS) {
            fprintf(stderr, "VITRIOL: pin pool alloc %zu MB on GPU %d failed (%d) — pinning disabled for this device\n",
                    pool_size / 1024 / 1024, dev, (int)err);
            g_pin_pool_failed[dev] = true;
            return 0;
        }
        g_pin_pool_total[dev] = pool_size;
        g_pin_pool_offset[dev] = 0;
        if (g_vitriol_config.verbose)
            printf("VITRIOL: pin pool allocated: %zu MB (%d slots x ~%zu MB each) on GPU %d\n",
                   pool_size / 1024 / 1024, expected_tensors,
                   total_size / 1024 / 1024, dev);
    }

    /* Check if this tensor fits in remaining pool */
    if (g_pin_pool_offset[dev] + total_size > g_pin_pool_total[dev]) {
        if (g_vitriol_config.verbose)
            printf("VITRIOL: pin pool exhausted for layer %d (%zu MB needed, %zu MB left) — skipping\n",
                   layer_idx, total_size / 1024 / 1024,
                   (g_pin_pool_total[dev] - g_pin_pool_offset[dev]) / 1024 / 1024);
        return 0;
    }

    CUdeviceptr vram_buf = g_pin_pool[dev] + g_pin_pool_offset[dev];
    g_pin_pool_offset[dev] += total_size;

    /* Async H2D copy of the full tensor to VRAM.
     * In lazy mode, ensure all expert pages are locked first. */
    if (vitriol_lazy_lock_active()) {
        for (int64_t i = 0; i < n_experts; i++) {
            vitriol_ensure_expert_locked(tensor_base, (int)i, tensor_nb02);
        }
    }
    cudaSetDevice(dev);
    CUresult err = cuMemcpyHtoDAsync(vram_buf, tensor_base, total_size, stream);
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "VITRIOL: pin H2D copy failed (%d)\n", (int)err);
        return 0;
    }

    /* Synchronize so matmul can safely read from VRAM */
    cuStreamSynchronize(stream);

    /* Store in pin map */
    {
        std::lock_guard<std::mutex> lock(g_pin_mtx);
        g_pin_map[(uintptr_t)tensor_base] = vram_buf;
        g_pinned_bytes += total_size;
        g_vitriol_config.pin_active = true;
    }

    if (g_vitriol_config.verbose)
        printf("VITRIOL: pinned layer %d tensor %p -> VRAM %llx (%zu MB, %d experts x %zu bytes)\n",
               layer_idx, tensor_base, (unsigned long long)vram_buf,
               total_size / 1024 / 1024, (int)n_experts, tensor_nb02);

    return vram_buf;
}

CUdeviceptr vitriol_pin_lookup(const void *tensor_base) {
    if (!tensor_base) return 0;
    std::lock_guard<std::mutex> lock(g_pin_mtx);
    auto it = g_pin_map.find((uintptr_t)tensor_base);
    return (it != g_pin_map.end()) ? it->second : 0;
}

static bool lru_ensure_stream(int device) {
    if (g_lru_stream[device] != 0)
        return true;
    std::lock_guard<std::mutex> lock(g_lru_init_mtx);
    if (g_lru_stream[device] != 0)
        return true;
    CUresult r;
    cudaSetDevice(device);
    r = cuStreamCreate(&g_lru_stream[device], CU_STREAM_NON_BLOCKING);
    if (r != CUDA_SUCCESS) return false;
    return true;
}

/* Initialize VRAM pool for one device.  Slot size is fixed at first
 * allocation; if a later tensor has larger experts they bypass the cache
 * (return 0 → host RAM read).  This prevents pool thrashing from resizing.
 * VITRIOL_LRU_MB applies per device (each GPU gets its own up-to-LRU_MB pool). */
static bool lru_init_pool(size_t min_expert_size, int device) {
    if (g_lru_pool[device] != 0)
        return true;
    std::lock_guard<std::mutex> lock(g_lru_init_mtx);
    if (g_lru_pool[device] != 0)
        return true;

    const char* pool_env = getenv("VITRIOL_LRU_MB");
    /* Per-device override: VITRIOL_LRU_MB_<dev> wins over the global. Lets the
     * 12 GB card keep a fat pool while the 8 GB card runs a small one. */
    char dev_env_name[32];
    snprintf(dev_env_name, sizeof(dev_env_name), "VITRIOL_LRU_MB_%d", device);
    const char* dev_env = getenv(dev_env_name);
    if (dev_env && dev_env[0]) {
        pool_env = dev_env;
    }
    size_t pool_size = VITRIOL_LRU_POOL_SIZE;
    if (pool_env) {
        unsigned long mb = strtoul(pool_env, NULL, 10);
        if (mb >= 64) pool_size = mb * 1024ULL * 1024;
    }

    size_t needed_slot = (min_expert_size + 255) & ~(size_t)255;
    int    needed_slots = (int)(pool_size / needed_slot);
    if (needed_slots > VITRIOL_LRU_MAX_SLOTS) needed_slots = VITRIOL_LRU_MAX_SLOTS;
    if (needed_slots < 1) needed_slots = 1;

    /* Stream must exist before slots events are created, and both must be
     * bound to this device's context. */
    if (!lru_ensure_stream(device))
        return false;

    cudaSetDevice(device);
    CUresult err = cuMemAlloc(&g_lru_pool[device], pool_size);
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "VITRIOL: LRU pool alloc %zu MB on GPU %d failed (%d)\n",
                pool_size / 1024 / 1024, device, (int)err);
        g_lru_pool[device] = 0;
        return false;
    }

    g_lru_pool_size[device]  = pool_size;
    g_lru_slot_size[device]  = needed_slot;
    g_lru_num_slots[device]  = needed_slots;

    if (g_vitriol_config.verbose)
        printf("VITRIOL: LRU pool %zu MB, %d slots x %zu bytes on GPU %d\n",
               pool_size / 1024 / 1024, g_lru_num_slots[device], g_lru_slot_size[device], device);

    if (!g_lru_slot_events[device]) {
        g_lru_slot_events[device] = new CUevent[g_lru_num_slots[device]];
        for (int i = 0; i < g_lru_num_slots[device]; i++)
            cuEventCreate(&g_lru_slot_events[device][i], CU_EVENT_DISABLE_TIMING);
    }

    return true;
}

/* ── LRU Cache Operations ────────────────────────────────────────── */

CUdeviceptr vitriol_lru_ensure(
    const void    *tensor_base,
    int            expert_idx,
    const void    *expert_data,
    size_t         expert_size,
    CUstream       compute_stream)
{
    if (!expert_data || expert_size == 0 || !tensor_base)
        return 0;

    int dev = vitriol_current_device();

    if (!lru_ensure_stream(dev))
        return 0;
    if (!lru_init_pool(expert_size, dev))
        return 0;

    /* Expert doesn't fit in fixed-size slot → bypass cache, read from host. */
    if (expert_size > g_lru_slot_size[dev])
        return 0;

    LRUKey key = { dev, (uintptr_t)tensor_base, expert_idx };

    CUstream cstream = compute_stream;

    /* All DMA below must execute in this device's context, because both the
     * LRU pool and the enter/transition streams are per-GPU (layer split). */
    cudaSetDevice(dev);

    /* Check cache */
    {
        std::lock_guard<std::mutex> lock(g_lru_mtx);
        auto it = g_lru_map.find(key);
        if (it != g_lru_map.end()) {
            g_lru_order.remove(key);
            g_lru_order.push_front(key);
            g_lru_stats.hits++;
            /* Wait for any in-flight prefetch DMA on this slot */
            cuStreamWaitEvent(cstream, g_lru_slot_events[dev][it->second], 0);
            return g_lru_pool[dev] + (size_t)it->second * g_lru_slot_size[dev];
        }
    }

    /* Cache miss */
    g_lru_stats.misses++;

    int slot;
    {
        std::lock_guard<std::mutex> lock(g_lru_mtx);
        if ((int)g_lru_map.size() < g_lru_num_slots[dev]) {
            slot = (int)g_lru_map.size();
        } else {
            // PROVENANCE: inspiration — kimi-k3-in-c (Apache-2.0; re-derived, not copied).
            // Three-state cache principle: skip INFLIGHT slots in victim selection. Pick
            // the LRU candidate whose last DMA has COMPLETED (cuEventQuery == SUCCESS)
            // instead of blindly evicting the LRU and stalling on its in-flight fill.
            LRUKey evict = g_lru_order.back();
            for (auto it = std::prev(g_lru_order.end()); it != g_lru_order.begin(); --it) {
                auto mit = g_lru_map.find(*it);
                int s = (mit != g_lru_map.end()) ? mit->second : -1;
                if (s >= 0 && s < g_lru_num_slots[dev] &&
                    cuEventQuery(g_lru_slot_events[dev][s]) == CUDA_SUCCESS) {
                    evict = *it;
                    break;
                }
            }
            g_lru_order.remove(evict);
            auto eit = g_lru_map.find(evict);
            slot = (eit != g_lru_map.end()) ? eit->second : 0;
            if (eit != g_lru_map.end()) g_lru_map.erase(eit);
            g_lru_stats.evictions++;
        }
        g_lru_map[key] = slot;
        g_lru_order.push_front(key);
    }

    CUdeviceptr dst = g_lru_pool[dev] + (size_t)slot * g_lru_slot_size[dev];

    /* Wait for compute to finish reading this slot before overwriting it */
    cuStreamWaitEvent(g_lru_stream[dev], g_lru_slot_events[dev][slot], 0);

    /* Async DMA on dedicated stream, then make compute stream wait */
    CUresult r;
    r = cuMemcpyHtoDAsync(dst, expert_data, expert_size, g_lru_stream[dev]);
    if (r != CUDA_SUCCESS) return 0;

    r = cuEventRecord(g_lru_slot_events[dev][slot], g_lru_stream[dev]);
    if (r != CUDA_SUCCESS) return 0;

    r = cuStreamWaitEvent(cstream, g_lru_slot_events[dev][slot], 0);
    if (r != CUDA_SUCCESS) return 0;

    return dst;
}

static void vitriol_lru_prefetch_async(
    const void    *tensor_base,
    int            expert_idx,
    const void    *expert_data,
    size_t         expert_size)
{
    /* Fire-and-forget: submit DMA on LRU stream but do NOT block
     * any compute stream.  The per-expert loop's vitriol_lru_ensure
     * will wait on the event when it actually needs the data. */
    if (!expert_data || expert_size == 0 || !tensor_base)
        return;
    int dev = vitriol_current_device();
    if (!lru_ensure_stream(dev))
        return;
    if (!lru_init_pool(expert_size, dev))
        return;
    if (expert_size > g_lru_slot_size[dev])
        return;

    LRUKey key = { dev, (uintptr_t)tensor_base, expert_idx };

    /* Check cache — if already present, DMA was already submitted */
    {
        std::lock_guard<std::mutex> lock(g_lru_mtx);
        if (g_lru_map.find(key) != g_lru_map.end())
            return;
    }

    cudaSetDevice(dev);

    /* Allocate or evict slot */
    int slot;
    {
        std::lock_guard<std::mutex> lock(g_lru_mtx);
        if ((int)g_lru_map.size() < g_lru_num_slots[dev]) {
            slot = (int)g_lru_map.size();
        } else {
            LRUKey evict = g_lru_order.back();
            g_lru_order.pop_back();
            auto eit = g_lru_map.find(evict);
            slot = (eit != g_lru_map.end()) ? eit->second : 0;
            if (eit != g_lru_map.end()) g_lru_map.erase(eit);
            g_lru_stats.evictions++;
        }
        g_lru_map[key] = slot;
        g_lru_order.push_front(key);
    }

    CUdeviceptr dst = g_lru_pool[dev] + (size_t)slot * g_lru_slot_size[dev];

    /* Wait for compute to finish with this slot before overwriting */
    cuStreamWaitEvent(g_lru_stream[dev], g_lru_slot_events[dev][slot], 0);

    /* Async DMA — no cuStreamWaitEvent, compute stream NOT blocked */
    CUresult r = cuMemcpyHtoDAsync(dst, expert_data, expert_size, g_lru_stream[dev]);
    if (r != CUDA_SUCCESS) return;
    cuEventRecord(g_lru_slot_events[dev][slot], g_lru_stream[dev]);
}

void vitriol_lru_prefetch(
    const void    *tensor_base,
    int            expert_idx,
    const void    *expert_data,
    size_t         expert_size,
    CUstream       compute_stream)
{
    (void)compute_stream;
    if (vitriol_lazy_lock_active()) {
        vitriol_ensure_expert_locked(tensor_base, expert_idx, expert_size);
    }
    vitriol_lru_prefetch_async(tensor_base, expert_idx, expert_data, expert_size);
}

ggml_backend_buffer_type_t vitriol_get_expert_buffer_type(void) {
    if (g_vitriol_config.mode != VITRIOL_MODE_STREAM)
        return NULL;
    if (g_vitriol_config.max_locked_mb > 0) {
        return vitriol_get_buffer_type_lazy(0);
    }
    return vitriol_get_buffer_type(0);
}

/* Device-aware variant for multi-GPU layer splits. The loader resolves which
 * GPU owns the current layer's expert tensor and asks for that device's
 * buffer type, instead of always landing on device 0. */
ggml_backend_buffer_type_t vitriol_get_expert_buffer_type_dev(int device) {
    if (g_vitriol_config.mode != VITRIOL_MODE_STREAM)
        return NULL;
    if (g_vitriol_config.max_locked_mb > 0) {
        return vitriol_get_buffer_type_lazy(device);
    }
    return vitriol_get_buffer_type(device);
}

void vitriol_lru_mark_compute_done(CUdeviceptr vram_ptr, CUstream stream) {
    if (!vram_ptr)
        return;
    /* Find which device's pool this pointer belongs to, then record the
     * completion event on that device's slot-events array. */
    for (int dev = 0; dev < GGML_CUDA_MAX_DEVICES; dev++) {
        if (!g_lru_slot_events[dev] || g_lru_pool[dev] == 0)
            continue;
        long long offset = (long long)(vram_ptr - g_lru_pool[dev]);
        int slot = (int)(offset / g_lru_slot_size[dev]);
        if (offset >= 0 && slot >= 0 && slot < g_lru_num_slots[dev] &&
            offset % (long long)g_lru_slot_size[dev] == 0) {
            cudaSetDevice(dev);
            /* stream is the compute stream on the same device (layer split
             * keeps a tensor's compute and cache on one GPU). */
            cuEventRecord(g_lru_slot_events[dev][slot], stream);
            return;
        }
    }
}

void vitriol_cuda_cleanup_vram(void) {
    for (int dev = 0; dev < GGML_CUDA_MAX_DEVICES; dev++) {
        if (g_lru_slot_events[dev]) {
            cudaSetDevice(dev);
            for (int i = 0; i < g_lru_num_slots[dev]; i++)
                cuEventDestroy(g_lru_slot_events[dev][i]);
            delete[] g_lru_slot_events[dev];
            g_lru_slot_events[dev] = nullptr;
        }
        if (g_lru_stream[dev] != 0) {
            cudaSetDevice(dev);
            cuStreamDestroy(g_lru_stream[dev]);
            g_lru_stream[dev] = 0;
        }
        if (g_lru_pool[dev] != 0) {
            cudaSetDevice(dev);
            CUresult r = cuMemFree(g_lru_pool[dev]);
            if (r != CUDA_SUCCESS) {
                fprintf(stderr, "VITRIOL: cuMemFree(LRU pool, GPU %d) failed: %d\n", dev, (int)r);
            }
            g_lru_pool[dev] = 0;
        }
        if (g_output_cache_pool[dev] != 0) {
            cudaSetDevice(dev);
            CUresult r = cuMemFree(g_output_cache_pool[dev]);
            if (r != CUDA_SUCCESS) {
                fprintf(stderr, "VITRIOL: cuMemFree(output cache pool, GPU %d) failed: %d\n", dev, (int)r);
            }
            g_output_cache_pool[dev] = 0;
        }
        /* Free per-device pin pool (single allocation — all tensors) */
        if (g_pin_pool[dev] != 0) {
            cudaSetDevice(dev);
            CUresult r = cuMemFree(g_pin_pool[dev]);
            if (r != CUDA_SUCCESS) {
                fprintf(stderr, "VITRIOL: cuMemFree(pin pool, GPU %d) failed: %d\n", dev, (int)r);
            }
            g_pin_pool[dev] = 0;
            g_pin_pool_offset[dev] = 0;
            g_pin_pool_total[dev] = 0;
        }
    }
    {
        std::lock_guard<std::mutex> lock(g_pin_mtx);
        g_pin_map.clear();
        g_pinned_bytes = 0;
        g_vitriol_config.pin_active = false;
    }
}

void vitriol_cuda_print_stats(void) {
    uint64_t total = g_lru_stats.hits + g_lru_stats.misses;
    float hr = (total > 0) ? 100.0f * (float)g_lru_stats.hits / (float)total : 0.0f;
    printf("=== VITRIOL Statistics ===\n");
    printf("Mode: %d\n", g_vitriol_config.mode);
    for (int has_pool = 0; has_pool < GGML_CUDA_MAX_DEVICES; has_pool++) {
        if (g_lru_pool[has_pool] != 0) {
            printf("LRU Cache (GPU %d): pool=%llu MB, slots=%d, slot_size=%zu\n",
                   has_pool,
                   (unsigned long long)(g_lru_pool_size[has_pool] / 1024 / 1024),
                   g_lru_num_slots[has_pool], g_lru_slot_size[has_pool]);
        }
    }
    printf("LRU Hits: %llu\n", g_lru_stats.hits);
    printf("LRU Misses: %llu\n", g_lru_stats.misses);
    printf("LRU Hit Rate: %.2f%%\n", hr);
    printf("LRU Evictions: %llu\n", g_lru_stats.evictions);
    printf("Predictor: %s\n", g_vitriol_config.async_prefetch ? "cross-layer + temporal" : "none");
    printf("Output Cache: %s\n", g_vitriol_config.output_cache ? "enabled (approximate)" : "none");
    printf("Expert Pinning: %s\n", g_vitriol_config.pin_active ? "active" :
           g_vitriol_config.pin_first_n_layers > 0 ? "configured but no layers pinned yet" : "none");
    if (g_vitriol_config.pin_active) {
        printf("Pinned Tensors: %zu\n", g_pin_map.size());
        printf("Pinned Memory: %zu MB (%.1f GiB)\n",
               g_pinned_bytes / 1024 / 1024,
               (double)g_pinned_bytes / (1024.0 * 1024.0 * 1024.0));
    }
    if (g_vitriol_config.output_cache) {
        uint64_t oc_total = g_output_cache_stats.hits + g_output_cache_stats.misses;
        float oc_hr = (oc_total > 0) ? 100.0f * (float)g_output_cache_stats.hits / (float)oc_total : 0.0f;
        printf("Output Cache Hits: %llu\n", g_output_cache_stats.hits);
        printf("Output Cache Misses: %llu\n", g_output_cache_stats.misses);
        printf("Output Cache Hit Rate: %.2f%%\n", oc_hr);
    }
    printf("Strategy: RAM Shot + LRU VRAM cache\n");
    printf("===============================\n");
}
