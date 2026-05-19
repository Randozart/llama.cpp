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
    }
} s_vitriol_config_init;

/* ── LRU Cache ──────────────────────────────────────────────────── */

#define VITRIOL_LRU_POOL_SIZE  (2048ULL * 1024 * 1024)  // default 2 GB VRAM pool (env VITRIOL_LRU_MB overrides)
#define VITRIOL_LRU_MAX_SLOTS  65536

static CUdeviceptr g_lru_pool = 0;
static size_t      g_lru_pool_size = 0;
static size_t      g_lru_slot_size = 0;
static int         g_lru_num_slots = 0;

/* Composite key: (tensor_base_address, expert_idx) prevents
 * cross-layer collisions where expert 0 of layer 1 != expert 0 of layer 2. */
struct LRUKey {
    uintptr_t tensor_base;
    int       expert_idx;
    bool operator==(const LRUKey &o) const {
        return tensor_base == o.tensor_base && expert_idx == o.expert_idx;
    }
};

struct LRUKeyHash {
    size_t operator()(const LRUKey &k) const {
        return (size_t)(k.tensor_base * 2654435761U) ^ (size_t)k.expert_idx;
    }
};

static std::unordered_map<LRUKey, int, LRUKeyHash> g_lru_map;
static std::list<LRUKey> g_lru_order;
static std::mutex        g_lru_mtx;

/* Dedicated stream + event for async DMA */
static CUstream  g_lru_stream = 0;
static CUevent   g_lru_event  = 0;

static struct LRUStats {
    unsigned long long hits;
    unsigned long long misses;
    unsigned long long evictions;
} g_lru_stats;

static bool lru_init_pool(size_t min_expert_size);
static bool lru_ensure_stream(void);

/* ── Initialization ──────────────────────────────────────────────── */

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

    if (g_vitriol_config.mode == VITRIOL_MODE_STREAM) {
        if (g_vitriol_config.verbose)
            printf("VITRIOL: stream mode — page-locked host RAM + LRU VRAM cache\n");
    }

    if (g_vitriol_config.async_prefetch) {
        if (g_vitriol_config.verbose)
            printf("VITRIOL: predictive prefetching enabled (cross-layer + temporal)\n");
    }

    if (g_vitriol_config.disk_offload) {
        if (g_vitriol_config.verbose)
            printf("VITRIOL: disk offload mode — file-backed mmap, no page-lock\n");
    }

    static bool first_init = true;
    if (first_init) {
        memset(&g_lru_stats, 0, sizeof(g_lru_stats));
        first_init = false;

        /* Register atexit handler for stats dump */
        atexit(vitriol_cuda_print_stats);
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

    if (!lru_ensure_stream())
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

static bool lru_ensure_stream(void) {
    if (g_lru_stream != 0)
        return true;
    CUresult r;
    r = cuStreamCreate(&g_lru_stream, CU_STREAM_NON_BLOCKING);
    if (r != CUDA_SUCCESS) return false;
    r = cuEventCreate(&g_lru_event, CU_EVENT_DISABLE_TIMING);
    if (r != CUDA_SUCCESS) return false;
    return true;
}

/* Initialize VRAM pool once.  Slot size is fixed at first allocation;
 * if a later tensor has larger experts they bypass the cache (return 0
 * → host RAM read).  This prevents pool thrashing from resizing. */
static bool lru_init_pool(size_t min_expert_size) {
    if (g_lru_pool != 0)
        return true;

    const char* pool_env = getenv("VITRIOL_LRU_MB");
    size_t pool_size = VITRIOL_LRU_POOL_SIZE;
    if (pool_env) {
        unsigned long mb = strtoul(pool_env, NULL, 10);
        if (mb >= 64) pool_size = mb * 1024ULL * 1024;
    }

    size_t needed_slot = (min_expert_size + 255) & ~(size_t)255;
    int    needed_slots = (int)(pool_size / needed_slot);
    if (needed_slots > VITRIOL_LRU_MAX_SLOTS) needed_slots = VITRIOL_LRU_MAX_SLOTS;
    if (needed_slots < 1) needed_slots = 1;

    CUresult err = cuMemAlloc(&g_lru_pool, pool_size);
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "VITRIOL: LRU pool alloc %zu MB failed (%d)\n",
                pool_size / 1024 / 1024, (int)err);
        g_lru_pool = 0;
        return false;
    }

    g_lru_pool_size  = pool_size;
    g_lru_slot_size  = needed_slot;
    g_lru_num_slots  = needed_slots;

    if (g_vitriol_config.verbose)
        printf("VITRIOL: LRU pool %zu MB, %d slots x %zu bytes\n",
               pool_size / 1024 / 1024, g_lru_num_slots, g_lru_slot_size);

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

    if (!lru_ensure_stream())
        return 0;
    if (!lru_init_pool(expert_size))
        return 0;

    /* Expert doesn't fit in fixed-size slot → bypass cache, read from host. */
    if (expert_size > g_lru_slot_size)
        return 0;

    LRUKey key = { (uintptr_t)tensor_base, expert_idx };

    /* Check cache */
    {
        std::lock_guard<std::mutex> lock(g_lru_mtx);
        auto it = g_lru_map.find(key);
        if (it != g_lru_map.end()) {
            g_lru_order.remove(key);
            g_lru_order.push_front(key);
            g_lru_stats.hits++;
            return g_lru_pool + (size_t)it->second * g_lru_slot_size;
        }
    }

    /* Cache miss */
    g_lru_stats.misses++;

    int slot;
    {
        std::lock_guard<std::mutex> lock(g_lru_mtx);
        if ((int)g_lru_map.size() < g_lru_num_slots) {
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

    CUdeviceptr dst = g_lru_pool + (size_t)slot * g_lru_slot_size;

    /* Async DMA on dedicated stream, then make compute stream wait */
    CUresult r;
    r = cuMemcpyHtoDAsync(dst, expert_data, expert_size, g_lru_stream);
    if (r != CUDA_SUCCESS) return 0;

    r = cuEventRecord(g_lru_event, g_lru_stream);
    if (r != CUDA_SUCCESS) return 0;

    r = cuStreamWaitEvent(compute_stream, g_lru_event, 0);
    if (r != CUDA_SUCCESS) return 0;

    return dst;
}

void vitriol_lru_prefetch(
    const void    *tensor_base,
    int            expert_idx,
    const void    *expert_data,
    size_t         expert_size,
    CUstream       compute_stream)
{
    /* Fire-and-forget: load into LRU but don't wait on compute_stream.
     * If already cached this is a no-op (the lookup/promotion still happens). */
    vitriol_lru_ensure(tensor_base, expert_idx, expert_data, expert_size, compute_stream);
}

ggml_backend_buffer_type_t vitriol_get_expert_buffer_type(void) {
    if (g_vitriol_config.mode != VITRIOL_MODE_STREAM)
        return NULL;
    return vitriol_get_buffer_type(0);
}

void vitriol_cuda_print_stats(void) {
    uint64_t total = g_lru_stats.hits + g_lru_stats.misses;
    float hr = (total > 0) ? 100.0f * (float)g_lru_stats.hits / (float)total : 0.0f;
    printf("=== VITRIOL Statistics ===\n");
    printf("Mode: %d\n", g_vitriol_config.mode);
    printf("LRU Cache: pool=%llu MB, slots=%d, slot_size=%zu\n",
           (unsigned long long)(g_lru_pool_size / 1024 / 1024),
           g_lru_num_slots, g_lru_slot_size);
    printf("LRU Hits: %llu\n", g_lru_stats.hits);
    printf("LRU Misses: %llu\n", g_lru_stats.misses);
    printf("LRU Hit Rate: %.2f%%\n", hr);
    printf("LRU Evictions: %llu\n", g_lru_stats.evictions);
    printf("Predictor: %s\n", g_vitriol_config.async_prefetch ? "cross-layer + temporal" : "none");
    printf("Strategy: RAM Shot + LRU VRAM cache\n");
    printf("===============================\n");
}
