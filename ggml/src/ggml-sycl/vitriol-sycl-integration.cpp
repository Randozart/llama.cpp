/*
 * vitriol-sycl-integration.cpp — VITRIOL streaming for SYCL backend
 *
 * LRU VRAM cache for hot expert weights, predictive prefetching,
 * and hooks into MUL_MAT_ID dispatch.
 */

#include "ggml-backend-impl.h"
#include "vitriol-sycl-buffer.hpp"
#include "ggml-sycl.h"
#include "dpct/helper.hpp"
#include <sycl/sycl.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unordered_map>
#include <list>
#include <mutex>
#include <vector>

/* ── LRU Cache ─────────────────────────────────────────────────── */

#define VITRIOL_LRU_MAX_SLOTS 65536

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

static void           *g_lru_pool = nullptr;
static size_t          g_lru_pool_size = 0;
static size_t          g_lru_slot_size = 0;
static int             g_lru_num_slots = 0;

static std::unordered_map<LRUKey, int, LRUKeyHash> g_lru_map;
static std::list<LRUKey> g_lru_order;
static std::mutex        g_lru_mtx;

static sycl::queue     *g_lru_queue = nullptr;
static sycl::event     *g_lru_slot_events = nullptr;
static std::mutex        g_lru_init_mtx;

static struct {
    unsigned long long hits;
    unsigned long long misses;
    unsigned long long evictions;
} g_lru_stats = {};

/* Per-slot event tracking for async DMA sync */
static int g_lru_last_slot = -1;
static bool g_lru_has_pending_dma = false;

/* ── Predictive Prefetcher ─────────────────────────────────────── */

#define VITRIOL_MAX_LAYERS 128

static uintptr_t g_layer_bases[VITRIOL_MAX_LAYERS];
static int       g_n_layers = 0;
static int       g_last_layer = -1;

static int  g_cur_exp[VITRIOL_MAX_LAYERS][256];
static int  g_cur_cnt[VITRIOL_MAX_LAYERS];
static int  g_prev_exp[VITRIOL_MAX_LAYERS][256];
static int  g_prev_cnt[VITRIOL_MAX_LAYERS];

static int get_layer_index(uintptr_t tensor_base) {
    for (int i = 0; i < g_n_layers && i < VITRIOL_MAX_LAYERS; i++) {
        if (g_layer_bases[i] == tensor_base) return i;
    }
    if (g_n_layers >= VITRIOL_MAX_LAYERS) return -1;
    g_layer_bases[g_n_layers] = tensor_base;
    return g_n_layers++;
}

static void detect_token_boundary(int layer_idx) {
    if (g_last_layer >= 0 && layer_idx <= g_last_layer) {
        for (int i = 0; i < VITRIOL_MAX_LAYERS; i++) {
            g_prev_cnt[i] = g_cur_cnt[i];
            memcpy(g_prev_exp[i], g_cur_exp[i], g_cur_cnt[i] * sizeof(int));
            g_cur_cnt[i] = 0;
        }
    }
    g_last_layer = layer_idx;
}

/* ── Initialization ────────────────────────────────────────────── */

static bool lru_ensure_queue(void) {
    if (g_lru_queue) return true;
    std::lock_guard<std::mutex> lock(g_lru_init_mtx);
    if (g_lru_queue) return true;

    auto devices = sycl::device::get_devices();
    if (devices.empty()) return false;

    g_lru_queue = new sycl::queue(devices[0], sycl::property_list{
        sycl::property::queue::in_order{}
    });
    return true;
}

static bool lru_init_pool(size_t min_expert_size) {
    if (g_lru_pool) return true;
    std::lock_guard<std::mutex> lock(g_lru_init_mtx);
    if (g_lru_pool) return true;

    if (!lru_ensure_queue()) return false;

    size_t pool_size = vitriol_sycl_lru_mb() * 1024ULL * 1024;
    size_t slot_size = (min_expert_size + 255) & ~(size_t)255;
    int num_slots = (int)(pool_size / slot_size);
    if (num_slots > VITRIOL_LRU_MAX_SLOTS) num_slots = VITRIOL_LRU_MAX_SLOTS;
    if (num_slots < 1) num_slots = 1;

    g_lru_pool = sycl::malloc_device(pool_size, *g_lru_queue);
    if (!g_lru_pool) {
        fprintf(stderr, "VITRIOL-SYCL: LRU pool alloc %zu MB failed\n", pool_size / 1024 / 1024);
        return false;
    }

    g_lru_pool_size = pool_size;
    g_lru_slot_size = slot_size;
    g_lru_num_slots = num_slots;
    g_lru_slot_events = new sycl::event[num_slots];

    if (vitriol_sycl_verbose())
        fprintf(stderr, "VITRIOL-SYCL: LRU pool %zu MB, %d slots x %zu bytes\n",
                pool_size / 1024 / 1024, num_slots, slot_size);

    return true;
}

/* ── LRU Cache Operations ──────────────────────────────────────── */

void * vitriol_sycl_lru_ensure(
    const void    *tensor_base,
    int            expert_idx,
    const void    *expert_data,
    size_t         expert_size)
{
    if (!expert_data || expert_size == 0 || !tensor_base)
        return nullptr;

    if (!lru_init_pool(expert_size))
        return nullptr;

    if (expert_size > g_lru_slot_size)
        return nullptr;

    LRUKey key = {(uintptr_t)tensor_base, expert_idx};

    /* Check cache */
    {
        std::lock_guard<std::mutex> lock(g_lru_mtx);
        auto it = g_lru_map.find(key);
        if (it != g_lru_map.end()) {
            g_lru_order.remove(key);
            g_lru_order.push_front(key);
            g_lru_stats.hits++;
            int slot = it->second;
            if (vitriol_sycl_verbose() && (g_lru_stats.hits % 100 == 0)) {
                fprintf(stderr, "VITRIOL-SYCL LRU: hits=%llu misses=%llu evictions=%llu (hit rate %.1f%%)\n",
                        g_lru_stats.hits, g_lru_stats.misses, g_lru_stats.evictions,
                        100.0 * g_lru_stats.hits / (g_lru_stats.hits + g_lru_stats.misses + 1));
            }
            if (slot < g_lru_num_slots) {
                g_lru_slot_events[slot].wait();
            }
            return (char *)g_lru_pool + slot * g_lru_slot_size;
        }
    }

    /* Cache miss */
    g_lru_stats.misses++;
    if (vitriol_sycl_verbose() && (g_lru_stats.misses % 50 == 0)) {
        fprintf(stderr, "VITRIOL-SYCL LRU: miss #%llu (hits=%llu evictions=%llu, hit rate %.1f%%)\n",
                g_lru_stats.misses, g_lru_stats.hits, g_lru_stats.evictions,
                100.0 * g_lru_stats.hits / (g_lru_stats.hits + g_lru_stats.misses + 1));
    }

    int slot;
    {
        std::lock_guard<std::mutex> lock(g_lru_mtx);
        if ((int)g_lru_map.size() < g_lru_num_slots) {
            slot = (int)g_lru_map.size();
        } else {
            /* Three-state eviction: skip slots with in-flight DMA.
             * PROVENANCE: inspired by CUDA VITRIOL's cuEventQuery approach
             * (kimi-k3-in-c, Apache-2.0; re-derived for SYCL). */
            LRUKey evict = g_lru_order.back();
            for (auto it = std::prev(g_lru_order.end()); it != g_lru_order.begin(); --it) {
                auto mit = g_lru_map.find(*it);
                int s = (mit != g_lru_map.end()) ? mit->second : -1;
                if (s >= 0 && s < g_lru_num_slots) {
                    auto status = g_lru_slot_events[s].get_info<sycl::info::event::command_execution_status>();
                    if (status == sycl::info::event_command_status::complete) {
                        evict = *it;
                        break;
                    }
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

    /* Wait for previous DMA on this slot */
    if (slot < g_lru_num_slots) {
        g_lru_slot_events[slot].wait();
    }

    /* Async DMA from host to device */
    void *dst = (char *)g_lru_pool + slot * g_lru_slot_size;
    g_lru_slot_events[slot] = g_lru_queue->memcpy(dst, expert_data, expert_size);

    /* In sync mode, block until DMA completes */
    if (!vitriol_sycl_async_dma()) {
        g_lru_slot_events[slot].wait();
    } else {
        g_lru_last_slot = slot;
    }

    return dst;
}

void vitriol_sycl_lru_sync(void) {
    /* In async mode, wait for the last-fired DMA event.
     * This is called before compute to ensure the expert copy is done.
     * The predictor prefetch fires DMA for NEXT layer's experts,
     * which overlap with THIS layer's compute. */
    if (g_lru_last_slot >= 0 && g_lru_last_slot < g_lru_num_slots) {
        g_lru_slot_events[g_lru_last_slot].wait();
        g_lru_last_slot = -1;
    }
}

void vitriol_sycl_lru_prefetch(
    const void    *tensor_base,
    int            expert_idx,
    const void    *expert_data,
    size_t         expert_size)
{
    if (!expert_data || expert_size == 0 || !tensor_base)
        return;

    if (!lru_init_pool(expert_size))
        return;

    if (expert_size > g_lru_slot_size)
        return;

    LRUKey key = {(uintptr_t)tensor_base, expert_idx};

    {
        std::lock_guard<std::mutex> lock(g_lru_mtx);
        if (g_lru_map.find(key) != g_lru_map.end())
            return;
    }

    int slot;
    {
        std::lock_guard<std::mutex> lock(g_lru_mtx);
        if ((int)g_lru_map.size() < g_lru_num_slots) {
            slot = (int)g_lru_map.size();
        } else {
            /* Three-state eviction: skip slots with in-flight DMA */
            LRUKey evict = g_lru_order.back();
            for (auto it = std::prev(g_lru_order.end()); it != g_lru_order.begin(); --it) {
                auto mit = g_lru_map.find(*it);
                int s = (mit != g_lru_map.end()) ? mit->second : -1;
                if (s >= 0 && s < g_lru_num_slots) {
                    auto status = g_lru_slot_events[s].get_info<sycl::info::event::command_execution_status>();
                    if (status == sycl::info::event_command_status::complete) {
                        evict = *it;
                        break;
                    }
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

    if (slot < g_lru_num_slots) {
        g_lru_slot_events[slot].wait();
    }

    void *dst = (char *)g_lru_pool + slot * g_lru_slot_size;
    /* Fire-and-forget DMA */
    g_lru_slot_events[slot] = g_lru_queue->memcpy(dst, expert_data, expert_size);
}

/* ── Predictive Prefetching ────────────────────────────────────── */

void vitriol_sycl_predictor_prefetch(
    const void    *tensor_base,
    size_t         expert_size)
{
    if (!vitriol_sycl_prefetch_enabled())
        return;

    int layer_idx = get_layer_index((uintptr_t)tensor_base);
    if (layer_idx < 0) return;

    detect_token_boundary(layer_idx);

    int predicted[256];
    int n_predicted = 0;

    auto add_pred = [&](int e) {
        if (e < 0) return;
        for (int i = 0; i < n_predicted; i++)
            if (predicted[i] == e) return;
        predicted[n_predicted++] = e;
    };

    if (layer_idx > 0) {
        for (int i = 0; i < g_cur_cnt[layer_idx - 1]; i++)
            add_pred(g_cur_exp[layer_idx - 1][i]);
    }

    for (int i = 0; i < g_prev_cnt[layer_idx]; i++)
        add_pred(g_prev_exp[layer_idx][i]);

    if (n_predicted == 0)
        return;

    for (int i = 0; i < n_predicted; i++) {
        int e = predicted[i];
        const void *expert_data = (const char *)tensor_base + (size_t)e * expert_size;
        vitriol_sycl_lru_prefetch(tensor_base, e, expert_data, expert_size);
    }
}

void vitriol_sycl_predictor_update(
    const void    *tensor_base,
    size_t         expert_size,
    const int32_t *expert_ids,
    int            n_experts)
{
    if (!vitriol_sycl_prefetch_enabled())
        return;

    int layer_idx = get_layer_index((uintptr_t)tensor_base);
    if (layer_idx < 0) return;

    detect_token_boundary(layer_idx);

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
}

/* ── Stats ─────────────────────────────────────────────────────── */

void vitriol_sycl_print_stats(void) {
    if (g_lru_stats.hits + g_lru_stats.misses == 0) return;
    fprintf(stderr, "VITRIOL-SYCL LRU: hits=%llu misses=%llu evictions=%llu (hit rate %.1f%%)\n",
            g_lru_stats.hits, g_lru_stats.misses, g_lru_stats.evictions,
            100.0 * g_lru_stats.hits / (g_lru_stats.hits + g_lru_stats.misses));
}

void vitriol_sycl_cleanup(void) {
    vitriol_sycl_print_stats();
    if (g_lru_slot_events) {
        delete[] g_lru_slot_events;
        g_lru_slot_events = nullptr;
    }
    if (g_lru_pool) {
        sycl::free(g_lru_pool, *g_lru_queue);
        g_lru_pool = nullptr;
    }
    if (g_lru_queue) {
        delete g_lru_queue;
        g_lru_queue = nullptr;
    }
}
