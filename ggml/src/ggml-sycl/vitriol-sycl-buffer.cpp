/*
 * vitriol-sycl-buffer.cpp — VITRIOL buffer type for SYCL backend
 *
 * Allocates host-pinned memory via sycl::malloc_host() for expert weights.
 * The GPU reads directly from pinned host memory -- no copy needed for
 * non-expert weights. Expert weights are streamed to device-local LRU cache
 * on demand via vitriol-sycl-integration.cpp.
 */

#include "ggml-backend-impl.h"
#include "vitriol-sycl-buffer.hpp"
#include "ggml-sycl.h"
#include "dpct/helper.hpp"
#include <sycl/sycl.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>

/* ── Config ─────────────────────────────────────────────────────── */

struct vitriol_sycl_config_t {
    bool enabled;
    bool verbose;
    size_t lru_mb;
    bool prefetch;
};

static vitriol_sycl_config_t g_vsycl_config = {};

void vitriol_sycl_init(void) {
    const char *mode = getenv("VITRIOL_MODE");
    g_vsycl_config.enabled = mode && (strcmp(mode, "stream") == 0);
    if (!g_vsycl_config.enabled) return;

    const char *verbose = getenv("VITRIOL_VERBOSE");
    g_vsycl_config.verbose = verbose && strcmp(verbose, "1") == 0;

    const char *lru = getenv("VITRIOL_LRU_MB");
    g_vsycl_config.lru_mb = lru ? strtoul(lru, nullptr, 10) : 2048;
    if (g_vsycl_config.lru_mb < 64) g_vsycl_config.lru_mb = 64;

    const char *pf = getenv("VITRIOL_PREDICTIVE_PREFETCH");
    g_vsycl_config.prefetch = pf && strcmp(pf, "1") == 0;

    if (g_vsycl_config.verbose)
        fprintf(stderr, "VITRIOL-SYCL: stream mode, LRU %zu MB, prefetch %s\n",
                g_vsycl_config.lru_mb, g_vsycl_config.prefetch ? "on" : "off");
}

bool vitriol_sycl_is_enabled(void) { return g_vsycl_config.enabled; }
bool vitriol_sycl_verbose(void) { return g_vsycl_config.verbose; }
size_t vitriol_sycl_lru_mb(void) { return g_vsycl_config.lru_mb; }
bool vitriol_sycl_prefetch_enabled(void) { return g_vsycl_config.prefetch; }

/* ── Buffer type context ───────────────────────────────────────── */

struct vitriol_sycl_buffer_type_context {
    int device;
    std::string name;
};

struct vitriol_sycl_buffer_context {
    void *base;
    size_t size;
};

/* ── Buffer type interface ─────────────────────────────────────── */

static const char * vitriol_sycl_buft_get_name(ggml_backend_buffer_type_t buft) {
    auto * ctx = (vitriol_sycl_buffer_type_context *)buft->context;
    return ctx->name.c_str();
}

static bool vitriol_sycl_buft_is_host(ggml_backend_buffer_type_t buft) {
    return true;
}

static ggml_backend_buffer_t vitriol_sycl_buft_alloc(ggml_backend_buffer_type_t buft, size_t size) {
    /* Allocate host-pinned memory via SYCL USM */
    sycl::queue q(sycl::gpu_selector_v);
    void *ptr = nullptr;
    if (size > 0) {
        ptr = sycl::malloc_host(size, q);
        if (!ptr) {
            fprintf(stderr, "VITRIOL-SYCL: sycl::malloc_host(%zu) failed\n", size);
            return nullptr;
        }
    }

    if (g_vsycl_config.verbose)
        fprintf(stderr, "VITRIOL-SYCL: allocated %zu MiB host-pinned buffer\n",
                size / 1024 / 1024);

    auto * ctx = new vitriol_sycl_buffer_context{ptr, size};
    return ggml_backend_buffer_init(buft, {
        /* .free_buffer    = */ [](ggml_backend_buffer_t buf) {
            auto * c = (vitriol_sycl_buffer_context *)buf->context;
            if (c->base) {
                sycl::queue q(sycl::gpu_selector_v);
                sycl::free(c->base, q);
            }
            delete c;
        },
        /* .get_base       = */ [](ggml_backend_buffer_t buf) -> void * {
            return ((vitriol_sycl_buffer_context *)buf->context)->base;
        },
        /* .init_tensor    = */ nullptr,
        /* .memset_tensor  = */ nullptr,
        /* .set_tensor     = */ [](ggml_backend_buffer_t, ggml_tensor *t, const void *data, size_t offset, size_t size) {
            memcpy((char *)t->data + offset, data, size);
        },
        /* .get_tensor     = */ [](ggml_backend_buffer_t, const ggml_tensor *t, void *data, size_t offset, size_t size) {
            memcpy(data, (const char *)t->data + offset, size);
        },
        /* .set_tensor_2d  = */ [](ggml_backend_buffer_t, ggml_tensor *t, const void *data,
                size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
            char *base = (char *)t->data + offset;
            const char *src = (const char *)data;
            for (size_t i = 0; i < n_copies; i++)
                memcpy(base + i * stride_tensor, src + i * stride_data, size);
        },
        /* .get_tensor_2d  = */ nullptr,
        /* .cpy_tensor     = */ nullptr,
        /* .clear          = */ [](ggml_backend_buffer_t buf, uint8_t value) {
            auto * c = (vitriol_sycl_buffer_context *)buf->context;
            if (c->base && c->size > 0)
                memset(c->base, value, c->size);
        },
        /* .reset          = */ nullptr,
    }, ctx, size);
}

static size_t vitriol_sycl_buft_get_alignment(ggml_backend_buffer_type_t buft) {
    return 32;
}

static size_t vitriol_sycl_buft_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor *tensor) {
    return (ggml_nbytes(tensor) + 31) & ~(size_t)31;
}

static const ggml_backend_buffer_type_i vitriol_sycl_buft_iface = {
    /* .get_name         = */ vitriol_sycl_buft_get_name,
    /* .alloc_buffer     = */ vitriol_sycl_buft_alloc,
    /* .get_alignment    = */ vitriol_sycl_buft_get_alignment,
    /* .get_max_size     = */ nullptr,
    /* .get_alloc_size   = */ vitriol_sycl_buft_get_alloc_size,
    /* .is_host          = */ vitriol_sycl_buft_is_host,
};

/* ── Singleton access ──────────────────────────────────────────── */

bool vitriol_sycl_is_vitriol_buffer_type(ggml_backend_buffer_type_t buft) {
    if (!buft || !buft->context) return false;
    auto * ctx = (vitriol_sycl_buffer_type_context *)buft->context;
    return ctx->name == "VITRIOL_SYCL";
}

ggml_backend_buffer_type_t vitriol_sycl_get_buffer_type(int device) {
    static std::mutex mtx;
    std::lock_guard<std::mutex> lock(mtx);

    static ggml_backend_buffer_type types[GGML_SYCL_MAX_DEVICES];
    static bool initialized = false;

    if (!initialized) {
        size_t n_devs = ggml_backend_reg_dev_count(ggml_backend_sycl_reg());
        for (size_t i = 0; i < n_devs && i < GGML_SYCL_MAX_DEVICES; i++) {
            types[i] = {
                /* .iface   = */ vitriol_sycl_buft_iface,
                /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_sycl_reg(), i),
                /* .context = */ new vitriol_sycl_buffer_type_context{(int)i, "VITRIOL_SYCL"},
            };
        }
        initialized = true;
    }

    if (device >= GGML_SYCL_MAX_DEVICES) return nullptr;
    return &types[device];
}

bool vitriol_sycl_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    return vitriol_sycl_is_vitriol_buffer_type(buft);
}

/* ── Extra buffer types (for model loader discovery) ─────────── */

/* Null-terminated array of extra buffer types per device, returned
 * via ggml_backend_dev_get_extra_bufts so the model loader discovers
 * VITRIOL SYCL bufts and allocates expert weights in host-pinned memory. */
ggml_backend_buffer_type_t * vitriol_sycl_get_extra_bufts(ggml_backend_dev_t dev) {
    if (!g_vsycl_config.enabled) return nullptr;

    /* Find device index by matching the dev pointer against known buft devices.
     * We avoid calling ggml_backend_reg_dev_get here because the SYCL reg
     * may not be fully initialized when the model loader first calls us. */
    static ggml_backend_buffer_type_t extra_bufts[GGML_SYCL_MAX_DEVICES][2];
    static bool extra_initialized = false;

    if (!extra_initialized) {
        size_t n_devs = ggml_backend_reg_dev_count(ggml_backend_sycl_reg());
        for (size_t i = 0; i < n_devs && i < GGML_SYCL_MAX_DEVICES; i++) {
            extra_bufts[i][0] = vitriol_sycl_get_buffer_type((int)i);
            extra_bufts[i][1] = nullptr;
        }
        extra_initialized = true;
    }

    /* Match dev to index */
    for (size_t i = 0; i < GGML_SYCL_MAX_DEVICES; i++) {
        ggml_backend_buffer_type_t buft = extra_bufts[i][0];
        if (buft && buft->device == dev) {
            return extra_bufts[i];
        }
    }

    return nullptr;
}
