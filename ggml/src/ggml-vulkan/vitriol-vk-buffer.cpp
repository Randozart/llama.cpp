/* vitriol-vk-buffer.cpp — VITRIOL Vulkan buffer type for dense tensor placement
 *
 * Allocates page-locked host RAM via mmap + mlock, then imports into Vulkan
 * via VK_EXT_external_memory_host. Dense tensor weights (SSM, attention,
 * norms) live in host RAM and are read by the GPU over PCIe with zero-copy.
 *
 * Buffer type interface:
 *   - is_host() = true (data is CPU-accessible, in page-locked RAM)
 *   - alloc_buffer() = mmap + mlock, creates VkBuffer via external_memory_host
 *   - free_buffer() = munmap + munlock
 *   - get_base() = host pointer
 *
 * The VkBuffer is created lazily on first Vulkan dispatch (via
 * ggml_vk_buffer_from_host_ptr) and cached in the VK buffer context.
 */

#include "vitriol-vk-buffer.h"
#include "ggml-backend-impl.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <sys/mman.h>
#include <unistd.h>

/* ── Chimera mode flag ─────────────────────────────────────────── */

static bool g_chimera_enabled = false;

void vitriol_set_chimera_mode(bool enabled) {
    g_chimera_enabled = enabled;
    fprintf(stderr, "VITRIOL: Chimera mode %s\n", enabled ? "enabled" : "disabled");
}

bool vitriol_chimera_enabled(void) {
    return g_chimera_enabled;
}

/* ── Buffer type context ───────────────────────────────────────── */

struct vitriol_vk_buffer_type_context {
    int     device_idx;
    std::string name;
};

/* ── Buffer context (per allocation) ───────────────────────────── */

/* Extended context that also stores optional lazy-created VkBuffer info.
 * The VkBuffer is created lazily because ggml_vk_buffer_from_host_ptr
 * needs the vk_device, which is only available during Vulkan dispatch. */
struct vitriol_vk_buffer_ctx {
    void *  ptr;        /* mmap'd base address — CPU-accessible */
    size_t  size;       /* total allocation size */
    size_t  alignment;  /* alignment used */
    void *  vk_device;  /* opaque pointer to vk_device */
    void *  vk_buf;     /* opaque void* slots — set by ggml-vulkan.cpp */
    void *  raw_vkbuf;  /* raw VkBuffer handle */
    void *  raw_memory; /* raw VkDeviceMemory handle */
};

/* ── Buffer interface ──────────────────────────────────────────── */

static const char * vitriol_vk_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    auto * ctx = (vitriol_vk_buffer_type_context *)buft->context;
    return ctx->name.c_str();
}

static void vitriol_vk_buffer_free(ggml_backend_buffer_t buffer) {
    auto * ctx = (vitriol_vk_buffer_ctx *)buffer->context;
    if (ctx->vk_buf) {
        /* Clean up the nested shared_ptr stored by ggml-vulkan.cpp */
        auto * stash = (std::shared_ptr<void> *)ctx->vk_buf;
        delete stash;
        ctx->vk_buf = nullptr;
    }
    if (ctx->ptr && ctx->size) {
        munlock(ctx->ptr, ctx->size);
        munmap(ctx->ptr, ctx->size);
    }
    delete ctx;
}

static void * vitriol_vk_buffer_get_base(ggml_backend_buffer_t buffer) {
    auto * ctx = (vitriol_vk_buffer_ctx *)buffer->context;
    return ctx->ptr;
}

static void vitriol_vk_buffer_set_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    auto * ctx = (vitriol_vk_buffer_ctx *)buffer->context;
    memcpy((char *)ctx->ptr + offset, data, size);
}

static void vitriol_vk_buffer_get_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    auto * ctx = (vitriol_vk_buffer_ctx *)buffer->context;
    memcpy(data, (const char *)ctx->ptr + offset, size);
}

static void vitriol_vk_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    auto * ctx = (vitriol_vk_buffer_ctx *)buffer->context;
    memset(ctx->ptr, value, ctx->size);
}

static const struct ggml_backend_buffer_i vitriol_vk_buffer_interface = {
    /* .free_buffer    = */ vitriol_vk_buffer_free,
    /* .get_base       = */ vitriol_vk_buffer_get_base,
    /* .init_tensor    = */ nullptr,
    /* .memset_tensor  = */ nullptr,
    /* .set_tensor     = */ vitriol_vk_buffer_set_tensor,
    /* .get_tensor     = */ vitriol_vk_buffer_get_tensor,
    /* .set_tensor_2d  = */ nullptr,
    /* .get_tensor_2d  = */ nullptr,
    /* .cpy_tensor     = */ nullptr,
    /* .clear          = */ vitriol_vk_buffer_clear,
    /* .reset          = */ nullptr,
};

static bool vitriol_vk_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    return true;  /* page-locked host RAM, GPU via VK_EXT_external_memory_host */
}

static size_t vitriol_vk_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return 4096;  /* page-aligned — fine for mmap + VK_EXT_external_memory_host */
}

static size_t vitriol_vk_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const struct ggml_tensor * tensor) {
    return ggml_nbytes(tensor);
}

static ggml_backend_buffer_t vitriol_vk_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    /* Align to 64K for VK_EXT_external_memory_host compatibility */
    size_t alignment = 65536;
    size_t aligned_size = (size + alignment - 1) & ~(alignment - 1);

    void * ptr = nullptr;
    if (posix_memalign(&ptr, alignment, aligned_size) != 0) {
        fprintf(stderr, "VITRIOL VK: posix_memalign(%zu) failed\n", aligned_size);
        return nullptr;
    }

    /* Touch each page to ensure physical RAM backing */
    {
        volatile char * touch = (volatile char *)ptr;
        for (size_t off = 0; off < aligned_size; off += 4096) {
            touch[off] = 0;
        }
    }

    /* Pin in RAM — never swap */
    mlock(ptr, aligned_size);

    auto * ctx = new vitriol_vk_buffer_ctx{
        ptr, aligned_size, alignment, nullptr, nullptr, nullptr, nullptr
    };

    return ggml_backend_buffer_init(buft, vitriol_vk_buffer_interface, ctx, aligned_size);
}

static const struct ggml_backend_buffer_type_i vitriol_vk_buffer_type_interface = {
    /* .get_name         = */ vitriol_vk_buffer_type_get_name,
    /* .alloc_buffer     = */ vitriol_vk_buffer_type_alloc_buffer,
    /* .get_alignment    = */ vitriol_vk_buffer_type_get_alignment,
    /* .get_max_size     = */ nullptr,
    /* .get_alloc_size   = */ vitriol_vk_buffer_type_get_alloc_size,
    /* .is_host          = */ vitriol_vk_buffer_type_is_host,
};

/* ── Singleton access ─────────────────────────────────────────── */

ggml_backend_buffer_type_t vitriol_get_vk_buffer_type(int device_idx) {
    static std::mutex mtx;
    std::lock_guard<std::mutex> lock(mtx);

    static ggml_backend_buffer_type types[16];
    static bool initialized = false;

    if (!initialized) {
        for (int i = 0; i < 16; i++) {
            types[i] = {
                /* .iface   = */ vitriol_vk_buffer_type_interface,
                /* .device  = */ nullptr,
                /* .context = */ new vitriol_vk_buffer_type_context{i, "VITRIOL_VK"},
            };
        }
        initialized = true;
    }

    if (device_idx < 0 || device_idx >= 16) {
        return nullptr;
    }
    return &types[device_idx];
}

bool vitriol_is_vitriol_vk_buffer_type(ggml_backend_buffer_type_t buft) {
    if (!buft || !buft->context) return false;
    /* Check by name — VITRIOL VK has a unique context type name */
    auto * ctx = (vitriol_vk_buffer_type_context *)buft->context;
    return ctx->name == "VITRIOL_VK";
}

void * vitriol_vk_buffer_get_host_ptr(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor) {
    auto * ctx = (vitriol_vk_buffer_ctx *)buffer->context;
    if (!ctx || !ctx->ptr) return nullptr;
    return (char *)ctx->ptr + tensor->view_offs;
}

/* ── Lazy VkBuffer slot management ────────────────────────────── */

void ** vitriol_vk_buffer_get_vk_buf_slot(ggml_backend_buffer_t buffer) {
    auto * ctx = (vitriol_vk_buffer_ctx *)buffer->context;
    if (!ctx) return nullptr;
    return &ctx->vk_buf;
}

void ** vitriol_vk_buffer_get_device_slot(ggml_backend_buffer_t buffer) {
    auto * ctx = (vitriol_vk_buffer_ctx *)buffer->context;
    if (!ctx) return nullptr;
    return &ctx->vk_device;
}

void ** vitriol_vk_buffer_get_raw_vkbuf_slot(ggml_backend_buffer_t buffer) {
    auto * ctx = (vitriol_vk_buffer_ctx *)buffer->context;
    if (!ctx) return nullptr;
    return &ctx->raw_vkbuf;
}

void ** vitriol_vk_buffer_get_raw_memory_slot(ggml_backend_buffer_t buffer) {
    auto * ctx = (vitriol_vk_buffer_ctx *)buffer->context;
    if (!ctx) return nullptr;
    return &ctx->raw_memory;
}