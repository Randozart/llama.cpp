/*
 * vitriol-buffer.cpp — VITRIOL buffer type for expert tensor placement
 *
 * RAM Shot:
 *   - Allocates hugepage-backed system RAM via mmap
 *   - mlock() pins it — never swapped
 *   - cudaHostRegister() page-locks it — GPU reads directly over PCIe DMA
 *   - Reports is_host=true — scheduler routes MUL_MAT_ID to CUDA
 *   - Set_tensor copies expert data normally into the buffer
 *
 * Expert weights live in page-locked host RAM. The GPU's Copy Engine
 * streams them across PCIe during MUL_MAT_ID kernels. No VRAM used for
 * weight storage — only for activations, KV cache, and compute buffers.
 */

#include "vitriol-buffer.h"
#include "vitriol-cuda-integration.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <sys/mman.h>
#include <unistd.h>
#include <fcntl.h>
#include <cuda_runtime.h>

/* ── Buffer type interface ────────────────────────────────────── */

struct vitriol_buffer_type_context {
    int device;
    std::string name;
};

static const char * vitriol_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    auto * ctx = (vitriol_buffer_type_context *)buft->context;
    return ctx->name.c_str();
}

/* Check if a buffer type is VITRIOL by comparing the get_name function pointer */
bool vitriol_is_vitriol_buffer_type(ggml_backend_buffer_type_t buft) {
    return buft && buft->iface.get_name == vitriol_buffer_type_get_name;
}

/* VITRIOL buffer context */
struct vitriol_buffer_context {
    void * base;
    size_t size;
};

static void vitriol_buffer_free(ggml_backend_buffer_t buffer) {
    auto * ctx = (vitriol_buffer_context *)buffer->context;
    if (ctx->base && ctx->size) {
        cudaError_t err = cudaHostUnregister(ctx->base);
        if (err != cudaSuccess) {
            fprintf(stderr, "VITRIOL: cudaHostUnregister failed: %s\n", cudaGetErrorString(err));
        }
        munmap(ctx->base, ctx->size);
    }
    delete ctx;
}

static void * vitriol_buffer_get_base(ggml_backend_buffer_t buffer) {
    auto * ctx = (vitriol_buffer_context *)buffer->context;
    return ctx->base;
}

static void vitriol_buffer_set_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    auto * ctx = (vitriol_buffer_context *)buffer->context;
    memcpy((char *)ctx->base + offset, data, size);
}

static void vitriol_buffer_get_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    auto * ctx = (vitriol_buffer_context *)buffer->context;
    memcpy(data, (const char *)ctx->base + offset, size);
}

static void vitriol_buffer_set_tensor_2d(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, const void * data,
        size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    auto * ctx = (vitriol_buffer_context *)buffer->context;
    char * base = (char *)ctx->base + offset;
    const char * src = (const char *)data;
    for (size_t i = 0; i < n_copies; i++) {
        memcpy(base + i * stride_tensor, src + i * stride_data, size);
    }
}

static void vitriol_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    auto * ctx = (vitriol_buffer_context *)buffer->context;
    memset(ctx->base, value, ctx->size);
}

static const struct ggml_backend_buffer_i vitriol_buffer_interface = {
    /* .free_buffer    = */ vitriol_buffer_free,
    /* .get_base       = */ vitriol_buffer_get_base,
    /* .init_tensor    = */ nullptr,
    /* .memset_tensor  = */ nullptr,
    /* .set_tensor     = */ vitriol_buffer_set_tensor,
    /* .get_tensor     = */ vitriol_buffer_get_tensor,
    /* .set_tensor_2d  = */ vitriol_buffer_set_tensor_2d,
    /* .get_tensor_2d  = */ nullptr,
    /* .cpy_tensor     = */ nullptr,
    /* .clear          = */ vitriol_buffer_clear,
    /* .reset          = */ nullptr,
};

/* ── Buffer type allocation ───────────────────────────────────── */

static bool vitriol_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    return true;  // page-locked host RAM (or disk offload with cudaHostRegister), GPU via DMA
}

static ggml_backend_buffer_t vitriol_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    bool disk_offload = g_vitriol_config.disk_offload;
    void * ptr = nullptr;

    if (disk_offload) {
        const char * model_path = getenv("VITRIOL_MODEL_PATH");
        if (!model_path || !*model_path) {
            fprintf(stderr, "VITRIOL: disk offload requires VITRIOL_MODEL_PATH\n");
            return nullptr;
        }
        int fd = open(model_path, O_RDONLY);
        if (fd < 0) {
            fprintf(stderr, "VITRIOL: cannot open model %s: %m\n", model_path);
            return nullptr;
        }
        off_t file_size = lseek(fd, 0, SEEK_END);
        if (file_size < 0) {
            fprintf(stderr, "VITRIOL: lseek failed: %m\n");
            close(fd);
            return nullptr;
        }
        if ((size_t)file_size < size) {
            fprintf(stderr, "VITRIOL: file smaller (%lld) than requested (%zu)\n",
                    (long long)file_size, size);
            close(fd);
            return nullptr;
        }
        ptr = mmap(NULL, size, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_POPULATE, fd, 0);
        close(fd);
        if (ptr == MAP_FAILED) {
            fprintf(stderr, "VITRIOL: file mmap(%zu) failed: %m\n", size);
            return nullptr;
        }
        madvise(ptr, size, MADV_HUGEPAGE);

        /* cudaHostRegister pins pages for GPU DMA access.
         * No mlock — pages can be evicted by OS if not accessed by GPU. */
        cudaError_t err = cudaHostRegister(ptr, size, 0);
        if (err != cudaSuccess) {
            fprintf(stderr, "VITRIOL: disk offload cudaHostRegister(%zu) failed: %s\n",
                    size, cudaGetErrorString(err));
            munmap(ptr, size);
            return nullptr;
        }

        if (g_vitriol_config.verbose)
            printf("VITRIOL: disk offload — file-backed mmap (%zu MiB)\n",
                   size / 1024 / 1024);
    } else {
        ptr = mmap(NULL, size, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE, -1, 0);
        if (ptr == MAP_FAILED) {
            fprintf(stderr, "VITRIOL: mmap(%zu) failed: %m\n", size);
            return nullptr;
        }

        /* Touch each page to ensure it's backed by physical RAM */
        {
            volatile char * touch = (volatile char *)ptr;
            for (size_t offset = 0; offset < size; offset += 4096) {
                touch[offset] = 0;
            }
        }

        /* Hint: coalesce into transparent hugepages (2 MB) for lower TLB pressure */
        madvise(ptr, size, MADV_HUGEPAGE);

        /* Pin in RAM — never swap */
        if (mlock(ptr, size) != 0) {
            fprintf(stderr, "VITRIOL: mlock(%zu) failed: %m — continuing without mlock (may cause swap stutter)\n", size);
        }

        /* Register for GPU DMA access — makes it accessible from CUDA kernels */
        cudaError_t err = cudaHostRegister(ptr, size, 0);
        if (err != cudaSuccess) {
            fprintf(stderr, "VITRIOL: cudaHostRegister(%zu) failed: %s\n", size, cudaGetErrorString(err));
            munmap(ptr, size);
            return nullptr;
        }
    }

    auto * ctx = new vitriol_buffer_context{ptr, size};
    return ggml_backend_buffer_init(buft, vitriol_buffer_interface, ctx, size);
}

static size_t vitriol_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return 32;
}

static size_t vitriol_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const struct ggml_tensor * tensor) {
    size_t size = ggml_nbytes(tensor);
    return (size + 31) & ~(size_t)31;
}

static const struct ggml_backend_buffer_type_i vitriol_buffer_type_interface = {
    /* .get_name         = */ vitriol_buffer_type_get_name,
    /* .alloc_buffer     = */ vitriol_buffer_type_alloc_buffer,
    /* .get_alignment    = */ vitriol_buffer_type_get_alignment,
    /* .get_max_size     = */ nullptr,
    /* .get_alloc_size   = */ vitriol_buffer_type_get_alloc_size,
    /* .is_host          = */ vitriol_buffer_type_is_host,
};

/* ── Singleton access ─────────────────────────────────────────── */

ggml_backend_buffer_type_t vitriol_get_buffer_type(int device) {
    static std::mutex mtx;
    std::lock_guard<std::mutex> lock(mtx);

    static ggml_backend_buffer_type types[GGML_CUDA_MAX_DEVICES];
    static bool initialized = false;

    if (!initialized) {
        for (int i = 0; i < ggml_backend_cuda_get_device_count(); i++) {
            types[i] = {
                /* .iface   = */ vitriol_buffer_type_interface,
                /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_cuda_reg(), i),
                /* .context = */ new vitriol_buffer_type_context{i, "VITRIOL"},
            };
        }
        initialized = true;
    }

    if (device >= ggml_backend_cuda_get_device_count())
        return nullptr;
    return &types[device];
}
