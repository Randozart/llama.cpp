#include "vitriol-cuda-integration.h"
#include "vitriol_copy_engine.h"
#include "ggml-cuda.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
#include <cuda_runtime.h>

vitriol_config_t g_vitriol_config = {
    .mode = (vitriol_mode_t)0,
    .async_prefetch = false,
    .prefetch_ahead = 1,
    .static_layers = 15,
    .window_size_mb = 256,
    .use_double_buffer = true,
    .buffer_count = 2,
    .verbose = false,
    .benchmark = false
};

static vitriol_layer_state_t vitriol_state;
static vitriol_ce_channel_t g_ce;
static bool g_ce_initialized = false;

static struct {
    size_t sync_copies;
    size_t async_prefetch;
    size_t stream_loads;
    double total_time_ms;
} vitriol_stats;

void vitriol_cuda_init(void) {
    memset(&vitriol_state, 0, sizeof(vitriol_state));
    vitriol_state.current_layer = -1;
    vitriol_state.prefetch_layer = -1;

    g_vitriol_config.mode = (vitriol_mode_t)0;

    const char* mode_env = getenv("VITRIOL_MODE");
    if (mode_env) {
        if (strcmp(mode_env, "disabled") == 0 || strcmp(mode_env, "off") == 0)
            g_vitriol_config.mode = (vitriol_mode_t)0;
        else if (strcmp(mode_env, "sync") == 0)
            g_vitriol_config.mode = (vitriol_mode_t)1;
        else if (strcmp(mode_env, "async") == 0) {
            g_vitriol_config.mode = (vitriol_mode_t)2;
            g_vitriol_config.async_prefetch = true;
        } else if (strcmp(mode_env, "stream") == 0)
            g_vitriol_config.mode = (vitriol_mode_t)3;
    }

    const char* verbose_env = getenv("VITRIOL_VERBOSE");
    if (verbose_env && strcmp(verbose_env, "1") == 0)
        g_vitriol_config.verbose = true;

    if (g_vitriol_config.verbose) {
        const char* mode_names[] = {"disabled", "sync", "async", "stream"};
        printf("VITRIOL: Initialized (mode=%s)\n",
               mode_names[g_vitriol_config.mode & 3]);
    }

    if (g_vitriol_config.mode == (vitriol_mode_t)3) {
        if (vitriol_ce_init(&g_ce) == 0) {
            g_ce_initialized = true;
            printf("VITRIOL: Copy Engine DMA initialized\n");
        } else {
            printf("VITRIOL: CE init failed, falling back to cudaMemcpy\n");
        }
    }

    memset(&vitriol_stats, 0, sizeof(vitriol_stats));
}

void vitriol_cuda_set_current_layer(int layer_id) {
    vitriol_state.current_layer = layer_id;
    if (g_vitriol_config.verbose && vitriol_is_async_enabled())
        printf("VITRIOL: Computing layer %d (static=%s)\n",
               layer_id,
               vitriol_is_static_layer(layer_id) ? "yes" : "no");
    if (vitriol_is_async_enabled() && !vitriol_is_static_layer(layer_id)) {
        int next_layer = layer_id + g_vitriol_config.prefetch_ahead;
        vitriol_cuda_trigger_prefetch(next_layer);
    }
}

void vitriol_cuda_trigger_prefetch(int next_layer) {
    if (next_layer <= vitriol_state.current_layer) return;
    vitriol_state.prefetch_layer = next_layer;
    vitriol_state.prefetch_pending = true;
    vitriol_stats.async_prefetch++;
    if (g_vitriol_config.verbose)
        printf("VITRIOL: Triggered prefetch for layer %d\n", next_layer);
}

static bool do_ce_dma(struct ggml_tensor* tensor, const void* data, size_t size) {
    if (!g_ce_initialized) return false;

    /* Register host data as CUDA pinned memory to get GPU VA */
    CUdeviceptr host_gpu_va;
    CUresult err = cuMemHostRegister((void*)data, size, CU_MEMHOSTREGISTER_DEVICEMAP);
    if (err != CUDA_SUCCESS) {
        if (g_vitriol_config.verbose)
            printf("VITRIOL: cuMemHostRegister failed (%d), trying cudaMemcpy\n", err);
        return false;
    }
    err = cuMemHostGetDevicePointer(&host_gpu_va, (void*)data, 0);
    if (err != CUDA_SUCCESS) {
        cuMemHostUnregister((void*)data);
        return false;
    }

    /* CE DMA from host (via GPU VA) → VRAM */
    int ret = vitriol_ce_dma(&g_ce, host_gpu_va,
                             (CUdeviceptr)(uintptr_t)tensor->data, size);

    cuMemHostUnregister((void*)data);

    if (ret != 0) {
        if (g_vitriol_config.verbose)
            printf("VITRIOL: CE DMA failed, falling back\n");
        return false;
    }
    return true;
}

bool vitriol_cuda_set_tensor_hook(
    struct ggml_tensor* tensor,
    const void* data,
    size_t size,
    uint64_t tensor_file_offset
) {
    (void)tensor_file_offset;

    if (g_vitriol_config.mode == VITRIOL_MODE_DISABLED)
        return false;

    const char* name = tensor->name;
    int layer_id = -1;
    const char* layers_prefix = strstr(name, "layers.");
    if (layers_prefix) {
        char* end;
        layer_id = strtol(layers_prefix + 7, &end, 10);
    }

    if (layer_id < 0)
        return false;

    if (layer_id != vitriol_state.current_layer)
        vitriol_cuda_set_current_layer(layer_id);

    switch (g_vitriol_config.mode) {
    case VITRIOL_MODE_SYNC:
        vitriol_stats.sync_copies++;
        return false;

    case VITRIOL_MODE_ASYNC:
        vitriol_stats.async_prefetch++;
        return false;

    case VITRIOL_MODE_STREAM:
        if (!vitriol_is_static_layer(layer_id)) {
            vitriol_stats.stream_loads++;
            if (do_ce_dma(tensor, data, size)) {
                if (g_vitriol_config.verbose)
                    printf("VITRIOL: CE DMA loaded %s (layer %d, %zu bytes)\n",
                           name, layer_id, size);
                return true;
            }
        }
        return false;

    default:
        return false;
    }
}

void vitriol_cuda_print_stats(void) {
    printf("=== VITRIOL Statistics ===\n");
    printf("Mode: %d\n", g_vitriol_config.mode);
    printf("CE DMA: %s\n", g_ce_initialized ? "active" : "inactive");
    printf("Sync Copies: %zu\n", vitriol_stats.sync_copies);
    printf("Async Prefetch: %zu\n", vitriol_stats.async_prefetch);
    printf("Stream Loads: %zu\n", vitriol_stats.stream_loads);
    printf("=========================\n");
}
