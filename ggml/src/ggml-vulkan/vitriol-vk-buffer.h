/* vitriol-vk-buffer.h — VITRIOL Vulkan buffer type for dense tensor placement
 *
 * Allocates page-locked host RAM (mmap + mlock) and imports it into Vulkan
 * via VK_EXT_external_memory_host. The GPU reads dense tensor weights
 * (SSM scan, attention, norms) directly from host RAM over PCIe with
 * zero-copy — no VRAM used for weight storage.
 */

#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include <cstddef>

struct vitriol_vk_buffer_context {
    void *       host_ptr;   /* mmap'd, mlock'd, CPU-accessible host RAM */
    size_t       size;       /* total buffer size */
    size_t       alignment;  /* alignment constraint (typically 64K-256K) */
};

#ifdef __cplusplus
extern "C" {
#endif

/* Get the VITRIOL VK buffer type for a given device index.
 * Returns nullptr if VK_EXT_external_memory_host is not available. */
GGML_API ggml_backend_buffer_type_t vitriol_get_vk_buffer_type(int device_idx);

/* Check if a buffer type is the VITRIOL VK type */
GGML_API bool vitriol_is_vitriol_vk_buffer_type(ggml_backend_buffer_type_t buft);

/* Get the host pointer from a VITRIOL VK buffer for a given tensor */
GGML_API void * vitriol_vk_buffer_get_host_ptr(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor);

/* Get the opaque VkBuffer cache slot from a VITRIOL VK buffer context.
 * Returns a pointer-to-pointer (void**) that ggml-vulkan.cpp can store
 * a vk_buffer (shared_ptr<vk_buffer_struct>) into. */
GGML_API void ** vitriol_vk_buffer_get_vk_buf_slot(ggml_backend_buffer_t buffer);

/* Get the vk_device cache slot from a VITRIOL VK buffer context.
 * Returns a pointer-to-pointer (void**) that ggml-vulkan.cpp can store
 * a vk_device pointer into for lazy initialization. */
GGML_API void ** vitriol_vk_buffer_get_device_slot(ggml_backend_buffer_t buffer);

/* Get the VkBuffer and VkDeviceMemory raw handle slots.
 * ggml-vulkan.cpp stores raw VkBuffer/VkDeviceMemory handles here
 * instead of using shared_ptr types not visible from this TU. */
GGML_API void ** vitriol_vk_buffer_get_raw_vkbuf_slot(ggml_backend_buffer_t buffer);
GGML_API void ** vitriol_vk_buffer_get_raw_memory_slot(ggml_backend_buffer_t buffer);

/* Set whether Chimera mode is active (routes dense tensors to VK type) */
GGML_API void vitriol_set_chimera_mode(bool enabled);

/* Check if Chimera mode is active */
GGML_API bool vitriol_chimera_enabled(void);

#ifdef __cplusplus
}
#endif