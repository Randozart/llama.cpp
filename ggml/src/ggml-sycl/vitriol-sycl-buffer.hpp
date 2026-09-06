/*
 * vitriol-sycl-buffer.hpp — VITRIOL SYCL buffer type + streaming API
 */

#ifndef VITRIOL_SYCL_BUFFER_HPP
#define VITRIOL_SYCL_BUFFER_HPP

#include "ggml-backend.h"
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

struct ggml_tensor;

/* Initialization */
void vitriol_sycl_init(void);
bool vitriol_sycl_is_enabled(void);
bool vitriol_sycl_verbose(void);
size_t vitriol_sycl_lru_mb(void);
bool vitriol_sycl_prefetch_enabled(void);
bool vitriol_sycl_async_dma(void);

/* Buffer type */
bool vitriol_sycl_is_vitriol_buffer_type(ggml_backend_buffer_type_t buft);
ggml_backend_buffer_type_t vitriol_sycl_get_buffer_type(int device);
ggml_backend_buffer_type_t * vitriol_sycl_get_extra_bufts(ggml_backend_dev_t dev);

/* LRU cache */
void * vitriol_sycl_lru_ensure(
    const void    *tensor_base,
    int            expert_idx,
    const void    *expert_data,
    size_t         expert_size);

void vitriol_sycl_lru_sync(void);

void vitriol_sycl_lru_prefetch(
    const void    *tensor_base,
    int            expert_idx,
    const void    *expert_data,
    size_t         expert_size);

/* Predictive prefetching */
void vitriol_sycl_predictor_prefetch(
    const void    *tensor_base,
    size_t         expert_size);

void vitriol_sycl_predictor_update(
    const void    *tensor_base,
    size_t         expert_size,
    const int32_t *expert_ids,
    int            n_experts);

/* Cleanup */
void vitriol_sycl_print_stats(void);
void vitriol_sycl_cleanup(void);

#ifdef __cplusplus
}
#endif

#endif /* VITRIOL_SYCL_BUFFER_HPP */
