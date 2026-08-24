#pragma once

#include <cstdint>
#include <utility>
#include <vector>

struct ggml_context;
struct ggml_cgraph;
struct ggml_tensor;
class llama_kv_cache_context;

// VITRIOL LULL Phase 1 — attention-probe KV scoring.
//
// During graph build, the attention hook only RECORDS (il, q) pairs plus the
// owning kv-cache context. After model.build_graph() completes,
// append_scores() adds one small subgraph at the TAIL of the compute graph:
// per recorded layer, softmax(q_g · K_g / sqrt(dk)) summed over KV-head
// groups and layers into a single [n_kv] f32 tensor. Appending at the tail
// keeps the model builders' strict last-node-expansion invariant intact
// (mid-build insertion aborts on GGML_ASSERT in ggml_build_forward_impl).
//
// The score tensor is downloaded once per scored step at
// llama_context::synchronize() and accumulated into per-cell importance
// scores consumed by evict_sparse().
//
// Approximations (documented in .opencode/plans/lull-plan-2026-08-24.md):
//  - one representative query head per GQA group (first head);
//  - empty cells are not masked pre-softmax (hole-free while appending);
//  - single-KV-stream sequences only;
//  - decode runs on one thread (server model).
//
// Env vars:
//   VITRIOL_KV_SCORE       = probe | off   (default off)
//   VITRIOL_KV_SCORE_EVERY = N             (score every N decode steps, default 16)
//   VITRIOL_KV_DECAY       = F             (per-step exponential decay, default 0.90)
//
// PROVENANCE: original VITRIOL implementation. Idea-level inspiration:
// H2O heavy-hitter scoring (Zhang et al., NeurIPS 2023, paper) and
// StreamingLLM sink preservation (Xiao et al., 2023, paper) — learned from
// papers, nothing copied (implementations are Apache-2.0 and off-limits).

namespace vitriol_probe {

bool enabled();
float decay();
uint32_t every();

// Called at the start of a decode batch. Arms scoring for single-token
// batches on their cadence; prompt batches never consume cadence.
void begin_batch(bool is_single_token);

// true between a due begin_batch() and end_batch_pending()
bool active();

// ── graph-build side (called from llm_graph_context::build_attn) ──

void push_capture(int32_t il, ggml_tensor * q, float kq_scale);
void set_source(llama_kv_cache_context * ctx);

// ── completion side (called from llama_context) ──

// Builds the tail subgraph into gf/ctx using the captures collected during
// this build, marks the score output on the source kv-cache context, and
// disarms. Returns false if nothing was appended (guards failed).
bool append_scores(ggml_context * ctx, struct ggml_cgraph * gf);

// Downloads the finished score tensor (GPU work is done by the time this is
// called) and folds it into per-cell scores. Safe to call unconditionally.
void finish_pending();

} // namespace vitriol_probe
