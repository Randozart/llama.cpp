#include "vitriol-kv-probe.h"

#include "ggml.h"
#include "llama-kv-cache.h"

#include <atomic>
#include <cstdlib>
#include <cstring>

#define NL "\n"

namespace vitriol_probe {

namespace {

struct cfg {
    bool     on      = false;
    uint32_t every_n = 16;
    float    dec     = 0.90f;
};

const cfg & get_cfg() {
    static const cfg c = [] {
        cfg out;
        if (const char * mode = getenv("VITRIOL_KV_SCORE")) {
            out.on = strcmp(mode, "probe") == 0;
        }
        if (out.on) {
            if (const char * e = getenv("VITRIOL_KV_SCORE_EVERY")) {
                const unsigned long v = strtoul(e, nullptr, 10);
                if (v >= 1) {
                    out.every_n = (uint32_t) v;
                }
            }
            if (const char * e = getenv("VITRIOL_KV_DECAY")) {
                const float f = strtof(e, nullptr);
                if (f > 0.0f && f <= 1.0f) {
                    out.dec = f;
                }
            }
        }
        return out;
    }();
    return c;
}

struct capture_rec {
    int32_t       il;
    ggml_tensor * q;
    float         kq_scale;
};

// decode-side state; llama.cpp server decodes on one thread
std::atomic<uint64_t> g_step { 0 };
std::atomic<bool>     g_armed { false };

llama_kv_cache_context * g_source  = nullptr;
llama_kv_cache_context * g_pending = nullptr;
std::vector<capture_rec> g_captures;

} // namespace

bool enabled() {
    return get_cfg().on;
}

float decay() {
    return get_cfg().dec;
}

uint32_t every() {
    return get_cfg().every_n;
}

void begin_batch(const bool is_single_token) {
    if (!enabled()) {
        g_armed.store(false, std::memory_order_relaxed);
        return;
    }

    if (!is_single_token) {
        // prompt batches neither consume cadence nor carry scoring ops
        return;
    }

    const uint64_t step = g_step.fetch_add(1, std::memory_order_relaxed);

    // score the first two single-token steps (immediate feedback), then
    // every N-th decode step
    const bool due = (step < 2) || ((step % every()) == 0);

    g_armed.store(due, std::memory_order_relaxed);
}

bool active() {
    return g_armed.load(std::memory_order_relaxed);
}

void push_capture(const int32_t il, ggml_tensor * q, const float kq_scale) {
    g_captures.push_back({ il, q, kq_scale });
}

void set_source(llama_kv_cache_context * ctx) {
    g_source = ctx;
}

bool append_scores(ggml_context * ctx, ggml_cgraph * gf) {
    llama_kv_cache_context * src = g_source;
    g_source = nullptr;

    if (!active()) {
        g_captures.clear();
        return false;
    }

    if (!src || g_captures.empty()) {
        g_armed.store(false, std::memory_order_relaxed);
        g_captures.clear();
        return false;
    }

    src->vitriol_probe_reset();

    ggml_tensor * acc       = nullptr;
    uint32_t      n_layers  = 0;

    for (const auto & cap : g_captures) {
        // q: [dk, nh, 1] post-rope; fresh K view: [dk, nhk, n_kv, ns]
        ggml_tensor * k = src->get_k(ctx, cap.il);
        if (!k || k->ne[3] != 1 || k->ne[1] == 0) {
            continue;
        }

        const int64_t n_kv = k->ne[2];
        if (n_kv <= 0 || cap.q->ne[2] != 1 || cap.q->ne[0] != k->ne[0]) {
            continue;
        }

        const int64_t dk  = k->ne[0];
        const int64_t nhk = k->ne[1];
        const int64_t nh  = cap.q->ne[1];
        if (nh % nhk != 0) {
            continue;
        }

        bool contributed = false;

        for (int64_t g = 0; g < nhk; ++g) {
            // K head slice: [dk, n_kv]
            ggml_tensor * kh = ggml_view_3d(ctx, k,
                    dk, n_kv, 1,
                    ggml_row_size(k->type, dk),
                    k->nb[2],
                    ggml_row_size(k->type, dk) * g);

            // representative query head of group g: [dk, 1]
            ggml_tensor * qh = ggml_view_2d(ctx, cap.q,
                    dk, 1,
                    cap.q->nb[1],
                    cap.q->nb[1] * (g * (nh / nhk)));

            ggml_tensor * logits = ggml_mul_mat(ctx, kh, qh);   // [n_kv, 1]
            logits = ggml_scale(ctx, logits, cap.kq_scale);
            ggml_tensor * sm = ggml_soft_max(ctx, logits);      // f32

            acc = acc ? ggml_add(ctx, acc, sm) : sm;
            contributed = true;
        }

        if (contributed) {
            ++n_layers;
        }
    }

    g_captures.clear();

    const bool ok = acc != nullptr && n_layers > 0;
    // one-shot: a scored step appends exactly once; later ubatches of the
    // same decode (e.g. trailing prompt chunks) must not reset the output
    g_armed.store(false, std::memory_order_relaxed);
    if (ok) {
        ggml_build_forward_expand(gf, acc);
        src->vitriol_probe_mark_output(acc, n_layers);
        g_pending = src;
    } else {
        g_armed.store(false, std::memory_order_relaxed);
    }
    return ok;
}

void finish_pending() {
    if (g_pending) {
        llama_kv_cache_context * p = g_pending;
        g_pending = nullptr;
        p->vitriol_probe_finish();
    }
    g_armed.store(false, std::memory_order_relaxed);
}

bool has_pending_output() {
    return active();
}

} // namespace vitriol_probe
