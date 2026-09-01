// PROVENANCE: inspiration — OurobourOS bitnet-rs oracle harness (own work,
// Apache-2.0 OR MIT) and ggml-org/llama.cpp (MIT), cb_eval eval-callback
// pattern from the public llama_context_params API; differential-capture
// concept learned from T-MAC-era kernel parity workflows, implemented fresh.
//
// vitriol-oracle: capture every graph node of one decode step.
//
// Registers params.cb_eval, runs a single decode, and dumps each non-view
// node (name, type, shape, raw bytes) to <out>.bin with a JSONL index
// <out>.idx. Two captures (e.g. CPU vs CUDA, or build A vs build B) are
// compared by tools/vitriol-oracle/diff.py against the parity ladder
// (L0 byte-exact, L1 cos >= 0.999, L2 greedy-equal).
//
// Usage:
//   llama-vitriol-oracle -m model.gguf -p "prompt" [-ngl 99] [-o prefix]

#include "arg.h"
#include "common.h"
#include "log.h"
#include "sampling.h"
#include "llama.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
#include <cstdlib>

namespace {

struct capture_state {
    std::ofstream data;
    std::ofstream index;
    uint64_t offset = 0;
    int nodes = 0;
};

bool capture_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * st = (capture_state *) user_data;
    if (ask) {
        // Claim every node so the callback fires again with data present.
        return true;
    }
    if (!t || !t->data || t->view_src != nullptr) {
        return true; // skip views: their storage is captured at the source
    }

    const uint64_t nbytes = ggml_nbytes(t);
    std::vector<char> buf(nbytes);
    ggml_backend_tensor_get(t, buf.data(), 0, nbytes);
    st->data.write(buf.data(), nbytes);

    char row[1024];
    snprintf(row, sizeof(row),
             "{\"i\":%d,\"name\":\"%s\",\"type\":\"%s\",\"ne\":[%lld,%lld,%lld,%lld],\"off\":%llu,\"n\":%llu}\n",
             st->nodes, t->name, ggml_type_name(t->type),
             (long long) t->ne[0], (long long) t->ne[1],
             (long long) t->ne[2], (long long) t->ne[3],
             (unsigned long long) st->offset, (unsigned long long) nbytes);
    st->index << row;

    st->offset += nbytes;
    st->nodes++;
    return true;
}

} // namespace

int main(int argc, char ** argv) {
    common_params params;
    params.n_predict = 1;
    params.prompt = "The capital of France is";

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    auto * st = new capture_state();
    const char * no_cb = getenv("ORACLE_NO_CB");
    if (!(no_cb && *no_cb)) params.cb_eval = capture_cb;
    params.cb_eval_user_data = st;

    LOG_INF("%s: capture start | model=%s n_gpu_layers=%d\n", __func__,
            params.model.path.c_str(), params.n_gpu_layers);

    llama_backend_init();
    llama_numa_init(params.numa);

    auto init = common_init_from_params(params);
    if (!init || !init->context()) {
        LOG_ERR("%s: failed to init model/context\n", __func__);
        return 1;
    }
    llama_context * ctx = init->context();

    const char * env_out = getenv("ORACLE_OUT");
    const std::string out_prefix = (env_out && *env_out) ? env_out : "oracle-capture";
    st->data.open(out_prefix + ".bin", std::ios::binary);
    st->index.open(out_prefix + ".idx");
    if (!st->data || !st->index) {
        LOG_ERR("%s: cannot open %s.bin/.idx\n", __func__, out_prefix.c_str());
        return 1;
    }
    // common_init_from_params() runs a warmup decode: its callback firings hit
    // the (open but unused) streams above with zero bytes; reset so the MEASURED
    // decode is captured from offset 0.
    st->offset = 0;
    st->nodes = 0;
    st->data.seekp(0);
    st->index.seekp(0);

    std::vector<llama_token> tokens = common_tokenize(ctx, params.prompt, true);
    llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());
    if (llama_decode(ctx, batch) != 0) {
        LOG_ERR("%s: decode failed\n", __func__);
        return 1;
    }
    llama_synchronize(ctx);

    llama_token tok = common_sampler_sample(init->sampler(0), ctx, -1);
    LOG_INF("%s: sampled token %d | nodes captured: %d | bytes: %llu\n",
            __func__, tok, st->nodes, (unsigned long long) st->offset);

    st->data.close();
    st->index.close();
    LOG_INF("%s: wrote %s.bin and %s.idx\n", __func__, out_prefix.c_str(), out_prefix.c_str());

    llama_backend_free();
    delete st;
    return 0;
}
