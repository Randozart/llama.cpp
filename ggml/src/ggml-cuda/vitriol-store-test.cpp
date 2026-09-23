// vitriol-store-self-test: mkfs a store file, write tensors, read back, verify.
// C++26 per docs/cpp26-conventions.md. Run: g++ -std=c++26 ... -luring && ./a.out
#include "vitriol-store.h"
#include <cstdio>
#include <cstring>
#include <vector>
#include <unistd.h>
#include <fcntl.h>

using namespace vitriol;

int main() {
    const char * path = "/tmp/opencode/store_selftest.bin";
    unlink(path);

    // 3 expert tensors: 2 routed (512 experts, 100 B real -> 512 slot), 1 shexp
    // (1 expert, 4096 B real). Non-512 data_bytes exercise the padded-slot path.
    std::vector<store_tensor_desc> tensors = {
        {0, 0, 512, 512, 100, 0, 0},
        {0, 2, 512, 512, 100, 0, 0},
        {0, 3, 1, 4096, 4096, 0, 0},
    };
    const uint64_t embed = 4096 * 3;
    const uint64_t kv = 0;

    auto m = store_mkfs(path, tensors, embed, kv);
    if (!m) { printf("mkfs FAIL: %d\n", (int)m.error()); return 1; }
    printf("mkfs ok\n");

    auto o = store_open(path, O_RDWR);
    if (!o) { printf("open FAIL: %d\n", (int)o.error()); return 1; }
    store * st = *o;
    printf("open ok: tensors=%u embed=%llu\n",
           store_tensor_count(st), (unsigned long long)store_embed_bytes(st));
    if (store_tensor_count(st) != 3) { printf("count mismatch\n"); return 1; }

    // write tensor 0 (512 experts x 100 B real, 512 B padded slot)
    constexpr uint32_t PADDED = 512;   // align_up_512(100)
    constexpr uint32_t REAL = 100;
    std::vector<uint8_t> t0(512 * REAL);
    for (uint32_t e = 0; e < 512; ++e)
        for (int i = 0; i < (int)REAL; ++i)
            t0[e * REAL + i] = (uint8_t)(e * 7 + i);
    ssize_t w = store_write_expert_tensor(st, 0, t0.data());
    if (w != (ssize_t)t0.size()) { printf("write t0 FAIL (%zd)\n", w); return 1; }

    // write shexp tensor
    std::vector<uint8_t> shexp(4096, 0x5A);
    w = store_write_expert_tensor(st, 2, shexp.data());
    if (w != (ssize_t)shexp.size()) { printf("write shexp FAIL (%zd)\n", w); return 1; }
    printf("wrote tensors\n");

    // read back expert slices via store_read_expert (padded slot size)
    for (uint32_t e = 0; e < 512; ++e) {
        std::vector<uint8_t> rd(PADDED, 0);
        ssize_t r = store_read_expert(st, 0, e, rd.data());
        if (r != (ssize_t)PADDED) { printf("read expert %u FAIL (%zd)\n", e, r); return 1; }
        for (uint32_t i = 0; i < REAL; ++i)
            if (rd[i] != (uint8_t)(e * 7 + i)) { printf("data mismatch e=%u @%u\n", e, i); return 1; }
    }
    printf("expert slice readback ok (512 slices, padded slots)\n");

    // shexp readback
    std::vector<uint8_t> sr(4096, 0);
    ssize_t rr = store_read_expert(st, 2, 0, sr.data());
    if (rr != 4096) { printf("shexp read FAIL\n"); return 1; }
    for (int i = 0; i < 4096; ++i)
        if (sr[i] != 0x5A) { printf("shexp mismatch @%d\n", i); return 1; }
    printf("shexp readback ok\n");

    // embed write/read
    std::vector<std::byte> ep(embed, std::byte{0xAB});
    ssize_t ew = store_write_embed(st, 0, std::span<const std::byte>(ep));
    if (ew != (ssize_t)embed) { printf("embed write FAIL\n"); return 1; }
    std::vector<std::byte> er(embed);
    ssize_t eres = store_read_embed(st, 0, std::span<std::byte>(er));
    if (eres != (ssize_t)embed) { printf("embed read FAIL\n"); return 1; }
    for (size_t i = 0; i < embed; ++i)
        if (er[i] != std::byte{0xAB}) { printf("embed mismatch @%zu\n", i); return 1; }
    printf("embed readback ok\n");

    // async path: read expert 3 of tensor 0 (padded slot)
    std::vector<uint8_t> ar(PADDED, 0);
    int aq = store_read_expert_async(st, 0, 3, ar.data());
    if (aq != 0) { printf("async queue FAIL\n"); return 1; }
    ssize_t arres = store_read_expert_wait(st);
    if (arres != (ssize_t)PADDED) { printf("async wait FAIL (%zd)\n", arres); return 1; }
    for (uint32_t i = 0; i < REAL; ++i)
        if (ar[i] != (uint8_t)(3 * 7 + i)) { printf("async mismatch @%u\n", i); return 1; }
    printf("async readback ok\n");

    store_close(st);
    unlink(path);
    printf("ALL PASS\n");
    return 0;
}