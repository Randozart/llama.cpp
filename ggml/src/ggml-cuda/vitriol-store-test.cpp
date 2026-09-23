// vitriol-store-self-test: mkfs a store file, write slots, read back, verify.
// C++26 per docs/cpp26-conventions.md. Run: g++ -std=c++26 ... && ./a.out
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

    const uint32_t n_slots = 16;
    const uint64_t embed = 4096 * 3;
    const uint64_t kv = 0;

    // mkfs
    auto m = store_mkfs(path, n_slots, embed, kv);
    if (!m) { printf("mkfs FAIL: %d\n", (int)m.error()); return 1; }
    printf("mkfs ok\n");

    // open
    auto o = store_open(path, O_RDWR);
    if (!o) { printf("open FAIL: %d\n", (int)o.error()); return 1; }
    store * st = *o;
    printf("open ok: experts=%u slot_size=%u embed=%llu\n",
           store_expert_count(st), store_slot_size(st),
           (unsigned long long)store_embed_bytes(st));
    if (store_expert_count(st) != n_slots) { printf("count mismatch\n"); return 1; }

    // write slot pattern + read back
    std::vector<uint8_t> pat(store_slot_size(st));
    for (uint32_t s = 0; s < n_slots; ++s) {
        for (size_t i = 0; i < pat.size(); ++i) pat[i] = (uint8_t)(s * 31 + i);
        ssize_t w = store_write_slot(st, s, pat.data());
        if (w != (ssize_t)pat.size()) { printf("write slot %u FAIL (%zd)\n", s, w); return 1; }
    }
    printf("wrote %u slots\n", n_slots);

    std::vector<uint8_t> rd(store_slot_size(st));
    for (uint32_t s = 0; s < n_slots; ++s) {
        for (size_t i = 0; i < rd.size(); ++i) rd[i] = 0;
        ssize_t r = store_read_slot(st, s, rd.data());
        if (r != (ssize_t)rd.size()) { printf("read slot %u FAIL (%zd)\n", s, r); return 1; }
        for (size_t i = 0; i < rd.size(); ++i) {
            if (rd[i] != (uint8_t)(s * 31 + i)) { printf("data mismatch slot %u @%zu\n", s, i); return 1; }
        }
    }
    printf("slot readback ok\n");

    // embed write/read
    std::vector<std::byte> ep(embed, std::byte{0xAB});
    ssize_t ew = store_write_embed(st, 0, std::span<const std::byte>(ep));
    if (ew != (ssize_t)embed) { printf("embed write FAIL\n"); return 1; }
    std::vector<std::byte> er(embed);
    ssize_t eres = store_read_embed(st, 0, std::span<std::byte>(er));
    if (eres != (ssize_t)embed) { printf("embed read FAIL\n"); return 1; }
    for (size_t i = 0; i < embed; ++i) {
        if (er[i] != std::byte{0xAB}) { printf("embed mismatch @%zu\n", i); return 1; }
    }
    printf("embed readback ok\n");

    // async path
    std::vector<uint8_t> ar(store_slot_size(st));
    for (size_t i = 0; i < ar.size(); ++i) ar[i] = 0;
    int aq = store_read_slot_async(st, 3, ar.data());
    if (aq != 0) { printf("async queue FAIL\n"); return 1; }
    ssize_t arres = store_read_slot_wait(st);
    if (arres != (ssize_t)ar.size()) { printf("async wait FAIL (%zd)\n", arres); return 1; }
    for (size_t i = 0; i < ar.size(); ++i) {
        if (ar[i] != (uint8_t)(3 * 31 + i)) { printf("async mismatch @%zu\n", i); return 1; }
    }
    printf("async readback ok\n");

    store_close(st);
    unlink(path);
    printf("ALL PASS\n");
    return 0;
}