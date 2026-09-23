// VITRIOL Store - raw Optane 900P zone for expert/embed/KV streaming
//
// C++26 per docs/cpp26-conventions.md. Contracts (pre/post) on public API;
// std::expected at boundaries; scalar returns on hot read path.
//
// No-filesystem fixed-slot layout on a raw partition. O(1) addressing:
// offset = zone_base + slot * stride. RAM-resident index rebuilt from a
// persisted index zone at open (1 superblock + 1 index read, then O(1)).

#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>
#include <sys/types.h>

namespace vitriol {

constexpr uint32_t VITRIOL_STORE_MAGIC = 0x56495453;  // 'VITS'
constexpr uint32_t VITRIOL_STORE_VERSION = 1;
constexpr uint32_t VITRIOL_STORE_BLK = 4096;          // superblock size
constexpr uint64_t VITRIOL_STORE_SLOT = 2 * 1024 * 1024;  // 2 MiB expert slot
constexpr uint64_t VITRIOL_STORE_KVCELL = 2 * 1024 * 1024; // 2 MiB KV cell

constexpr uint32_t VITRIOL_STORE_Z_EXPERT = 1;
constexpr uint32_t VITRIOL_STORE_Z_EMBED  = 2;
constexpr uint32_t VITRIOL_STORE_Z_KV     = 3;

// On-disk superblock. Compile-time layout invariants enforced by static_assert
// below (always free, always on).
struct alignas(4096) store_super {
    uint32_t magic;
    uint32_t version;
    uint32_t expert_count;       // number of live expert slots
    uint32_t _pad0;
    uint64_t expert_base;        // byte offset of EXPERT zone
    uint64_t expert_stride;      // bytes between slots (== VITRIOL_STORE_SLOT)
    uint64_t embed_base;         // byte offset of EMBED zone (0 if absent)
    uint64_t embed_bytes;        // total embed tensor bytes
    uint64_t kv_base;            // byte offset of KV ring (0 if absent)
    uint64_t kv_ring_bytes;      // total KV ring size (0 if absent)
    uint64_t kv_cell_size;
    uint32_t _pad[1005];         // pad fields 72..4092 so crc32 lands at 4092
    uint32_t crc32;              // LAST field: CRC over bytes 0..4091
};

static_assert(sizeof(store_super) == 4096, "superblock must be 4 KiB");
static_assert(alignof(store_super) == 4096, "superblock must be 4KiB aligned");
static_assert(offsetof(store_super, crc32) == 4092, "crc32 must be the last field at 4092");

// Error type for boundary APIs.
enum class store_error {
    none = 0,
    open_failed,
    not_a_store,      // bad magic / version
    corrupt,          // CRC mismatch
    index_corrupt,
    io_error,
    out_of_bounds,    // slot >= expert_count
    not_implemented,  // KV ring not yet wired
};

struct store;
struct store_impl;

// Persistent per-slot descriptor (index zone). Dense array [0, expert_count).
struct store_slot_desc {
    uint32_t layer;     // layer index (for LRU key reconstruction)
    uint32_t expert;    // expert index within layer
    uint64_t offset;    // file offset where slot lives (= expert_base + slot*stride)
};

// ---- boundary API (std::expected) ------------------------------------------

// Open store on raw device or plain file. O_RDONLY or O_RDWR.
// Reads superblock + index zone; errno-style error via store_error.
std::expected<store *, store_error> store_open(const char * path, int flags)
    pre(path != nullptr)
    pre(path[0] != '\0');

void store_close(store * st);

uint32_t store_expert_count(const store * st) pre(st != nullptr);
uint32_t store_slot_size(const store * st) pre(st != nullptr);
uint64_t store_embed_bytes(const store * st) pre(st != nullptr);

// ---- hot read path (scalar returns, no expected) ---------------------------

// Read one expert slot into dst (slot_size bytes). Returns bytes read, or -1.
// Caller MUST ensure dst points to >= slot_size bytes. This is the hot MoE
// path - scalar return only.
ssize_t store_read_slot(store * st, uint32_t slot, void * dst)
    pre(st != nullptr)
    pre(dst != nullptr)
    pre(slot < store_expert_count(st));

// Read a slice of the embed tensor. Sequential bulk path.
// Returns bytes read, or -1.
ssize_t store_read_embed(store * st, uint64_t offset, std::span<std::byte> dst)
    pre(st != nullptr)
    pre(offset + dst.size() <= store_embed_bytes(st));

// Asynchronous read of one expert slot via io_uring. Queues the op; wait
// with store_read_slot_wait. Returns 0 queued, -1 error.
int store_read_slot_async(store * st, uint32_t slot, void * dst)
    pre(st != nullptr)
    pre(dst != nullptr)
    pre(slot < store_expert_count(st));

// Wait for the outstanding async op. Returns bytes read or -1.
ssize_t store_read_slot_wait(store * st) pre(st != nullptr);

// ---- mkfs ------------------------------------------------------------------

// Format a raw device/file. Computes zones from expert_count + embed_bytes.
// kv_bytes reserved (P3). Returns expected<store_error>; 0 (none) on success.
std::expected<void, store_error> store_mkfs(
    const char * path,
    uint32_t expert_count,
    uint64_t embed_bytes,
    uint64_t kv_bytes)
    pre(path != nullptr);

// Write one expert slot's bytes (used by mkfs tool to populate).
ssize_t store_write_slot(store * st, uint32_t slot, const void * src)
    pre(st != nullptr)
    pre(src != nullptr)
    pre(slot < store_expert_count(st));

// Write the embed tensor bytes (mkfs tool).
ssize_t store_write_embed(store * st, uint64_t offset, std::span<const std::byte> src)
    pre(st != nullptr)
    pre(offset + src.size() <= store_embed_bytes(st));

// ---- diagnostics ------------------------------------------------------------

// CRC32 (IEEE) over a byte span. Used by superblock + index validation.
uint32_t store_crc32(std::span<const std::byte> data);

}  // namespace vitriol