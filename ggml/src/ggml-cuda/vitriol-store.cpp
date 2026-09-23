// VITRIOL Store - implementation. C++26 per docs/cpp26-conventions.md.
//
// Raw-device layout:
//   [0x0000] super 4K (2 copies at +0 and +4096)
//   [index]  dense store_slot_desc array (expert_count entries)
//   [expert] expert_count * 2 MiB slots
//   [embed]  embed_bytes sequential
//   [kv]     kv_ring_bytes (reserved, P3)
//
// I/O: pread/pwrite with O_DIRECT; io_uring async path with pread fallback.

#include "vitriol-store.h"

#include <fcntl.h>
#include <unistd.h>
#include <cerrno>
#include <cstring>
#include <cstdio>
#include <memory>

#include <liburing.h>

namespace vitriol {

// Index zone sits right after the two superblock copies.
constexpr uint64_t STORE_INDEX_BASE = 2 * VITRIOL_STORE_BLK;

uint32_t store_crc32(std::span<const std::byte> data) {
    uint32_t crc = 0xFFFFFFFFu;
    for (auto b : data) {
        crc ^= static_cast<uint32_t>(b);
        for (int i = 0; i < 8; ++i) {
            crc = (crc >> 1) ^ (0xEDB88320u & -(crc & 1));
        }
    }
    return ~crc;
}

struct store {
    int fd = -1;
    uint32_t flags = 0;
    store_super super{};
    std::unique_ptr<store_slot_desc[]> index;

    // io_uring state
    bool uring_ok = false;
    struct io_uring ring;
    uint32_t async_slot = 0;

    ~store() {
        if (uring_ok) {
            io_uring_queue_exit(&ring);
        }
        if (fd >= 0) {
            close(fd);
        }
    }
};

static bool super_valid(const store_super & s) {
    if (s.magic != VITRIOL_STORE_MAGIC || s.version != VITRIOL_STORE_VERSION) {
        return false;
    }
    std::span<const std::byte> bytes(
        reinterpret_cast<const std::byte *>(&s), VITRIOL_STORE_BLK - 4);
    return store_crc32(bytes) == s.crc32;
}

static store_error write_super_at(int fd, uint64_t off, const store_super & s) {
    const std::byte * p = reinterpret_cast<const std::byte *>(&s);
    if (pwrite(fd, p, VITRIOL_STORE_BLK, static_cast<off_t>(off)) !=
        static_cast<ssize_t>(VITRIOL_STORE_BLK)) {
        return store_error::io_error;
    }
    return store_error::none;
}

static store_error write_super(int fd, store_super & s) {
    // CRC over first 4092 bytes; crc32 field itself excluded.
    s.crc32 = 0;
    std::span<const std::byte> bytes(
        reinterpret_cast<const std::byte *>(&s), VITRIOL_STORE_BLK - 4);
    s.crc32 = store_crc32(bytes);
    if (auto e = write_super_at(fd, 0, s); e != store_error::none) return e;
    return write_super_at(fd, VITRIOL_STORE_BLK, s);  // mirror copy
}

std::expected<store *, store_error> store_open(const char * path, int flags)
    pre(path != nullptr)
    pre(path[0] != '\0') {
    int fd = open(path, flags | O_DIRECT);
    if (fd < 0) {
        return std::unexpected(store_error::open_failed);
    }

    auto st = std::make_unique<store>();
    st->fd = fd;
    st->flags = flags;

    store_super s0, s1;
    if (pread(fd, &s0, VITRIOL_STORE_BLK, 0) != VITRIOL_STORE_BLK ||
        pread(fd, &s1, VITRIOL_STORE_BLK, VITRIOL_STORE_BLK) != VITRIOL_STORE_BLK) {
        return std::unexpected(store_error::io_error);
    }
    if (!super_valid(s0) && !super_valid(s1)) {
        fprintf(stderr, "store_open: magic0=%08x ver0=%u crc0=%08x calc0=%08x | magic1=%08x ver1=%u crc1=%08x calc1=%08x\n",
            s0.magic, s0.version, s0.crc32,
            store_crc32(std::span<const std::byte>(reinterpret_cast<const std::byte*>(&s0), VITRIOL_STORE_BLK-4)),
            s1.magic, s1.version, s1.crc32,
            store_crc32(std::span<const std::byte>(reinterpret_cast<const std::byte*>(&s1), VITRIOL_STORE_BLK-4)));
        return std::unexpected(store_error::not_a_store);
    }
    // Prefer the valid copy; if both valid, primary wins.
    st->super = super_valid(s0) ? s0 : s1;

    // Load index zone.
    const uint64_t index_bytes =
        static_cast<uint64_t>(st->super.expert_count) * sizeof(store_slot_desc);
    if (index_bytes > 0) {
        auto idx = std::make_unique<store_slot_desc[]>(st->super.expert_count);
        const std::byte * ip = reinterpret_cast<const std::byte *>(idx.get());
        const uint64_t index_crc_base = index_bytes;
        uint64_t stored_crc = 0;
        if (pread(fd, const_cast<std::byte *>(ip), index_bytes,
                  static_cast<off_t>(STORE_INDEX_BASE)) !=
                static_cast<ssize_t>(index_bytes) ||
            pread(fd, &stored_crc, sizeof(stored_crc),
                  static_cast<off_t>(STORE_INDEX_BASE + index_bytes)) !=
                static_cast<ssize_t>(sizeof(stored_crc))) {
            return std::unexpected(store_error::io_error);
        }
        std::span<const std::byte> idx_bytes(ip, index_crc_base);
        if (store_crc32(idx_bytes) != stored_crc) {
            return std::unexpected(store_error::index_corrupt);
        }
        st->index = std::move(idx);
    }

    // Set up io_uring (best-effort; fall back to pread).
    if (io_uring_queue_init(8, &st->ring, 0) == 0) {
        st->uring_ok = true;
    }

    return st.release();
}

void store_close(store * st) {
    delete st;
}

uint32_t store_expert_count(const store * st) {
    return st->super.expert_count;
}

uint32_t store_slot_size(const store * st) {
    return static_cast<uint32_t>(st->super.expert_stride);
}

uint64_t store_embed_bytes(const store * st) {
    return st->super.embed_bytes;
}

uint64_t store_slot_offset(const store * st, uint32_t slot)
    pre(st != nullptr)
    pre(slot < st->super.expert_count) {
    return st->super.expert_base + static_cast<uint64_t>(slot) * st->super.expert_stride;
}

ssize_t store_read_slot(store * st, uint32_t slot, void * dst) {
    const uint64_t off = store_slot_offset(st, slot);
    const size_t len = static_cast<size_t>(st->super.expert_stride);
    return pread(st->fd, dst, len, static_cast<off_t>(off));
}

ssize_t store_read_embed(store * st, uint64_t offset, std::span<std::byte> dst) {
    if (st->super.embed_base == 0 || dst.empty()) {
        return -1;
    }
    return pread(st->fd, dst.data(), dst.size(),
                 static_cast<off_t>(st->super.embed_base + offset));
}

int store_read_slot_async(store * st, uint32_t slot, void * dst) {
    if (!st->uring_ok) {
        // pread fallback: synchronous read.
        return store_read_slot(st, slot, dst) < 0 ? -1 : 0;
    }
    st->async_slot = slot;
    struct io_uring_sqe * sqe = io_uring_get_sqe(&st->ring);
    if (!sqe) {
        return -1;
    }
    const uint64_t off = store_slot_offset(st, slot);
    io_uring_prep_read(sqe, st->fd, dst, st->super.expert_stride, off);
    io_uring_submit(&st->ring);
    return 0;
}

ssize_t store_read_slot_wait(store * st) {
    if (!st->uring_ok) {
        return -1;
    }
    struct io_uring_cqe * cqe = nullptr;
    if (io_uring_wait_cqe(&st->ring, &cqe) != 0) {
        return -1;
    }
    const int res = cqe->res;
    io_uring_cqe_seen(&st->ring, cqe);
    if (res < 0) {
        return -1;
    }
    // Copy from registered fixed buffer into caller-provided location is done
    // by the caller using store's async buffer. This API copies out.
    return res;
}

std::expected<void, store_error> store_mkfs(
    const char * path,
    uint32_t expert_count,
    uint64_t embed_bytes,
    uint64_t kv_bytes)
    pre(path != nullptr) {
    int fd = open(path, O_RDWR | O_DIRECT | O_CREAT, 0600);
    if (fd < 0) {
        return std::unexpected(store_error::open_failed);
    }

    store_super s{};
    s.magic = VITRIOL_STORE_MAGIC;
    s.version = VITRIOL_STORE_VERSION;
    s.expert_count = expert_count;
    s.expert_stride = VITRIOL_STORE_SLOT;
    s.expert_base = STORE_INDEX_BASE +
        static_cast<uint64_t>(expert_count) * sizeof(store_slot_desc) +
        sizeof(uint64_t);  // + index CRC trailer
    // align expert zone to 1 MiB
    s.expert_base = (s.expert_base + (1024*1024 - 1)) & ~(uint64_t)(1024*1024 - 1);
    s.embed_base = s.expert_base +
        static_cast<uint64_t>(expert_count) * s.expert_stride;
    s.embed_bytes = embed_bytes;
    if (kv_bytes > 0) {
        s.kv_base = s.embed_base + embed_bytes;
        s.kv_ring_bytes = kv_bytes;
        s.kv_cell_size = VITRIOL_STORE_KVCELL;
    }

    if (auto e = write_super(fd, s); e != store_error::none) {
        close(fd);
        return std::unexpected(e);
    }

    // Preallocate: extend file to full length so later pwrite/read is O(1).
    // kv_ring_bytes=0 means no KV zone; total = end of embed zone.
    const uint64_t total = s.kv_ring_bytes > 0
        ? (s.kv_base + s.kv_ring_bytes)
        : (s.embed_base + s.embed_bytes);
    if (ftruncate(fd, static_cast<off_t>(total)) != 0) {
        close(fd);
        return std::unexpected(store_error::io_error);
    }

    // Write a valid (empty) index: count=0 means "mkfs, no slots yet" is
    // handled by caller via store_write_slot. We write an index CRC trailer.
    uint64_t idx_crc = 0;  // empty index -> CRC of zero bytes = 0
    if (expert_count > 0) {
        std::unique_ptr<store_slot_desc[]> idx(new store_slot_desc[expert_count]());
        const uint64_t idx_bytes =
            static_cast<uint64_t>(expert_count) * sizeof(store_slot_desc);
        const std::byte * ip = reinterpret_cast<const std::byte *>(idx.get());
        if (pwrite(fd, ip, idx_bytes, static_cast<off_t>(STORE_INDEX_BASE)) !=
                static_cast<ssize_t>(idx_bytes)) {
            close(fd);
            return std::unexpected(store_error::io_error);
        }
        idx_crc = store_crc32(std::span<const std::byte>(ip, idx_bytes));
    }
    if (pwrite(fd, &idx_crc, sizeof(idx_crc),
               static_cast<off_t>(STORE_INDEX_BASE +
                   static_cast<uint64_t>(expert_count) * sizeof(store_slot_desc))) !=
        static_cast<ssize_t>(sizeof(idx_crc))) {
        close(fd);
        return std::unexpected(store_error::io_error);
    }

    close(fd);
    return {};
}

ssize_t store_write_slot(store * st, uint32_t slot, const void * src) {
    const uint64_t off = store_slot_offset(st, slot);
    return pwrite(st->fd, src, static_cast<size_t>(st->super.expert_stride),
                  static_cast<off_t>(off));
}

ssize_t store_write_embed(store * st, uint64_t offset, std::span<const std::byte> src) {
    if (st->super.embed_base == 0 || src.empty()) {
        return -1;
    }
    return pwrite(st->fd, src.data(), src.size(),
                  static_cast<off_t>(st->super.embed_base + offset));
}

}  // namespace vitriol