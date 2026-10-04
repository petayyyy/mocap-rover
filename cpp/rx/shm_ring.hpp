// Decoded frames of one camera in shared memory, for readers in other
// processes (the Python web page, the Python tract while it is being ported).
//
// /dev/shm/mocap_<camera_id>, little-endian:
//   ShmHeader (4096 bytes reserved)
//   slots x { ShmSlot (128 bytes) | Y (max_w*max_h) | U | V (max_w/2*max_h/2 each) }
// The writer fills slot (write_count % slots) under a sequence lock (odd while
// writing), then bumps write_count.  A reader takes the newest slot, copies it
// and checks the sequence did not move.  Reader: localization_contracts/shm_frames.py.
#pragma once

#include <atomic>
#include <cstdint>
#include <string>

#include "frame.hpp"

namespace mocap::rx {

constexpr uint32_t kShmMagic = 0x4853434D;   // "MCSH"
constexpr uint32_t kShmVersion = 1;
constexpr size_t kShmHeaderBytes = 4096;
constexpr size_t kShmSlotHeaderBytes = 128;

#pragma pack(push, 1)
struct ShmHeader {
    uint32_t magic, version, slots, max_width, max_height;
    uint32_t reserved0;
    uint64_t slot_stride;          // bytes from one slot to the next
    uint64_t write_count;          // slots published so far (atomic)
    char camera_id[32];
};

struct ShmSlot {
    uint64_t seq_lock;             // odd while the writer is inside
    uint64_t frame_seq;
    int64_t stamp_ns;              // start of exposure of row 0, this host's REALTIME
    int64_t node_stamp_ns;
    uint32_t exposure_ns, line_time_ns;
    uint32_t width, height, sensor_width, sensor_height;
    uint32_t color, keyframe;
    int64_t receive_ns, decoded_ns, published_ns;
};
#pragma pack(pop)

static_assert(sizeof(ShmSlot) <= kShmSlotHeaderBytes, "slot header too big");

class ShmRing {
public:
    ShmRing(const std::string& camera_id, int slots = 4, int max_width = 1640, int max_height = 1232);
    ~ShmRing();
    void publish(const Frame& f);
    const std::string& path() const { return path_; }

private:
    std::string name_, path_;
    uint8_t* base_ = nullptr;
    size_t size_ = 0;
    ShmHeader* header_ = nullptr;
};

}  // namespace mocap::rx
