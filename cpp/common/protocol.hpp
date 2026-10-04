// Wire protocol between a camera node and the laptop: version 1 of
// pi_cam/lan_protocol.py, byte for byte, plus the H.264 frame format.
//
// One TCP stream per node.  Every message is
//   MessageHeader (14 bytes, little-endian)
//   type-specific fixed header (header_len bytes; FrameHeader for frames)
//   JSON metadata (json_len bytes; may be empty)
//   payload (data_len bytes; one H.264 access unit, Annex B, for frames)
#pragma once

#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace mocap::proto {

constexpr uint16_t kMagic = 0x4D43;   // "CM"
constexpr uint8_t kVersion = 1;

enum MessageType : uint8_t {
    kMsgFrame = 1,     // node -> laptop
    kMsgStatus = 2,    // node -> laptop, JSON
    kMsgCommand = 3,   // laptop -> node, JSON
    kMsgAck = 4,       // node -> laptop, JSON answer to a command
    kMsgHello = 5,     // node -> laptop, JSON, first message after connect
};

enum Format : uint8_t {
    kFormatY8 = 0,       // grey, uncompressed
    kFormatJpeg = 1,
    kFormatBayer8 = 2,
    kFormatH264 = 3,     // one H.264 access unit, Annex B
};

enum Flags : uint8_t {
    kFlagScaled = 0x01,     // the whole sensor frame resized to width x height
    kFlagKeyframe = 0x02,   // H.264 IDR with SPS/PPS in front: a receiver can start here
    kFlagColor = 0x04,      // H.264 carries real chroma (otherwise U = V = 128)
};

constexpr int64_t kUnknown = std::numeric_limits<int64_t>::min();

#pragma pack(push, 1)
struct MessageHeader {
    uint16_t magic = kMagic;
    uint8_t version = kVersion;
    uint8_t type = 0;
    uint16_t header_len = 0;
    uint32_t json_len = 0;
    uint32_t data_len = 0;
};

struct FrameHeader {
    char camera_id[16] = {};
    uint64_t frame_seq = 0;          // sensor frame counter on the node
    int64_t stamp_ns = 0;            // common scale: start of exposure of the first sensor row
    uint32_t exposure_ns = 0;
    uint32_t line_time_ns = 0;       // rolling-shutter row period
    uint32_t frame_duration_ns = 0;
    uint16_t row0 = 0, col0 = 0, width = 0, height = 0;
    uint8_t format = 0;
    uint8_t window_index = 0;
    uint8_t window_count = 1;
    uint8_t flags = 0;
    uint16_t sensor_width = 0, sensor_height = 0;
    uint32_t request_id = 0;
    int64_t node_send_ns = 0;        // common-scale time just before send
    int64_t sensor_stamp_ns = 0;     // raw frame-start stamp on the node (Python node: BOOTTIME; C++ node: MONOTONIC)
    int64_t clock_offset_ns = 0;     // node REALTIME - (that clock) at capture
    int64_t ptp_offset_ns = kUnknown;
};
#pragma pack(pop)

static_assert(sizeof(MessageHeader) == 14, "MessageHeader must match lan_protocol.py");
static_assert(sizeof(FrameHeader) == 96, "FrameHeader must match lan_protocol.py");

inline void set_camera_id(FrameHeader& h, const std::string& id) {
    std::memset(h.camera_id, 0, sizeof h.camera_id);
    std::memcpy(h.camera_id, id.data(), std::min(id.size(), sizeof h.camera_id));
}

inline std::string camera_id(const FrameHeader& h) {
    return std::string(h.camera_id, strnlen(h.camera_id, sizeof h.camera_id));
}

// One complete message in a buffer: headers, JSON, then room for the payload.
inline std::vector<uint8_t> encode(MessageType type, const void* fixed, uint16_t fixed_len,
                                   const std::string& json, const void* data, uint32_t data_len) {
    MessageHeader m;
    m.type = type;
    m.header_len = fixed_len;
    m.json_len = static_cast<uint32_t>(json.size());
    m.data_len = data_len;
    std::vector<uint8_t> out(sizeof m + fixed_len + json.size() + data_len);
    uint8_t* p = out.data();
    std::memcpy(p, &m, sizeof m);
    p += sizeof m;
    if (fixed_len) std::memcpy(p, fixed, fixed_len);
    p += fixed_len;
    if (!json.empty()) std::memcpy(p, json.data(), json.size());
    p += json.size();
    if (data_len) std::memcpy(p, data, data_len);
    return out;
}

inline std::vector<uint8_t> encode_json(MessageType type, const std::string& json) {
    return encode(type, nullptr, 0, json, nullptr, 0);
}

}  // namespace mocap::proto
