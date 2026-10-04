// A decoded camera frame on the laptop, and the tiny JSON helpers the
// receiver needs (the node's JSON is flat; no JSON library on the laptop).
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace mocap::rx {

struct Frame {
    std::string camera_id;
    uint64_t frame_seq = 0;
    int64_t stamp_ns = 0;          // start of exposure of the first row, on THIS host's REALTIME scale
    int64_t node_stamp_ns = 0;     // the same, as the node sent it (its REALTIME)
    uint32_t exposure_ns = 0;
    uint32_t line_time_ns = 0;     // row period: the stamp of row r is stamp_ns + r * line_time_ns * (sensor rows / height)
    int width = 0, height = 0;
    int sensor_width = 0, sensor_height = 0;
    bool color = false;
    bool keyframe = false;
    int64_t receive_ns = 0;        // this host's REALTIME when the message was complete
    int64_t decoded_ns = 0;        // ... when the picture was decoded
    std::vector<uint8_t> y;        // width x height, tightly packed
    std::vector<uint8_t> u, v;     // width/2 x height/2 each, colour only
};

// "key":<number> in a flat JSON object; 0 if absent.
inline double json_number(const std::string& j, const std::string& key) {
    auto p = j.find("\"" + key + "\":");
    if (p == std::string::npos) return 0;
    try {
        return std::stod(j.substr(p + key.size() + 3));
    } catch (...) {
        return 0;
    }
}

inline std::string json_string(const std::string& j, const std::string& key) {
    auto p = j.find("\"" + key + "\":\"");
    if (p == std::string::npos) return "";
    p += key.size() + 4;
    auto e = j.find('"', p);
    return e == std::string::npos ? "" : j.substr(p, e - p);
}

}  // namespace mocap::rx
