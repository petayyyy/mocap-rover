// Dataset recording straight from the wire: each camera's H.264 as it came
// (no re-encoding, ~15 Mbit/s per camera) plus one JSON line per frame with
// its timing.  <dir>/<camera_id>.h264 and <dir>/<camera_id>.jsonl; the file
// starts at the camera's first keyframe so any decoder can read it.
#pragma once

#include <cstdint>
#include <fstream>
#include <map>
#include <mutex>
#include <string>

#include "../common/protocol.hpp"

namespace mocap::rx {

class Recorder {
public:
    explicit Recorder(std::string dir);
    // stamp_offset_ns: node clock minus this host's (0 under PTP).
    void write(const proto::FrameHeader& h, const uint8_t* data, size_t n, int64_t receive_ns,
               int64_t stamp_offset_ns);

private:
    struct Stream {
        std::ofstream h264, jsonl;
        uint64_t offset = 0;
        bool started = false;
    };
    std::string dir_;
    std::mutex lock_;
    std::map<std::string, Stream> streams_;
};

}  // namespace mocap::rx
