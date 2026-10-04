#include "recorder.hpp"

#include <filesystem>
#include <sstream>

namespace mocap::rx {

Recorder::Recorder(std::string dir) : dir_(std::move(dir)) { std::filesystem::create_directories(dir_); }

void Recorder::write(const proto::FrameHeader& h, const uint8_t* data, size_t n, int64_t receive_ns,
                     int64_t stamp_offset_ns) {
    std::string id = proto::camera_id(h);
    bool key = h.flags & proto::kFlagKeyframe;
    std::lock_guard<std::mutex> g(lock_);
    Stream& s = streams_[id];
    if (!s.started) {
        if (!key) return;
        s.h264.open(dir_ + "/" + id + ".h264", std::ios::binary);
        s.jsonl.open(dir_ + "/" + id + ".jsonl");
        s.started = true;
    }
    s.h264.write(reinterpret_cast<const char*>(data), std::streamsize(n));
    std::ostringstream j;
    j << "{\"camera_id\":\"" << id << "\",\"frame_seq\":" << h.frame_seq << ",\"stamp_ns\":" << h.stamp_ns - stamp_offset_ns
      << ",\"node_stamp_ns\":" << h.stamp_ns << ",\"exposure_ns\":" << h.exposure_ns
      << ",\"line_time_ns\":" << h.line_time_ns << ",\"width\":" << h.width << ",\"height\":" << h.height
      << ",\"sensor_width\":" << h.sensor_width << ",\"sensor_height\":" << h.sensor_height
      << ",\"color\":" << ((h.flags & proto::kFlagColor) ? "true" : "false") << ",\"keyframe\":" << (key ? "true" : "false")
      << ",\"offset\":" << s.offset << ",\"bytes\":" << n << ",\"receive_ns\":" << receive_ns << "}\n";
    s.jsonl << j.str();
    s.offset += n;
}

}  // namespace mocap::rx
