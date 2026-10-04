// The CM4's hardware H.264 encoder (bcm2835-codec, V4L2 memory-to-memory).
//
// Grey goes in as the Y plane of YUV420 with the chroma held at 128 (prefilled
// once).  The caller writes the picture straight into an input buffer
// (input_plane) and queues it with a frame id; the id rides in the buffer
// timestamp, which the driver copies to the encoded buffer, so every output
// is matched to its input exactly.  Measured on a CM4: the encoder does not
// hold a frame back; 1640x1232 takes ~22 ms alone, ~30 ms while the camera
// and ISP are busy; level 4.2 is refused above ~45 fps at full size (5.1 then).
#pragma once

#include <atomic>
#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace mocap {

struct EncoderSettings {
    int width = 1640;
    int height = 1232;
    double fps = 50.0;
    int bitrate = 15'000'000;
    int gop = 10;               // keyframe period, frames
    bool constant_bitrate = false;
};

class H264Encoder {
public:
    // frame_id, done (BOOTTIME ns), data, size, keyframe
    using OutputFn = std::function<void(uint64_t, int64_t, const uint8_t*, size_t, bool)>;

    H264Encoder(const std::string& device, const EncoderSettings& s, OutputFn on_output);
    ~H264Encoder();
    H264Encoder(const H264Encoder&) = delete;
    H264Encoder& operator=(const H264Encoder&) = delete;

    // An input buffer to fill, or -1 if the encoder still holds all of them.
    int acquire_input();
    uint8_t* input_plane(int index) const { return in_maps_[size_t(index)].ptr; }
    int stride() const { return stride_; }
    int buffer_height() const { return buf_height_; }
    // YUV420 planes of an input buffer: Y (stride), then U and V (stride / 2).
    uint8_t* u_plane(int index) const { return input_plane(index) + size_t(stride_) * size_t(buf_height_); }
    uint8_t* v_plane(int index) const { return u_plane(index) + size_t(stride_ / 2) * size_t(buf_height_ / 2); }
    void queue_input(int index, uint64_t frame_id);
    void release_input(int index);          // give back an acquired buffer unused
    void force_keyframe();
    bool set_bitrate(int bitrate);

    const std::string& card() const { return card_; }

    static std::string find_device();

private:
    struct Map { uint8_t* ptr = nullptr; size_t len = 0; };

    void request_buffers(uint32_t type, int count, std::vector<Map>& maps);
    void queue(uint32_t type, int index, uint32_t bytesused, uint64_t frame_id);
    bool dequeue(uint32_t type, int& index, uint64_t& frame_id, uint32_t& bytesused, uint32_t& flags);
    bool set_ctrl(uint32_t id, int32_t value);
    void reader();

    int fd_ = -1;
    int stride_ = 0, buf_height_ = 0;
    uint32_t sizeimage_ = 0;
    std::string card_;
    std::vector<Map> in_maps_, out_maps_;
    std::mutex lock_;
    std::deque<int> free_in_;
    OutputFn on_output_;
    std::atomic<bool> stop_{false};
    std::thread thread_;
};

}  // namespace mocap
