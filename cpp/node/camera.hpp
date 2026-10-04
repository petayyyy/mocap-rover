// The IMX219 straight from the CSI receiver (unicam) through V4L2, no libcamera.
//
// Measured on a CM4 (2026-10-04): a raw 1640x1232 frame reaches userspace
// 11.7 ms after its timestamp -- the readout itself -- while libcamera hands
// the same frame over ~22 ms after it (it waits for the ISP statistics even
// with auto exposure off).  The sensor is set through its V4L2 subdevice:
// format, frame length (VBLANK), exposure (lines) and analogue gain.
//
// A frame points into the mapped buffer; the consumer releases it (the
// buffer is queued back) once it has made its grey picture.
#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

namespace mocap {

struct CameraSettings {
    int width = 1640;
    int height = 1232;
    double fps = 50.0;            // the sensor runs at the stream rate
    int exposure_us = 800;
    float analogue_gain = 4.0f;
    int buffer_count = 4;
    std::string video_device = "/dev/video0";       // unicam-image
    std::string subdev = "/dev/v4l-subdev0";        // imx219
};

struct Frame {
    int index = -1;                    // V4L2 buffer index
    const uint8_t* data = nullptr;     // raw SBGGR8
    int stride = 0;
    int width = 0, height = 0;
    uint64_t sequence = 0;
    int64_t sensor_stamp_ns = 0;       // CLOCK_MONOTONIC: frame start = start of readout of row 0
    int64_t exposure_ns = 0;
    int64_t frame_duration_ns = 0;
    float gain = 0;
};

class Camera {
public:
    using FrameFn = std::function<void(const Frame&)>;

    Camera(const CameraSettings& s, FrameFn on_frame);
    ~Camera();

    void start();
    void stop();
    void release(const Frame& f);      // queue the buffer back
    void set_controls(std::optional<int> exposure_us, std::optional<float> gain,
                      std::optional<double> fps);

    std::string mode() const { return mode_; }
    uint64_t missed() const { return missed_; }
    double line_time_ns() const { return line_ns_; }
    // What the sensor actually runs: exposure in whole rows, gain in register steps.
    int exposure_us() const { return int(exposure_ns_ / 1000); }
    float gain() const { return gain_; }
    double fps() const { return frame_duration_ns_ ? 1e9 / double(frame_duration_ns_) : 0; }
    int max_exposure_us() const { return int((frame_lines_ - 4) * line_ns_ / 1000); }

private:
    void apply_timing(double fps, int exposure_us, float gain);
    void loop();

    CameraSettings s_;
    FrameFn on_frame_;
    int vd_ = -1, sd_ = -1;
    int stride_ = 0;
    double line_ns_ = 0;
    int frame_lines_ = 0;
    std::atomic<int64_t> exposure_ns_{0}, frame_duration_ns_{0};
    std::atomic<float> gain_{0};
    std::vector<std::pair<uint8_t*, size_t>> maps_;
    std::string mode_;
    std::mutex controls_lock_;
    std::atomic<bool> running_{false};
    std::atomic<uint64_t> missed_{0};
    int64_t last_sequence_ = -1;
    std::thread thread_;
};

}  // namespace mocap
