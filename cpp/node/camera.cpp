#include "camera.hpp"

#include <fcntl.h>
#include <linux/v4l2-subdev.h>
#include <linux/videodev2.h>
#include <poll.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>

namespace mocap {

namespace {

void xioctl(int fd, unsigned long req, void* arg, const char* what) {
    int r;
    do { r = ioctl(fd, req, arg); } while (r < 0 && errno == EINTR);
    if (r < 0) throw std::runtime_error(std::string(what) + ": " + std::strerror(errno));
}

bool set_ctrl(int fd, uint32_t id, int32_t v) {
    v4l2_control c{id, v};
    return ioctl(fd, VIDIOC_S_CTRL, &c) == 0;
}

int32_t get_ctrl(int fd, uint32_t id) {
    v4l2_control c{id, 0};
    xioctl(fd, VIDIOC_G_CTRL, &c, "G_CTRL");
    return c.value;
}

int64_t get_ctrl64(int fd, uint32_t id) {
    v4l2_ext_control c{};
    c.id = id;
    v4l2_ext_controls cs{};
    cs.which = V4L2_CTRL_WHICH_CUR_VAL;
    cs.count = 1;
    cs.controls = &c;
    xioctl(fd, VIDIOC_G_EXT_CTRLS, &cs, "G_EXT_CTRLS");
    return c.value64;
}

// IMX219 analogue gain register: gain = 256 / (256 - code), code 0..232 (1x..10.67x).
int gain_code(float gain) {
    int code = int(std::lround(256.0 - 256.0 / std::max(1.0f, gain)));
    return std::clamp(code, 0, 232);
}

}  // namespace

Camera::Camera(const CameraSettings& s, FrameFn on_frame) : s_(s), on_frame_(std::move(on_frame)) {
    sd_ = open(s.subdev.c_str(), O_RDWR);
    if (sd_ < 0) throw std::runtime_error("open " + s.subdev + ": " + std::strerror(errno));
    vd_ = open(s.video_device.c_str(), O_RDWR | O_NONBLOCK);
    if (vd_ < 0) throw std::runtime_error("open " + s.video_device + ": " + std::strerror(errno) +
                                          " (is camera_node or another camera user running?)");

    // Orientation as libcamera sets it: the overlay declares the sensor mounted
    // at 180 degrees, so both flips go on.  This also makes the mosaic BGGR;
    // after a cold boot the flips are off and the sensor offers RGGB, which is
    // why the format below is refused without this (seen on a bench CM4).
    v4l2_control rot{V4L2_CID_CAMERA_SENSOR_ROTATION, 0};
    bool rotated = ioctl(sd_, VIDIOC_G_CTRL, &rot) == 0 && rot.value == 180;
    if (rotated && !(set_ctrl(sd_, V4L2_CID_HFLIP, 1) && set_ctrl(sd_, V4L2_CID_VFLIP, 1)))
        throw std::runtime_error("sensor refused the flips for its 180-degree mounting");

    v4l2_subdev_format sf{};
    sf.which = V4L2_SUBDEV_FORMAT_ACTIVE;
    sf.format.width = uint32_t(s.width);
    sf.format.height = uint32_t(s.height);
    sf.format.code = MEDIA_BUS_FMT_SBGGR8_1X8;
    sf.format.field = V4L2_FIELD_NONE;
    xioctl(sd_, VIDIOC_SUBDEV_S_FMT, &sf, "sensor format");
    if (sf.format.width != uint32_t(s.width) || sf.format.height != uint32_t(s.height) ||
        sf.format.code != MEDIA_BUS_FMT_SBGGR8_1X8)
        throw std::runtime_error("sensor refused 1640x1232 SBGGR8 (offers code 0x" + [&] {
            char b[16];
            std::snprintf(b, sizeof b, "%x", sf.format.code);
            return std::string(b);
        }() + ")");

    v4l2_format vf{};
    vf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    vf.fmt.pix.width = uint32_t(s.width);
    vf.fmt.pix.height = uint32_t(s.height);
    vf.fmt.pix.pixelformat = V4L2_PIX_FMT_SBGGR8;
    vf.fmt.pix.field = V4L2_FIELD_NONE;
    xioctl(vd_, VIDIOC_S_FMT, &vf, "unicam format");
    stride_ = int(vf.fmt.pix.bytesperline);

    // Row period from the sensor's own pixel rate and line length.
    int64_t pixel_rate = get_ctrl64(sd_, V4L2_CID_PIXEL_RATE);
    int hblank = get_ctrl(sd_, V4L2_CID_HBLANK);
    line_ns_ = 1e9 * double(s.width + hblank) / double(pixel_rate);
    apply_timing(s.fps, s.exposure_us, s.analogue_gain);

    v4l2_requestbuffers rb{};
    rb.count = uint32_t(s.buffer_count);
    rb.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    rb.memory = V4L2_MEMORY_MMAP;
    xioctl(vd_, VIDIOC_REQBUFS, &rb, "REQBUFS");
    for (uint32_t i = 0; i < rb.count; ++i) {
        v4l2_buffer b{};
        b.type = rb.type;
        b.memory = V4L2_MEMORY_MMAP;
        b.index = i;
        xioctl(vd_, VIDIOC_QUERYBUF, &b, "QUERYBUF");
        void* p = mmap(nullptr, b.length, PROT_READ, MAP_SHARED, vd_, b.m.offset);
        if (p == MAP_FAILED) throw std::runtime_error("mmap of a camera buffer failed");
        maps_.emplace_back(static_cast<uint8_t*>(p), b.length);
    }
    char buf[160];
    std::snprintf(buf, sizeof buf, "%dx%d SBGGR8 via unicam (V4L2), %s, stride %d, line %.1f ns, %d lines/frame",
                  s.width, s.height, rotated ? "rotated 180 (flips on)" : "no rotation", stride_, line_ns_,
                  frame_lines_);
    mode_ = buf;
}

Camera::~Camera() {
    stop();
    for (auto& m : maps_) munmap(m.first, m.second);
    v4l2_requestbuffers rb{};
    rb.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    rb.memory = V4L2_MEMORY_MMAP;
    ioctl(vd_, VIDIOC_REQBUFS, &rb);
    close(vd_);
    close(sd_);
}

template <class T>
T Camera::value_at(const std::deque<std::pair<int64_t, T>>& h, int64_t seq) {
    T v = h.empty() ? T{} : h.front().second;
    for (const auto& [from, value] : h) {
        if (from > seq) break;
        v = value;
    }
    return v;
}

void Camera::apply_timing(double fps, int exposure_us, float gain) {
    std::lock_guard<std::mutex> g(controls_lock_);
    // The frame after the last delivered one is already being read out; a
    // write now reaches the sensor's next frame start, and IMX219 latches it
    // one more frame later for exposure and frame length.
    const int64_t next = running_ ? last_sequence_ + 1 : 0;
    const int64_t exp_from = running_ ? next + 2 : 0, gain_from = running_ ? next + 1 : 0;
    if (fps > 0) {
        frame_lines_ = int(std::lround(1e9 / fps / line_ns_));
        if (!set_ctrl(sd_, V4L2_CID_VBLANK, frame_lines_ - s_.height))
            throw std::runtime_error("sensor refused the frame length for this rate");
        frame_duration_ns_ = int64_t(frame_lines_ * line_ns_);
        duration_hist_.emplace_back(exp_from, frame_duration_ns_);
    }
    if (exposure_us > 0) {
        int lines = std::clamp(int(std::lround(exposure_us * 1000.0 / line_ns_)), 1, frame_lines_ - 4);
        set_ctrl(sd_, V4L2_CID_EXPOSURE, lines);
        exposure_ns_ = int64_t(lines * line_ns_);
        exposure_hist_.emplace_back(exp_from, exposure_ns_);
    }
    if (gain > 0) {
        int code = gain_code(gain);
        set_ctrl(sd_, V4L2_CID_ANALOGUE_GAIN, code);
        gain_ = float(256.0 / (256.0 - code));
        gain_hist_.emplace_back(gain_from, gain_);
    }
    for (auto* h : {&exposure_hist_, &duration_hist_})   // keep the last few changes
        while (h->size() > 8) h->pop_front();
    while (gain_hist_.size() > 8) gain_hist_.pop_front();
}

void Camera::start() {
    for (size_t i = 0; i < maps_.size(); ++i) {
        v4l2_buffer b{};
        b.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        b.memory = V4L2_MEMORY_MMAP;
        b.index = uint32_t(i);
        xioctl(vd_, VIDIOC_QBUF, &b, "QBUF");
    }
    int t = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    xioctl(vd_, VIDIOC_STREAMON, &t, "STREAMON");
    running_ = true;
    thread_ = std::thread(&Camera::loop, this);
}

void Camera::stop() {
    if (!running_.exchange(false)) return;
    if (thread_.joinable()) thread_.join();
    int t = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    ioctl(vd_, VIDIOC_STREAMOFF, &t);
}

void Camera::set_controls(std::optional<int> exposure_us, std::optional<float> gain, std::optional<double> fps) {
    apply_timing(fps.value_or(0), exposure_us.value_or(0), gain.value_or(0));
}

void Camera::loop() {
    pollfd p{vd_, POLLIN, 0};
    while (running_) {
        if (poll(&p, 1, 100) <= 0) continue;
        v4l2_buffer b{};
        b.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        b.memory = V4L2_MEMORY_MMAP;
        if (ioctl(vd_, VIDIOC_DQBUF, &b) < 0) continue;
        if (b.flags & V4L2_BUF_FLAG_ERROR) {   // corrupt frame: give it straight back
            ioctl(vd_, VIDIOC_QBUF, &b);
            continue;
        }
        Frame f;
        f.index = int(b.index);
        f.data = maps_[b.index].first;
        f.stride = stride_;
        f.width = s_.width;
        f.height = s_.height;
        f.sequence = b.sequence;
        // unicam stamps the frame-start interrupt: the start of readout of row 0
        // (the frame is in userspace exactly one readout, 11.7 ms, later).
        f.sensor_stamp_ns = int64_t(b.timestamp.tv_sec) * 1'000'000'000 + int64_t(b.timestamp.tv_usec) * 1000;
        {
            std::lock_guard<std::mutex> g(controls_lock_);
            f.exposure_ns = value_at(exposure_hist_, int64_t(f.sequence));
            f.frame_duration_ns = value_at(duration_hist_, int64_t(f.sequence));
            f.gain = value_at(gain_hist_, int64_t(f.sequence));
        }
        if (last_sequence_ >= 0 && int64_t(f.sequence) > last_sequence_ + 1)
            missed_ += uint64_t(int64_t(f.sequence) - last_sequence_ - 1);
        last_sequence_ = int64_t(f.sequence);
        on_frame_(f);
    }
}

void Camera::release(const Frame& f) {
    if (!running_ && f.index < 0) return;
    v4l2_buffer b{};
    b.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    b.memory = V4L2_MEMORY_MMAP;
    b.index = uint32_t(f.index);
    ioctl(vd_, VIDIOC_QBUF, &b);
}

}  // namespace mocap
