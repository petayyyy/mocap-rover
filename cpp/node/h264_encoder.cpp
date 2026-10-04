#include "h264_encoder.hpp"

#include <fcntl.h>
#include <linux/videodev2.h>
#include <poll.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>

#include "../common/clock.hpp"

namespace mocap {

namespace {

constexpr uint32_t kOut = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE;   // raw frames in
constexpr uint32_t kCap = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;  // H.264 out

void xioctl(int fd, unsigned long req, void* arg, const char* what) {
    int r;
    do { r = ioctl(fd, req, arg); } while (r < 0 && errno == EINTR);
    if (r < 0) throw std::runtime_error(std::string(what) + ": " + std::strerror(errno));
}

// The lowest of 4.2 / 5.1 that covers the stream's macroblock rate: the CM4's
// firmware refuses STREAMON (ESRCH) when the level is too low for the stream.
int level_for(int w, int h, double fps) {
    double mbps = double((w + 15) / 16) * double((h + 15) / 16) * fps;
    return mbps <= 522240 ? V4L2_MPEG_VIDEO_H264_LEVEL_4_2 : V4L2_MPEG_VIDEO_H264_LEVEL_5_1;
}

}  // namespace

std::string H264Encoder::find_device() {
    namespace fs = std::filesystem;
    for (const auto& e : fs::directory_iterator("/sys/class/video4linux")) {
        std::ifstream f(e.path() / "name");
        std::string name;
        std::getline(f, name);
        if (name == "bcm2835-codec-encode") return "/dev/" + e.path().filename().string();
    }
    return "/dev/video11";
}

H264Encoder::H264Encoder(const std::string& device, const EncoderSettings& s, OutputFn on_output)
    : on_output_(std::move(on_output)) {
    fd_ = open(device.c_str(), O_RDWR | O_NONBLOCK);
    if (fd_ < 0) throw std::runtime_error("open " + device + ": " + std::strerror(errno));
    v4l2_capability cap{};
    xioctl(fd_, VIDIOC_QUERYCAP, &cap, "QUERYCAP");
    card_ = reinterpret_cast<const char*>(cap.card);

    v4l2_format fmt{};
    fmt.type = kOut;
    fmt.fmt.pix_mp.width = uint32_t(s.width);
    fmt.fmt.pix_mp.height = uint32_t(s.height);
    fmt.fmt.pix_mp.pixelformat = V4L2_PIX_FMT_YUV420;
    fmt.fmt.pix_mp.num_planes = 1;
    fmt.fmt.pix_mp.plane_fmt[0].bytesperline = uint32_t(s.width);
    xioctl(fd_, VIDIOC_S_FMT, &fmt, "S_FMT output");
    stride_ = int(fmt.fmt.pix_mp.plane_fmt[0].bytesperline);
    buf_height_ = int(fmt.fmt.pix_mp.height);
    sizeimage_ = fmt.fmt.pix_mp.plane_fmt[0].sizeimage;
    if (fmt.fmt.pix_mp.pixelformat != V4L2_PIX_FMT_YUV420 || stride_ < s.width)
        throw std::runtime_error("encoder refused YU12 at this size");

    fmt = {};
    fmt.type = kCap;
    fmt.fmt.pix_mp.width = uint32_t(s.width);
    fmt.fmt.pix_mp.height = uint32_t(s.height);
    fmt.fmt.pix_mp.pixelformat = V4L2_PIX_FMT_H264;
    fmt.fmt.pix_mp.num_planes = 1;
    fmt.fmt.pix_mp.plane_fmt[0].sizeimage = uint32_t(std::max(512 * 1024, s.width * s.height));
    xioctl(fd_, VIDIOC_S_FMT, &fmt, "S_FMT capture");

    v4l2_streamparm parm{};
    parm.type = kOut;
    parm.parm.output.timeperframe.numerator = 1000;
    parm.parm.output.timeperframe.denominator = uint32_t(s.fps * 1000.0 + 0.5);
    ioctl(fd_, VIDIOC_S_PARM, &parm);   // rate control hint; not fatal

    set_ctrl(V4L2_CID_MPEG_VIDEO_BITRATE_MODE, s.constant_bitrate ? V4L2_MPEG_VIDEO_BITRATE_MODE_CBR
                                                                   : V4L2_MPEG_VIDEO_BITRATE_MODE_VBR);
    set_ctrl(V4L2_CID_MPEG_VIDEO_BITRATE, s.bitrate);
    set_ctrl(V4L2_CID_MPEG_VIDEO_H264_I_PERIOD, s.gop);
    set_ctrl(V4L2_CID_MPEG_VIDEO_REPEAT_SEQ_HEADER, 1);   // SPS/PPS on every keyframe
    set_ctrl(V4L2_CID_MPEG_VIDEO_H264_PROFILE, V4L2_MPEG_VIDEO_H264_PROFILE_HIGH);
    set_ctrl(V4L2_CID_MPEG_VIDEO_H264_LEVEL, level_for(s.width, s.height, s.fps));

    request_buffers(kOut, 2, in_maps_);
    for (size_t i = 0; i < in_maps_.size(); ++i) {
        size_t y = size_t(stride_) * size_t(buf_height_);
        std::memset(in_maps_[i].ptr + y, 128, sizeimage_ - y);   // neutral chroma
        free_in_.push_back(int(i));
    }
    request_buffers(kCap, 4, out_maps_);
    for (size_t i = 0; i < out_maps_.size(); ++i) queue(kCap, int(i), 0, 0);
    for (uint32_t t : {kOut, kCap}) {
        int type = int(t);
        xioctl(fd_, VIDIOC_STREAMON, &type, "STREAMON");
    }
    thread_ = std::thread(&H264Encoder::reader, this);
}

H264Encoder::~H264Encoder() {
    stop_ = true;
    if (thread_.joinable()) thread_.join();
    for (uint32_t t : {kOut, kCap}) {
        int type = int(t);
        ioctl(fd_, VIDIOC_STREAMOFF, &type);
    }
    for (auto* maps : {&in_maps_, &out_maps_})
        for (auto& m : *maps) munmap(m.ptr, m.len);
    for (uint32_t t : {kOut, kCap}) {
        v4l2_requestbuffers req{};
        req.type = t;
        req.memory = V4L2_MEMORY_MMAP;
        ioctl(fd_, VIDIOC_REQBUFS, &req);
    }
    close(fd_);
}

bool H264Encoder::set_ctrl(uint32_t id, int32_t value) {
    v4l2_control c{id, value};
    return ioctl(fd_, VIDIOC_S_CTRL, &c) == 0;
}

bool H264Encoder::set_bitrate(int bitrate) { return set_ctrl(V4L2_CID_MPEG_VIDEO_BITRATE, bitrate); }

void H264Encoder::force_keyframe() { set_ctrl(V4L2_CID_MPEG_VIDEO_FORCE_KEY_FRAME, 1); }

void H264Encoder::request_buffers(uint32_t type, int count, std::vector<Map>& maps) {
    v4l2_requestbuffers req{};
    req.count = uint32_t(count);
    req.type = type;
    req.memory = V4L2_MEMORY_MMAP;
    xioctl(fd_, VIDIOC_REQBUFS, &req, "REQBUFS");
    for (uint32_t i = 0; i < req.count; ++i) {
        v4l2_plane plane{};
        v4l2_buffer buf{};
        buf.index = i;
        buf.type = type;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.length = 1;
        buf.m.planes = &plane;
        xioctl(fd_, VIDIOC_QUERYBUF, &buf, "QUERYBUF");
        void* p = mmap(nullptr, plane.length, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, plane.m.mem_offset);
        if (p == MAP_FAILED) throw std::runtime_error(std::string("mmap: ") + std::strerror(errno));
        maps.push_back({static_cast<uint8_t*>(p), plane.length});
    }
}

void H264Encoder::queue(uint32_t type, int index, uint32_t bytesused, uint64_t frame_id) {
    v4l2_plane plane{};
    plane.bytesused = bytesused;
    plane.length = uint32_t((type == kOut ? in_maps_ : out_maps_)[size_t(index)].len);
    v4l2_buffer buf{};
    buf.index = uint32_t(index);
    buf.type = type;
    buf.memory = V4L2_MEMORY_MMAP;
    buf.length = 1;
    buf.m.planes = &plane;
    buf.timestamp.tv_sec = time_t(frame_id / 1'000'000);
    buf.timestamp.tv_usec = suseconds_t(frame_id % 1'000'000);
    xioctl(fd_, VIDIOC_QBUF, &buf, "QBUF");
}

bool H264Encoder::dequeue(uint32_t type, int& index, uint64_t& frame_id, uint32_t& bytesused,
                          uint32_t& flags) {
    v4l2_plane plane{};
    v4l2_buffer buf{};
    buf.type = type;
    buf.memory = V4L2_MEMORY_MMAP;
    buf.length = 1;
    buf.m.planes = &plane;
    if (ioctl(fd_, VIDIOC_DQBUF, &buf) < 0) return false;   // EAGAIN: nothing ready
    index = int(buf.index);
    frame_id = uint64_t(buf.timestamp.tv_sec) * 1'000'000 + uint64_t(buf.timestamp.tv_usec);
    bytesused = plane.bytesused;
    flags = buf.flags;
    return true;
}

int H264Encoder::acquire_input() {
    std::lock_guard<std::mutex> g(lock_);
    if (free_in_.empty()) return -1;
    int i = free_in_.front();
    free_in_.pop_front();
    return i;
}

void H264Encoder::release_input(int index) {
    std::lock_guard<std::mutex> g(lock_);
    free_in_.push_back(index);
}

void H264Encoder::queue_input(int index, uint64_t frame_id) { queue(kOut, index, sizeimage_, frame_id); }

void H264Encoder::reader() {
    pollfd p{fd_, POLLIN | POLLOUT, 0};
    while (!stop_) {
        int r = poll(&p, 1, 100);
        if (r <= 0) continue;
        bool any = false;
        int index;
        uint64_t id;
        uint32_t bytes, flags;
        while (dequeue(kCap, index, id, bytes, flags)) {
            any = true;
            int64_t done = monotonic_ns();
            if (bytes && on_output_)
                on_output_(id, done, out_maps_[size_t(index)].ptr, bytes, flags & V4L2_BUF_FLAG_KEYFRAME);
            queue(kCap, index, 0, 0);
        }
        while (dequeue(kOut, index, id, bytes, flags)) {
            any = true;
            std::lock_guard<std::mutex> g(lock_);
            free_in_.push_back(index);
        }
        if (!any) usleep(500);   // POLLERR with nothing queued: do not spin
    }
}

}  // namespace mocap
