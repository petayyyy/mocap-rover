// mocap_camd -- the camera node: one stream per camera.
//
//   sensor (V4L2 unicam, raw SBGGR8 1640x1232, no libcamera)
//     -> grey, demosaiced straight into the encoder's input buffer (OpenCV)
//     -> hardware H.264 (bcm2835-codec)
//     -> TCP, protocol v1 of pi_cam/lan_protocol.py with FORMAT_H264
//
// No frame is copied on the way: the camera buffer is read once by the
// demosaic and released, the encoder output is copied once into the message.
// The sensor runs at the stream rate, so every frame is encoded.  When the
// link or the laptop falls behind, frames are dropped until the next keyframe
// (a P-frame without its reference is useless) and a keyframe is forced.
//
// Commands (JSON in a MSG_COMMAND): status, configure {exposure_us, gain, fps,
// bitrate}, keyframe, full_frame (one lossless grey frame, FORMAT_Y8, for
// calibration).  A status message goes out every second.

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <signal.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <nlohmann/json.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <optional>
#include <thread>
#include <vector>

#include "../common/clock.hpp"
#include "../common/protocol.hpp"
#include "camera.hpp"
#include "h264_encoder.hpp"

using nlohmann::json;
namespace proto = mocap::proto;

namespace {

std::atomic<bool> g_stop{false};
void on_signal(int) { g_stop = true; }

struct Config {
    std::string camera_id = "camera_1";
    int port = 5600;
    double fps = 50.0;
    int exposure_us = 800;
    float analogue_gain = 4.0f;
    int bitrate = 15'000'000;
    double gop_s = 0.2;
    int buffer_count = 4;
    int line_time_ns = 9452;     // IMX219 1640x1232 register model; LED probe pending
    int send_queue = 8;          // messages; beyond it frames are dropped to the next keyframe
    std::string encoder_device;  // found by name if empty
    std::string video_device = "/dev/video0";     // unicam-image
    std::string subdev = "/dev/v4l-subdev0";      // imx219

    static Config load(const std::string& path) {
        Config c;
        if (path.empty()) return c;
        std::ifstream f(path);
        if (!f) throw std::runtime_error("cannot read " + path);
        json j = json::parse(f);
        c.camera_id = j.value("camera_id", c.camera_id);
        c.port = j.value("port", c.port);
        c.fps = j.value("fps", c.fps);
        c.exposure_us = j.value("exposure_us", c.exposure_us);
        c.analogue_gain = j.value("analogue_gain", c.analogue_gain);
        c.bitrate = j.value("bitrate", c.bitrate);
        c.gop_s = j.value("gop_s", c.gop_s);
        c.buffer_count = j.value("buffer_count", c.buffer_count);
        c.line_time_ns = j.value("line_time_ns", c.line_time_ns);
        c.send_queue = j.value("send_queue", c.send_queue);
        c.encoder_device = j.value("encoder_device", c.encoder_device);
        c.video_device = j.value("video_device", c.video_device);
        c.subdev = j.value("subdev", c.subdev);
        return c;
    }
};

struct FrameMeta {
    uint64_t sequence;
    int64_t stamp_ns, sensor_stamp_ns, clock_offset_ns, exposure_ns, frame_duration_ns, in_node_ns;
};

// Values of one status period.
struct Window {
    std::vector<double> exp_to_encoded_ms, encoder_ms, prep_ms, exp_to_send_ms, cvt_ms;
    uint64_t frames = 0, encoded = 0, sent = 0, bytes = 0, keyframes = 0;
    uint64_t dropped_prep = 0, dropped_encoder = 0, dropped_link = 0;
};

class Node {
public:
    explicit Node(Config c) : cfg_(std::move(c)) {}

    void run() {
        mocap::EncoderSettings es;
        es.fps = cfg_.fps;
        es.bitrate = cfg_.bitrate;
        es.gop = std::max(1, int(cfg_.fps * cfg_.gop_s + 0.5));
        std::string dev = cfg_.encoder_device.empty() ? mocap::H264Encoder::find_device() : cfg_.encoder_device;
        encoder_ = std::make_unique<mocap::H264Encoder>(
            dev, es, [this](uint64_t id, int64_t done, const uint8_t* d, size_t n, bool key) {
                on_encoded(id, done, d, n, key);
            });
        mocap::CameraSettings cs;
        cs.video_device = cfg_.video_device;
        cs.subdev = cfg_.subdev;
        cs.fps = cfg_.fps;
        cs.exposure_us = cfg_.exposure_us;
        cs.analogue_gain = cfg_.analogue_gain;
        cs.buffer_count = cfg_.buffer_count;
        camera_ = std::make_unique<mocap::Camera>(cs, [this](const mocap::Frame& f) { on_frame(f); });
        cfg_.line_time_ns = int(camera_->line_time_ns() + 0.5);   // from the sensor's pixel rate
        std::cerr << "camera " << camera_->mode() << ", encoder " << encoder_->card() << " " << dev
                  << ", " << cfg_.fps << " fps, " << cfg_.bitrate / 1e6 << " Mbit/s, GOP " << es.gop << "\n";

        std::thread worker(&Node::worker_loop, this);
        std::thread sender(&Node::sender_loop, this);
        std::thread server(&Node::server_loop, this);
        camera_->start();
        int64_t next_status = mocap::monotonic_ns() + 1'000'000'000;
        while (!g_stop) {
            usleep(50'000);
            if (mocap::monotonic_ns() >= next_status) {
                next_status += 1'000'000'000;
                send_status();
            }
        }
        camera_->stop();
        wake_.notify_all();
        send_cv_.notify_all();
        if (listen_fd_ >= 0) shutdown(listen_fd_, SHUT_RDWR);
        drop_client();
        worker.join();
        sender.join();
        server.join();
        camera_.reset();
        encoder_.reset();
    }

private:
    // The camera thread: keep only the newest frame for the worker.
    void on_frame(const mocap::Frame& f) {
        std::optional<mocap::Frame> old;
        {
            std::lock_guard<std::mutex> g(slot_lock_);
            if (slot_) old = slot_;
            slot_ = f;
            slot_in_ns_ = mocap::monotonic_ns();
        }
        if (old) {
            camera_->release(*old);
            std::lock_guard<std::mutex> g(stats_lock_);
            win_.dropped_prep++;
        }
        wake_.notify_one();
    }

    void worker_loop() {
        while (!g_stop) {
            mocap::Frame f;
            int64_t in_node;
            {
                std::unique_lock<std::mutex> g(slot_lock_);
                wake_.wait_for(g, std::chrono::milliseconds(100), [this] { return slot_.has_value() || g_stop; });
                if (!slot_) continue;
                f = *slot_;
                in_node = slot_in_ns_;
                slot_.reset();
            }
            {
                std::lock_guard<std::mutex> g(stats_lock_);
                win_.frames++;
            }
            int idx = encoder_->acquire_input();
            if (idx < 0) {
                camera_->release(f);
                std::lock_guard<std::mutex> g(stats_lock_);
                win_.dropped_encoder++;
                continue;
            }
            // One pass, on all cores in row bands: each band is demosaiced
            // from the camera buffer (with two rows of margin, so the band
            // edges are exact) into a cached scratch band and written into
            // the encoder buffer.  Both buffers are DMA memory, uncached: each
            // is touched once, by four cores.
            int64_t t0 = mocap::monotonic_ns();
            cv::Mat src(f.height, f.width, CV_8UC1, const_cast<uint8_t*>(f.data), size_t(f.stride));
            cv::Mat dst(f.height, f.width, CV_8UC1, encoder_->input_plane(idx), size_t(encoder_->stride()));
            const int bands = 4, rows = f.height / bands / 2 * 2;
            cv::parallel_for_(cv::Range(0, bands), [&](const cv::Range& r) {
                for (int b = r.start; b < r.end; ++b) {
                    int y0 = b * rows, y1 = b == bands - 1 ? f.height : y0 + rows;
                    int m0 = std::max(0, y0 - 2), m1 = std::min(f.height, y1 + 2);   // even margins keep the Bayer phase
                    cv::Mat raw, gray;
                    src.rowRange(m0, m1).copyTo(raw);
                    // SBGGR8 is OpenCV's BayerBG pattern (same as the Python node and replay).
                    cv::cvtColor(raw, gray, cv::COLOR_BayerBG2GRAY);
                    gray.rowRange(y0 - m0, y0 - m0 + (y1 - y0)).copyTo(dst.rowRange(y0, y1));
                }
            }, bands);
            int64_t t1 = mocap::monotonic_ns();
            {
                std::lock_guard<std::mutex> g(stats_lock_);
                win_.cvt_ms.push_back((t1 - t0) / 1e6);
            }
            if (full_frame_requested_.exchange(false)) send_full_frame(dst, f);
            FrameMeta m;
            m.sequence = f.sequence;
            m.sensor_stamp_ns = f.sensor_stamp_ns;
            m.clock_offset_ns = mocap::realtime_minus_monotonic_ns();
            m.exposure_ns = f.exposure_ns;
            // The unicam stamp is the frame start: the start of readout of row 0
            // (same reference as libcamera's SensorTimestamp); exposure started
            // that much earlier.
            m.stamp_ns = f.sensor_stamp_ns - f.exposure_ns + m.clock_offset_ns;
            m.frame_duration_ns = f.frame_duration_ns;
            m.in_node_ns = in_node;
            camera_->release(f);
            uint64_t id = next_id_++;
            int64_t queued = mocap::monotonic_ns();
            {
                std::lock_guard<std::mutex> g(meta_lock_);
                meta_[id] = {m, queued};
            }
            {
                std::lock_guard<std::mutex> g(stats_lock_);
                win_.prep_ms.push_back((queued - in_node) / 1e6);
            }
            encoder_->queue_input(idx, id);
        }
    }

    void send_full_frame(const cv::Mat& gray, const mocap::Frame& f) {
        proto::FrameHeader h = base_header(f.sequence, f.sensor_stamp_ns - f.exposure_ns + mocap::realtime_minus_monotonic_ns(),
                                           f.sensor_stamp_ns, mocap::realtime_minus_monotonic_ns(), f.exposure_ns,
                                           f.frame_duration_ns);
        h.format = proto::kFormatY8;
        std::vector<uint8_t> pixels(size_t(f.width) * size_t(f.height));
        for (int r = 0; r < f.height; ++r) std::memcpy(&pixels[size_t(r) * size_t(f.width)], gray.ptr(r), size_t(f.width));
        h.node_send_ns = mocap::realtime_ns();
        push_message(proto::encode(proto::kMsgFrame, &h, sizeof h, "", pixels.data(), uint32_t(pixels.size())), true);
    }

    proto::FrameHeader base_header(uint64_t seq, int64_t stamp, int64_t sensor_stamp, int64_t offset,
                                   int64_t exposure, int64_t duration) const {
        proto::FrameHeader h;
        proto::set_camera_id(h, cfg_.camera_id);
        h.frame_seq = seq;
        h.stamp_ns = stamp;
        h.exposure_ns = uint32_t(exposure);
        h.line_time_ns = uint32_t(cfg_.line_time_ns);
        h.frame_duration_ns = uint32_t(duration);
        h.width = 1640;
        h.height = 1232;
        h.sensor_width = 1640;
        h.sensor_height = 1232;
        h.sensor_stamp_ns = sensor_stamp;
        h.clock_offset_ns = offset;
        return h;
    }

    // The encoder's reader thread.
    void on_encoded(uint64_t id, int64_t done, const uint8_t* data, size_t n, bool key) {
        std::pair<FrameMeta, int64_t> mq;
        {
            std::lock_guard<std::mutex> g(meta_lock_);
            auto it = meta_.find(id);
            if (it == meta_.end()) return;
            mq = it->second;
            meta_.erase(meta_.begin(), std::next(it));   // anything older will never come
        }
        const FrameMeta& m = mq.first;
        {
            std::lock_guard<std::mutex> g(stats_lock_);
            win_.encoded++;
            win_.encoder_ms.push_back((done - mq.second) / 1e6);
            win_.exp_to_encoded_ms.push_back((done + m.clock_offset_ns - m.stamp_ns) / 1e6);
            if (key) win_.keyframes++;
        }
        if (waiting_keyframe_ && !key) {
            std::lock_guard<std::mutex> g(stats_lock_);
            win_.dropped_link++;
            return;
        }
        waiting_keyframe_ = false;
        proto::FrameHeader h = base_header(m.sequence, m.stamp_ns, m.sensor_stamp_ns, m.clock_offset_ns,
                                           m.exposure_ns, m.frame_duration_ns);
        h.format = proto::kFormatH264;
        h.flags = key ? proto::kFlagKeyframe : 0;
        push_message(proto::encode(proto::kMsgFrame, &h, sizeof h, "", data, uint32_t(n)), false);
    }

    void push_message(std::vector<uint8_t> msg, bool control) {
        {
            std::lock_guard<std::mutex> g(send_lock_);
            if (client_fd_ < 0) return;
            if (!control && int(send_q_.size()) >= cfg_.send_queue) {
                // The link is behind: everything until the next keyframe is useless.
                send_q_.erase(std::remove_if(send_q_.begin(), send_q_.end(),
                                             [](const auto& q) { return !q.second; }),
                              send_q_.end());
                waiting_keyframe_ = true;
                encoder_->force_keyframe();
                std::lock_guard<std::mutex> s(stats_lock_);
                win_.dropped_link++;
                return;
            }
            send_q_.emplace_back(std::move(msg), control);
        }
        send_cv_.notify_one();
    }

    void sender_loop() {
        while (!g_stop) {
            std::vector<uint8_t> msg;
            int fd;
            {
                std::unique_lock<std::mutex> g(send_lock_);
                send_cv_.wait_for(g, std::chrono::milliseconds(100), [this] { return !send_q_.empty() || g_stop; });
                if (send_q_.empty()) continue;
                msg = std::move(send_q_.front().first);
                send_q_.pop_front();
                fd = client_fd_;
            }
            if (fd < 0) continue;
            // Stamp the send time into a frame header just before it leaves.
            auto* mh = reinterpret_cast<proto::MessageHeader*>(msg.data());
            if (mh->type == proto::kMsgFrame && mh->header_len == sizeof(proto::FrameHeader)) {
                auto* fh = reinterpret_cast<proto::FrameHeader*>(msg.data() + sizeof(proto::MessageHeader));
                int64_t now = mocap::realtime_ns();
                fh->node_send_ns = now;
                std::lock_guard<std::mutex> g(stats_lock_);
                win_.exp_to_send_ms.push_back((now - fh->stamp_ns) / 1e6);
                win_.sent++;
                win_.bytes += mh->data_len;
            }
            size_t off = 0;
            while (off < msg.size()) {
                ssize_t w = send(fd, msg.data() + off, msg.size() - off, MSG_NOSIGNAL);
                if (w <= 0) {
                    if (w < 0 && errno == EINTR) continue;
                    drop_client();
                    break;
                }
                off += size_t(w);
            }
        }
    }

    void drop_client() {
        std::lock_guard<std::mutex> g(send_lock_);
        if (client_fd_ >= 0) {
            shutdown(client_fd_, SHUT_RDWR);
            close(client_fd_);
            client_fd_ = -1;
        }
        send_q_.clear();
    }

    void server_loop() {
        listen_fd_ = socket(AF_INET, SOCK_STREAM, 0);
        int one = 1;
        setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
        sockaddr_in a{};
        a.sin_family = AF_INET;
        a.sin_port = htons(uint16_t(cfg_.port));
        a.sin_addr.s_addr = INADDR_ANY;
        if (bind(listen_fd_, reinterpret_cast<sockaddr*>(&a), sizeof a) || listen(listen_fd_, 1)) {
            std::cerr << "cannot listen on " << cfg_.port << ": " << std::strerror(errno) << "\n";
            g_stop = true;
            return;
        }
        while (!g_stop) {
            int fd = accept(listen_fd_, nullptr, nullptr);
            if (fd < 0) continue;
            setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
            drop_client();
            {
                std::lock_guard<std::mutex> g(send_lock_);
                client_fd_ = fd;
                waiting_keyframe_ = true;   // a new client starts at a keyframe
            }
            encoder_->force_keyframe();
            push_message(proto::encode_json(proto::kMsgHello, hello().dump()), true);
            std::thread(&Node::client_loop, this, fd).detach();
        }
        close(listen_fd_);
    }

    bool read_exact(int fd, void* p, size_t n) {
        auto* b = static_cast<uint8_t*>(p);
        while (n) {
            ssize_t r = recv(fd, b, n, 0);
            if (r <= 0) return false;
            b += r;
            n -= size_t(r);
        }
        return true;
    }

    void client_loop(int fd) {
        while (!g_stop) {
            proto::MessageHeader h;
            if (!read_exact(fd, &h, sizeof h) || h.magic != proto::kMagic) break;
            std::vector<uint8_t> rest(size_t(h.header_len) + h.json_len + h.data_len);
            if (!rest.empty() && !read_exact(fd, rest.data(), rest.size())) break;
            if (h.type != proto::kMsgCommand) continue;
            json cmd = json::parse(rest.begin() + h.header_len, rest.begin() + h.header_len + h.json_len,
                                   nullptr, false);
            push_message(proto::encode_json(proto::kMsgAck, handle(cmd).dump()), true);
        }
        std::lock_guard<std::mutex> g(send_lock_);
        if (client_fd_ == fd) {
            shutdown(fd, SHUT_RDWR);
            close(fd);
            client_fd_ = -1;
            send_q_.clear();
        }
    }

    json handle(const json& cmd) {
        json ans = {{"cmd", cmd.value("cmd", "")}, {"ok", true}, {"realtime_ns", mocap::realtime_ns()}};
        if (cmd.is_discarded()) return {{"ok", false}, {"error", "bad JSON"}};
        std::string c = cmd.value("cmd", "");
        if (c == "status") {
            ans["status"] = last_status_;
        } else if (c == "configure") {
            std::optional<int> e;
            std::optional<float> g;
            std::optional<double> fps;
            if (cmd.contains("exposure_us")) e = cmd["exposure_us"].get<int>();
            if (cmd.contains("gain")) g = cmd["gain"].get<float>();
            if (cmd.contains("fps")) fps = cmd["fps"].get<double>();
            camera_->set_controls(e, g, fps);
            if (cmd.contains("bitrate")) ans["bitrate_applied"] = encoder_->set_bitrate(cmd["bitrate"].get<int>());
        } else if (c == "keyframe") {
            encoder_->force_keyframe();
        } else if (c == "full_frame") {
            full_frame_requested_ = true;
        } else {
            ans["ok"] = false;
            ans["error"] = "unknown command";
        }
        return ans;
    }

    json hello() const {
        return {{"camera_id", cfg_.camera_id}, {"node", "mocap_camd"}, {"format", "h264"},
                {"width", 1640}, {"height", 1232}, {"fps", cfg_.fps}, {"bitrate", cfg_.bitrate},
                {"line_time_ns", cfg_.line_time_ns}, {"stamp_reference", "readout_start_first_row"},
                {"mode", camera_->mode()}};
    }

    static double read_temp() {
        std::ifstream f("/sys/class/thermal/thermal_zone0/temp");
        double t = 0;
        f >> t;
        return t / 1000.0;
    }

    static std::pair<uint64_t, uint64_t> cpu_ticks() {   // idle, total
        std::ifstream f("/proc/stat");
        std::string cpu;
        uint64_t v, idle = 0, total = 0;
        f >> cpu;
        for (int i = 0; i < 8 && f >> v; ++i) {
            total += v;
            if (i == 3 || i == 4) idle += v;
        }
        return {idle, total};
    }

    void send_status() {
        Window w;
        {
            std::lock_guard<std::mutex> g(stats_lock_);
            std::swap(w, win_);
        }
        auto [idle, total] = cpu_ticks();
        double cpu = total > cpu_total_ ? 100.0 * (1.0 - double(idle - cpu_idle_) / double(total - cpu_total_)) : 0;
        cpu_idle_ = idle;
        cpu_total_ = total;
        auto stat = [](const std::vector<double>& v) {
            return json{{"p50", mocap::percentile(v, 50)}, {"p95", mocap::percentile(v, 95)},
                        {"max", v.empty() ? 0.0 : *std::max_element(v.begin(), v.end())}};
        };
        json s = {{"camera_id", cfg_.camera_id},
                  {"frames", w.frames}, {"encoded", w.encoded}, {"sent", w.sent}, {"keyframes", w.keyframes},
                  {"mbit_s", w.bytes * 8 / 1e6},
                  {"sensor_missed_total", camera_->missed()},
                  {"dropped_prep", w.dropped_prep}, {"dropped_encoder", w.dropped_encoder},
                  {"dropped_link", w.dropped_link},
                  {"prep_ms", stat(w.prep_ms)}, {"demosaic_ms", stat(w.cvt_ms)}, {"encoder_ms", stat(w.encoder_ms)},
                  {"exp_to_encoded_ms", stat(w.exp_to_encoded_ms)}, {"exp_to_send_ms", stat(w.exp_to_send_ms)},
                  {"cpu_percent", cpu}, {"temp_c", read_temp()}, {"realtime_ns", mocap::realtime_ns()}};
        last_status_ = s;
        push_message(proto::encode_json(proto::kMsgStatus, s.dump()), true);
    }

    Config cfg_;
    std::unique_ptr<mocap::H264Encoder> encoder_;
    std::unique_ptr<mocap::Camera> camera_;

    std::mutex slot_lock_;
    std::condition_variable wake_;
    std::optional<mocap::Frame> slot_;
    int64_t slot_in_ns_ = 0;

    std::mutex meta_lock_;
    std::map<uint64_t, std::pair<FrameMeta, int64_t>> meta_;
    std::atomic<uint64_t> next_id_{1};
    std::atomic<bool> full_frame_requested_{false};

    std::mutex send_lock_;
    std::condition_variable send_cv_;
    std::deque<std::pair<std::vector<uint8_t>, bool>> send_q_;   // message, control (never dropped)
    int client_fd_ = -1;
    int listen_fd_ = -1;
    std::atomic<bool> waiting_keyframe_{true};

    std::mutex stats_lock_;
    Window win_;
    json last_status_;
    uint64_t cpu_idle_ = 0, cpu_total_ = 0;
};

}  // namespace

int main(int argc, char** argv) {
    std::string config_path;
    std::map<std::string, std::string> over;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--config" && i + 1 < argc) config_path = argv[++i];
        else if (a.rfind("--", 0) == 0 && i + 1 < argc) over[a.substr(2)] = argv[++i];
        else {
            std::cerr << "usage: mocap_camd [--config node.json] [--camera_id ID] [--port N] [--fps F]\n"
                         "                  [--exposure_us N] [--analogue_gain G] [--bitrate BPS]\n";
            return 2;
        }
    }
    try {
        Config c = Config::load(config_path);
        if (over.count("camera_id")) c.camera_id = over["camera_id"];
        if (over.count("port")) c.port = std::stoi(over["port"]);
        if (over.count("fps")) c.fps = std::stod(over["fps"]);
        if (over.count("exposure_us")) c.exposure_us = std::stoi(over["exposure_us"]);
        if (over.count("analogue_gain")) c.analogue_gain = std::stof(over["analogue_gain"]);
        if (over.count("bitrate")) c.bitrate = std::stoi(over["bitrate"]);
        signal(SIGINT, on_signal);
        signal(SIGTERM, on_signal);
        Node(c).run();
    } catch (const std::exception& e) {
        std::cerr << "mocap_camd: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
