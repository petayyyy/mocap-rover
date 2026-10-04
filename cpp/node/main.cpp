// mocap_camd -- the camera node: one stream per camera.
//
//   sensor (V4L2 unicam, raw SBGGR8 1640x1232, no libcamera)
//     -> the whole frame, resized to the stream size, grey or colour,
//        written straight into the encoder's input buffer (frame_prep)
//     -> hardware H.264 (bcm2835-codec)
//     -> TCP, protocol v1 of pi_cam/lan_protocol.py with FORMAT_H264
//
// The sensor runs at the stream rate, so every frame is encoded.  When the
// link or the laptop falls behind, frames are dropped until the next keyframe
// (a P-frame without its reference is useless) and a keyframe is forced.
//
// Settings (all changeable while running, from the TCP command "configure" or
// the web page): width x height (always the whole frame, resized), color,
// fps, exposure_us, gain, bitrate.  A size, colour or rate change rebuilds the
// encoder and starts the stream again at a keyframe.  "save" writes them to
// the config file.  Other commands: status, keyframe, full_frame (one lossless
// grey 1640x1232 frame, FORMAT_Y8, for calibration).
//
// Web page on http_port (8080): live picture, histogram, region statistics
// and the settings -- for tuning exposure on the arena itself.

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <signal.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <deque>
#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <nlohmann/json.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <optional>
#include <thread>
#include <vector>

#include "../common/clock.hpp"
#include "../common/protocol.hpp"
#include "camera.hpp"
#include "frame_prep.hpp"
#include "h264_encoder.hpp"
#include "http_server.hpp"
#include "ptp_monitor.hpp"
#include "web_page.hpp"

using nlohmann::json;
namespace proto = mocap::proto;

namespace {

std::atomic<bool> g_stop{false};
void on_signal(int) { g_stop = true; }

// What the web page and "configure" can change while running.
struct Settings {
    int width = 1640, height = 1232;   // the whole sensor frame resized to this, never a crop
    bool color = false;
    double fps = 50.0;
    int exposure_us = 800;
    float gain = 4.0f;
    int bitrate = 15'000'000;
    double gop_s = 0.2;
    float red_gain = 1.0f, blue_gain = 1.0f;   // white balance of the colour stream

    json to_json() const {
        return {{"width", width}, {"height", height}, {"color", color}, {"fps", fps},
                {"exposure_us", exposure_us}, {"gain", gain}, {"bitrate", bitrate}, {"gop_s", gop_s},
                {"red_gain", red_gain}, {"blue_gain", blue_gain}};
    }
    void from_json(const json& j) {
        width = j.value("width", width);
        height = j.value("height", height);
        color = j.value("color", color);
        fps = j.value("fps", fps);
        exposure_us = j.value("exposure_us", exposure_us);
        gain = j.value("gain", j.value("analogue_gain", gain));
        bitrate = j.value("bitrate", bitrate);
        gop_s = j.value("gop_s", gop_s);
        red_gain = j.value("red_gain", red_gain);
        blue_gain = j.value("blue_gain", blue_gain);
    }
};

struct Config {
    std::string camera_id = "camera_1";
    int port = 5600;
    int http_port = 8080;
    Settings settings;
    int buffer_count = 4;
    int line_time_ns = 9452;     // IMX219 1640x1232 register model; LED probe pending
    int send_queue = 8;          // messages; beyond it frames are dropped to the next keyframe
    std::string encoder_device;  // found by name if empty
    bool ptp_enabled = true;      // poll pmc for the offset from the PTP master
    double ptp_period_s = 5.0;
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
        c.http_port = j.value("http_port", c.http_port);
        c.settings.from_json(j);
        c.buffer_count = j.value("buffer_count", c.buffer_count);
        c.line_time_ns = j.value("line_time_ns", c.line_time_ns);
        c.send_queue = j.value("send_queue", c.send_queue);
        c.encoder_device = j.value("encoder_device", c.encoder_device);
        c.video_device = j.value("video_device", c.video_device);
        c.ptp_enabled = j.value("ptp_enabled", c.ptp_enabled);
        c.ptp_period_s = j.value("ptp_period_s", c.ptp_period_s);
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
    Node(Config c, std::string config_path) : cfg_(std::move(c)), config_path_(std::move(config_path)) {}

    void run() {
        Settings& st = cfg_.settings;
        validate(st);
        encoder_device_ = cfg_.encoder_device.empty() ? mocap::H264Encoder::find_device() : cfg_.encoder_device;
        mocap::CameraSettings cs;
        cs.video_device = cfg_.video_device;
        cs.subdev = cfg_.subdev;
        cs.fps = st.fps;
        cs.exposure_us = st.exposure_us;
        cs.analogue_gain = st.gain;
        cs.buffer_count = cfg_.buffer_count;
        camera_ = std::make_unique<mocap::Camera>(cs, [this](const mocap::Frame& f) { on_frame(f); });
        cfg_.line_time_ns = int(camera_->line_time_ns() + 0.5);   // from the sensor's pixel rate
        build_encoder(st);
        if (cfg_.ptp_enabled) ptp_ = std::make_unique<mocap::PtpMonitor>(cfg_.ptp_period_s);
        std::cerr << "camera " << camera_->mode() << "\nstream " << describe(st) << ", encoder "
                  << encoder_->card() << " " << encoder_device_ << "\nweb page on :" << cfg_.http_port << "\n";

        std::thread worker(&Node::worker_loop, this);
        std::thread sender(&Node::sender_loop, this);
        std::thread server(&Node::server_loop, this);
        mocap::HttpServer web(cfg_.http_port, [this](const mocap::HttpRequest& r) { return on_http(r); });
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
            if (rebuild_wanted_) {   // let build_encoder() take the pipeline first
                camera_->release(f);
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
                continue;
            }
            std::unique_lock<std::mutex> pipe(pipe_lock_);   // the encoder may be rebuilt meanwhile
            int idx = encoder_->acquire_input();
            if (idx < 0) {
                pipe.unlock();
                camera_->release(f);
                std::lock_guard<std::mutex> g(stats_lock_);
                win_.dropped_encoder++;
                continue;
            }
            int64_t t0 = mocap::monotonic_ns();
            mocap::EncoderPlanes planes{encoder_->input_plane(idx), encoder_->stride(), encoder_->u_plane(idx),
                                        encoder_->v_plane(idx), encoder_->stride() / 2};
            bool want_preview = t0 < preview_wanted_until_ && t0 - last_preview_ns_ > 150'000'000;
            cv::Mat preview;
            prep_.process(f.data, f.stride, f.width, f.height, out_, planes, want_preview ? &preview : nullptr);
            int64_t t1 = mocap::monotonic_ns();
            {
                std::lock_guard<std::mutex> g(stats_lock_);
                win_.cvt_ms.push_back((t1 - t0) / 1e6);
            }
            if (want_preview) {
                last_preview_ns_ = t1;
                std::lock_guard<std::mutex> g(preview_lock_);
                preview_ = preview;
                preview_jpeg_.clear();
            }
            if (full_frame_requested_.exchange(false)) {
                cv::Mat gray;
                mocap::FramePrep::full_gray(f.data, f.stride, f.width, f.height, gray);
                send_full_frame(gray, f);
            }
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
            if (force_keyframe_soon_.exchange(false)) encoder_->force_keyframe();
            encoder_->queue_input(idx, id);
        }
    }

    // ------------------------------------------------------------ settings

    static void validate(Settings& s) {
        s.width = std::clamp(s.width / 2 * 2, 160, 1640);
        s.height = std::clamp(s.height / 2 * 2, 120, 1232);
        s.fps = std::clamp(s.fps, 1.0, 83.0);
        s.exposure_us = std::max(10, s.exposure_us);
        s.gain = std::clamp(s.gain, 1.0f, 10.67f);
        s.bitrate = std::clamp(s.bitrate, 250'000, 50'000'000);
        s.gop_s = std::clamp(s.gop_s, 0.02, 10.0);
        s.red_gain = std::clamp(s.red_gain, 0.25f, 4.0f);
        s.blue_gain = std::clamp(s.blue_gain, 0.25f, 4.0f);
    }

    static std::string describe(const Settings& s) {
        return std::to_string(s.width) + "x" + std::to_string(s.height) + (s.color ? " colour" : " grey") + ", " +
               std::to_string(int(s.fps + 0.5)) + " fps, " + std::to_string(s.bitrate / 1'000'000) + " Mbit/s";
    }

    // (Re)build the encoder for a size, colour mode or rate; the worker holds
    // pipe_lock_ while it uses the encoder, so this waits for the current frame.
    void build_encoder(const Settings& s) {
        mocap::EncoderSettings es;
        es.width = s.width;
        es.height = s.height;
        es.fps = s.fps;
        es.bitrate = s.bitrate;
        es.gop = std::max(1, int(s.fps * s.gop_s + 0.5));
        rebuild_wanted_ = true;   // the worker yields instead of re-taking the lock frame after frame
        std::lock_guard<std::mutex> pipe(pipe_lock_);
        rebuild_wanted_ = false;
        encoder_.reset();   // joins its reader: no output of the old stream comes after this
        {
            std::lock_guard<std::mutex> g(meta_lock_);
            meta_.clear();
        }
        encoder_ = std::make_unique<mocap::H264Encoder>(
            encoder_device_, es, [this](uint64_t id, int64_t done, const uint8_t* d, size_t n, bool key) {
                on_encoded(id, done, d, n, key);
            });
        out_ = {s.width, s.height, s.color, s.red_gain, s.blue_gain};
        waiting_keyframe_ = true;   // the receiver restarts its decoder at the new stream's first keyframe
    }

    // Shared by the web page and the TCP "configure" command.
    json apply(const json& j_in) {
        std::string message;
        bool error = false;
        json j = j_in;
        if (j.contains("wb_roi")) {   // [x, y, w, h] normalised; grey-world over the whole frame if empty
            auto wb = white_balance(j["wb_roi"]);
            if (!wb) return state_json({}, "баланс белого: нет цветного кадра (включите цвет)", true);
            j["red_gain"] = wb->first;
            j["blue_gain"] = wb->second;
            message = "баланс белого выставлен";
        }
        {
            std::lock_guard<std::mutex> g(settings_lock_);
            error = !apply_locked(j, message);
        }
        return state_json({}, message, error);
    }

    // New red/blue gains that make the region grey: the preview already carries
    // the current gains, so the correction multiplies them.
    std::optional<std::pair<float, float>> white_balance(const json& roi) {
        cv::Mat bgr;
        {
            std::lock_guard<std::mutex> g(preview_lock_);
            if (preview_.empty() || preview_.channels() != 3) return std::nullopt;
            bgr = preview_.clone();
        }
        cv::Rect r(0, 0, bgr.cols, bgr.rows);
        if (roi.is_array() && roi.size() == 4) {
            r = cv::Rect(int(roi[0].get<double>() * bgr.cols), int(roi[1].get<double>() * bgr.rows),
                         std::max(1, int(roi[2].get<double>() * bgr.cols)),
                         std::max(1, int(roi[3].get<double>() * bgr.rows)));
            r &= cv::Rect(0, 0, bgr.cols, bgr.rows);
        }
        cv::Scalar m = cv::mean(bgr(r));
        if (m[0] < 1 || m[2] < 1) return std::nullopt;
        std::lock_guard<std::mutex> g(settings_lock_);
        return std::make_pair(float(cfg_.settings.red_gain * m[1] / m[2]), float(cfg_.settings.blue_gain * m[1] / m[0]));
    }

    bool apply_locked(const json& j, std::string& message) {
        Settings old = cfg_.settings, s = old;
        s.from_json(j);
        validate(s);
        try {
            if (s.exposure_us != old.exposure_us || s.gain != old.gain || s.fps != old.fps)
                camera_->set_controls(s.exposure_us, s.gain, s.fps != old.fps ? std::optional<double>(s.fps)
                                                                                : std::nullopt);
            if (s.width != old.width || s.height != old.height || s.color != old.color || s.fps != old.fps ||
                s.gop_s != old.gop_s) {
                build_encoder(s);
                message = "поток перестроен: " + describe(s);
            } else {
                std::lock_guard<std::mutex> pipe(pipe_lock_);
                if (s.bitrate != old.bitrate) encoder_->set_bitrate(s.bitrate);
                out_.red_gain = s.red_gain;
                out_.blue_gain = s.blue_gain;
            }
            cfg_.settings = s;
        } catch (const std::exception& e) {
            message = std::string("не применено: ") + e.what();
            try {   // back to what worked
                camera_->set_controls(old.exposure_us, old.gain, old.fps);
                build_encoder(old);
            } catch (...) {
            }
            return false;
        }
        return true;
    }


    json save() {
        std::string path = config_path_.empty() ? "camd.json" : config_path_;
        json j;
        {
            std::ifstream in(path);
            if (in) j = json::parse(in, nullptr, false);
            if (j.is_discarded() || !j.is_object()) j = json::object();
        }
        {
            std::lock_guard<std::mutex> g(settings_lock_);
            for (auto& [k, v] : cfg_.settings.to_json().items()) j[k] = v;
        }
        j["camera_id"] = cfg_.camera_id;
        std::ofstream out(path);
        if (!out) return {{"error", true}, {"message", "не могу записать " + path}};
        out << j.dump(2) << "\n";
        return {{"error", false}, {"message", "сохранено в " + path}};
    }

    // ------------------------------------------------------------ web page

    static json image_stats(const cv::Mat& gray) {
        int hist[256] = {};
        for (int y = 0; y < gray.rows; ++y) {
            const uint8_t* r = gray.ptr(y);
            for (int x = 0; x < gray.cols; ++x) hist[r[x]]++;
        }
        double n = double(gray.total()), sum = 0;
        int p5 = -1, p95 = -1, p99 = -1;
        double acc = 0;
        for (int v = 0; v < 256; ++v) {
            sum += double(v) * hist[v];
            acc += hist[v];
            if (p5 < 0 && acc >= 0.05 * n) p5 = v;
            if (p95 < 0 && acc >= 0.95 * n) p95 = v;
            if (p99 < 0 && acc >= 0.99 * n) p99 = v;
        }
        int sat = 0, dark = 0;
        for (int v = 250; v < 256; ++v) sat += hist[v];
        for (int v = 0; v <= 5; ++v) dark += hist[v];
        json bins = json::array();
        for (int b = 0; b < 64; ++b) bins.push_back(hist[4 * b] + hist[4 * b + 1] + hist[4 * b + 2] + hist[4 * b + 3]);
        return {{"mean", sum / n}, {"p5", p5}, {"p95", p95}, {"p99", p99},
                {"saturated_pct", 100.0 * sat / n}, {"dark_pct", 100.0 * dark / n},
                {"contrast", p95 + p5 > 0 ? double(p95 - p5) / double(p95 + p5) : 0.0}, {"hist", bins}};
    }

    json state_json(const std::map<std::string, std::string>& query, const std::string& message, bool error) {
        preview_wanted_until_ = mocap::monotonic_ns() + 3'000'000'000LL;
        json s;
        {
            std::lock_guard<std::mutex> g(settings_lock_);
            s["settings"] = cfg_.settings.to_json();
        }
        s["camera_id"] = cfg_.camera_id;
        s["mode"] = camera_->mode();
        s["actual"] = {{"exposure_us", camera_->exposure_us()}, {"gain", camera_->gain()}, {"fps", camera_->fps()}};
        s["max_exposure_us"] = camera_->max_exposure_us();
        s["status"] = last_status_;
        {
            std::lock_guard<std::mutex> g(send_lock_);
            s["client"] = client_fd_ >= 0;
        }
        cv::Mat gray;
        {
            std::lock_guard<std::mutex> g(preview_lock_);
            if (!preview_.empty()) {
                if (preview_.channels() == 3) cv::cvtColor(preview_, gray, cv::COLOR_BGR2GRAY);
                else gray = preview_.clone();
            }
        }
        if (!gray.empty()) {
            s["image"] = image_stats(gray);
            auto it = query.find("roi");
            if (it != query.end()) {
                double x, y, w, h;
                if (std::sscanf(it->second.c_str(), "%lf,%lf,%lf,%lf", &x, &y, &w, &h) == 4) {
                    cv::Rect r(int(x * gray.cols), int(y * gray.rows), std::max(1, int(w * gray.cols)),
                               std::max(1, int(h * gray.rows)));
                    r &= cv::Rect(0, 0, gray.cols, gray.rows);
                    if (r.area() > 0) {
                        json roi = image_stats(gray(r));
                        roi.erase("hist");
                        double k;
                        {
                            std::lock_guard<std::mutex> g(settings_lock_);
                            k = double(cfg_.settings.width) / gray.cols;
                        }
                        roi["w"] = int(r.width * k);   // in stream pixels
                        roi["h"] = int(r.height * k);
                        s["roi"] = roi;
                    }
                }
            }
        }
        if (!message.empty()) s["message"] = message;
        s["error"] = error;
        return s;
    }

    mocap::HttpResponse on_http(const mocap::HttpRequest& r) {
        mocap::HttpResponse res;
        if (r.method == "GET" && r.path == "/") {
            res.content_type = "text/html; charset=utf-8";
            res.body = mocap::kWebPage;
        } else if (r.method == "GET" && r.path == "/preview.jpg") {
            preview_wanted_until_ = mocap::monotonic_ns() + 3'000'000'000LL;
            std::lock_guard<std::mutex> g(preview_lock_);
            if (preview_jpeg_.empty() && !preview_.empty())
                cv::imencode(".jpg", preview_, preview_jpeg_, {cv::IMWRITE_JPEG_QUALITY, 80});
            res.content_type = "image/jpeg";
            res.body.assign(preview_jpeg_.begin(), preview_jpeg_.end());
            if (res.body.empty()) res.status = 404;
        } else if (r.path == "/api/state") {
            res.body = state_json(r.query, "", false).dump();
        } else if (r.method == "POST" && r.path == "/api/apply") {
            json j = json::parse(r.body, nullptr, false);
            res.body = (j.is_object() ? apply(j) : json{{"error", true}, {"message", "bad JSON"}}).dump();
        } else if (r.method == "POST" && r.path == "/api/save") {
            res.body = save().dump();
        } else if (r.method == "POST" && r.path == "/api/keyframe") {
            std::lock_guard<std::mutex> pipe(pipe_lock_);
            encoder_->force_keyframe();
            res.body = "{}";
        } else {
            res.status = 404;
            res.body = "{}";
        }
        return res;
    }

    void send_full_frame(const cv::Mat& gray, const mocap::Frame& f) {
        proto::FrameHeader h = base_header(f.sequence, f.sensor_stamp_ns - f.exposure_ns + mocap::realtime_minus_monotonic_ns(),
                                           f.sensor_stamp_ns, mocap::realtime_minus_monotonic_ns(), f.exposure_ns,
                                           f.frame_duration_ns);
        h.format = proto::kFormatY8;
        h.width = uint16_t(f.width);
        h.height = uint16_t(f.height);
        h.flags = 0;
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
        h.width = uint16_t(out_.width);
        h.height = uint16_t(out_.height);
        h.flags = (out_.width != 1640 || out_.height != 1232 ? proto::kFlagScaled : 0) |
                  (out_.color ? proto::kFlagColor : 0);
        h.sensor_width = 1640;
        h.sensor_height = 1232;
        h.sensor_stamp_ns = sensor_stamp;
        h.clock_offset_ns = offset;
        h.ptp_offset_ns = ptp_ ? ptp_->offset_ns() : proto::kUnknown;
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
        if (key) h.flags |= proto::kFlagKeyframe;
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
                force_keyframe_soon_ = true;   // asked outside the send lock, by the worker
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
            force_keyframe_soon_ = true;
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
            json st = apply(cmd);
            ans["ok"] = !st.value("error", false);
            ans["settings"] = st["settings"];
            ans["actual"] = st["actual"];
            if (st.contains("message")) ans["message"] = st["message"];
            push_message(proto::encode_json(proto::kMsgHello, hello().dump()), true);
        } else if (c == "save") {
            ans.update(save());
        } else if (c == "keyframe") {
            force_keyframe_soon_ = true;
        } else if (c == "full_frame") {
            full_frame_requested_ = true;
        } else {
            ans["ok"] = false;
            ans["error"] = "unknown command";
        }
        return ans;
    }

    json hello() {
        json s;
        {
            std::lock_guard<std::mutex> g(settings_lock_);
            s = cfg_.settings.to_json();
        }
        s.update({{"camera_id", cfg_.camera_id}, {"node", "mocap_camd"}, {"format", "h264"},
                  {"sensor_width", 1640}, {"sensor_height", 1232}, {"line_time_ns", cfg_.line_time_ns},
                  {"stamp_reference", "readout_start_first_row"}, {"resize", "whole frame, never a crop"},
                  {"mode", camera_->mode()}});
        return s;
    }

    json ptp_json() const {
        if (!ptp_) return {{"state", "disabled"}};
        int64_t off = ptp_->offset_ns();
        return {{"state", ptp_->state()}, {"age_s", ptp_->age_s()},
                {"offset_ns", off == proto::kUnknown ? json(nullptr) : json(off)}};
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
                  {"cpu_percent", cpu}, {"temp_c", read_temp()}, {"realtime_ns", mocap::realtime_ns()},
                  {"ptp", ptp_json()}};
        last_status_ = s;
        push_message(proto::encode_json(proto::kMsgStatus, s.dump()), true);
    }

    Config cfg_;
    std::string config_path_, encoder_device_;
    std::mutex settings_lock_;
    std::mutex pipe_lock_;                // encoder_, out_, prep_
    std::unique_ptr<mocap::H264Encoder> encoder_;
    mocap::OutputFormat out_;
    mocap::FramePrep prep_;
    std::unique_ptr<mocap::Camera> camera_;
    std::unique_ptr<mocap::PtpMonitor> ptp_;
    std::atomic<bool> force_keyframe_soon_{false};
    std::atomic<bool> rebuild_wanted_{false};

    std::mutex preview_lock_;
    cv::Mat preview_;
    std::vector<uint8_t> preview_jpeg_;
    std::atomic<int64_t> preview_wanted_until_{0};
    int64_t last_preview_ns_ = 0;

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
            std::cerr << "usage: mocap_camd [--config camd.json] [--camera_id ID] [--port N] [--http_port N]\n"
                         "                  [--width W --height H] [--color 0|1] [--fps F] [--exposure_us N]\n"
                         "                  [--gain G] [--bitrate BPS]\n";
            return 2;
        }
    }
    try {
        Config c = Config::load(config_path);
        if (over.count("camera_id")) c.camera_id = over["camera_id"];
        if (over.count("port")) c.port = std::stoi(over["port"]);
        if (over.count("http_port")) c.http_port = std::stoi(over["http_port"]);
        Settings& s = c.settings;
        if (over.count("width")) s.width = std::stoi(over["width"]);
        if (over.count("height")) s.height = std::stoi(over["height"]);
        if (over.count("color")) s.color = over["color"] == "1" || over["color"] == "true";
        if (over.count("fps")) s.fps = std::stod(over["fps"]);
        if (over.count("exposure_us")) s.exposure_us = std::stoi(over["exposure_us"]);
        if (over.count("gain")) s.gain = std::stof(over["gain"]);
        if (over.count("analogue_gain")) s.gain = std::stof(over["analogue_gain"]);
        if (over.count("bitrate")) s.bitrate = std::stoi(over["bitrate"]);
        signal(SIGINT, on_signal);
        signal(SIGTERM, on_signal);
        signal(SIGPIPE, SIG_IGN);
        Node(c, config_path).run();
    } catch (const std::exception& e) {
        std::cerr << "mocap_camd: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
