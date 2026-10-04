#include "node_receiver.hpp"

#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/frame.h>
}

#include <algorithm>
#include <cstring>

#include "../common/clock.hpp"

namespace mocap::rx {

namespace {

constexpr int64_t kSecond = 1'000'000'000;

int connect_to(const std::string& addr) {
    auto colon = addr.rfind(':');
    std::string host = addr.substr(0, colon);
    std::string port = colon == std::string::npos ? "5600" : addr.substr(colon + 1);
    addrinfo hints{}, *res = nullptr;
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    if (getaddrinfo(host.c_str(), port.c_str(), &hints, &res)) return -1;
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    timeval tv{2, 0};
    setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof tv);
    if (connect(fd, res->ai_addr, res->ai_addrlen)) {
        close(fd);
        fd = -1;
    }
    freeaddrinfo(res);
    if (fd >= 0) {
        int one = 1;
        setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
        timeval rt{3, 0};   // a silent node for 3 s counts as gone
        setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &rt, sizeof rt);
    }
    return fd;
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

void copy_plane(const uint8_t* src, int linesize, int w, int h, std::vector<uint8_t>& dst) {
    dst.resize(size_t(w) * size_t(h));
    for (int r = 0; r < h; ++r) std::memcpy(&dst[size_t(r) * size_t(w)], src + size_t(r) * size_t(linesize), size_t(w));
}

}  // namespace

NodeReceiver::NodeReceiver(std::string addr, bool ptp, FrameFn on_frame, PacketFn on_packet)
    : addr_(std::move(addr)), ptp_(ptp), on_frame_(std::move(on_frame)), on_packet_(std::move(on_packet)) {
    stats_.addr = addr_;
    thread_ = std::thread(&NodeReceiver::run, this);
    ping_thread_ = std::thread(&NodeReceiver::pinger, this);
}

NodeReceiver::~NodeReceiver() {
    stop_ = true;
    int fd = fd_.exchange(-1);
    if (fd >= 0) shutdown(fd, SHUT_RDWR);
    if (thread_.joinable()) thread_.join();
    if (ping_thread_.joinable()) ping_thread_.join();
}

bool NodeReceiver::command(const std::string& json) {
    int fd = fd_;
    if (fd < 0) return false;
    auto msg = proto::encode_json(proto::kMsgCommand, json);
    std::lock_guard<std::mutex> g(send_lock_);
    size_t off = 0;
    while (off < msg.size()) {
        ssize_t w = send(fd, msg.data() + off, msg.size() - off, MSG_NOSIGNAL);
        if (w <= 0) return false;
        off += size_t(w);
    }
    return true;
}

NodeStats NodeReceiver::take_stats() {
    std::lock_guard<std::mutex> g(stats_lock_);
    NodeStats out = stats_;
    stats_.arrive_ms.clear();
    stats_.decoded_ms.clear();
    stats_.decode_ms.clear();
    stats_.node_ms.clear();
    stats_.frames = stats_.keyframes = stats_.lost = stats_.decode_fail = stats_.bytes = 0;
    out.offset_ns = offset_ns_;
    {
        std::lock_guard<std::mutex> c(clock_lock_);
        int64_t best = INT64_MAX;
        for (auto& s : samples_) best = std::min(best, std::get<1>(s));
        out.offset_unc_ns = samples_.empty() ? -1 : best / 2;
    }
    return out;
}

void NodeReceiver::pinger() {
    while (!stop_) {
        for (int i = 0; i < 20 && !stop_; ++i) usleep(100'000);
        if (ptp_ || fd_ < 0) continue;
        ping_sent_ns_ = realtime_ns();
        command("{\"cmd\":\"status\"}");
    }
}

void NodeReceiver::on_ack(const std::string& json, int64_t receive_ns) {
    int64_t sent = ping_sent_ns_.exchange(0);
    double node = json_number(json, "realtime_ns");
    if (!sent || node <= 0 || ptp_) return;
    int64_t rtt = receive_ns - sent;
    int64_t offset = int64_t(node) - (sent + receive_ns) / 2;   // node clock minus ours
    std::lock_guard<std::mutex> g(clock_lock_);
    samples_.emplace_back(receive_ns, rtt, offset);
    while (!samples_.empty() && receive_ns - std::get<0>(samples_.front()) > 60 * kSecond) samples_.pop_front();
    auto best = std::min_element(samples_.begin(), samples_.end(),
                                 [](const auto& a, const auto& b) { return std::get<1>(a) < std::get<1>(b); });
    offset_ns_ = std::get<2>(*best);
    clock_known_ = true;
}

void NodeReceiver::run() {
    while (!stop_) {
        int fd = connect_to(addr_);
        if (fd < 0) {
            for (int i = 0; i < 10 && !stop_; ++i) usleep(100'000);
            continue;
        }
        fd_ = fd;
        {
            std::lock_guard<std::mutex> g(stats_lock_);
            stats_.connected = true;
        }
        // A few quick round trips first, so the first frames are already on our scale.
        if (!ptp_)
            for (int i = 0; i < 5; ++i) {
                ping_sent_ns_ = realtime_ns();
                command("{\"cmd\":\"status\"}");
                usleep(20'000);
            }
        session(fd);
        clock_known_ = false;
        {
            std::lock_guard<std::mutex> g(clock_lock_);
            samples_.clear();   // a new connection may be a restarted node with another clock
        }
        int expected = fd;
        if (fd_.compare_exchange_strong(expected, -1)) close(fd);
        std::lock_guard<std::mutex> g(stats_lock_);
        stats_.connected = false;
        stats_.reconnects++;
    }
}

bool NodeReceiver::session(int fd) {
    const AVCodec* codec = avcodec_find_decoder(AV_CODEC_ID_H264);
    AVCodecContext* ctx = avcodec_alloc_context3(codec);
    ctx->flags |= AV_CODEC_FLAG_LOW_DELAY;
    ctx->thread_count = 1;   // frame threading would add a frame of delay; cameras decode in parallel instead
    avcodec_open2(ctx, codec, nullptr);
    AVPacket* pkt = av_packet_alloc();
    AVFrame* pic = av_frame_alloc();
    bool started = false;
    int64_t last_seq = -1;
    std::vector<uint8_t> fixed, data;
    std::string json;

    while (!stop_) {
        proto::MessageHeader h;
        if (!read_exact(fd, &h, sizeof h) || h.magic != proto::kMagic) break;
        fixed.resize(h.header_len);
        json.assign(h.json_len, '\0');
        data.resize(h.data_len);
        if ((h.header_len && !read_exact(fd, fixed.data(), fixed.size())) ||
            (h.json_len && !read_exact(fd, json.data(), json.size())) ||
            (h.data_len && !read_exact(fd, data.data(), data.size())))
            break;
        int64_t receive_ns = realtime_ns();
        if (h.type == proto::kMsgAck) {
            on_ack(json, receive_ns);
            continue;
        }
        if (h.type == proto::kMsgStatus || h.type == proto::kMsgHello) {
            std::lock_guard<std::mutex> g(stats_lock_);
            if (h.type == proto::kMsgStatus) stats_.last_status = json;
            continue;
        }
        if (h.type != proto::kMsgFrame || fixed.size() < sizeof(proto::FrameHeader)) continue;
        proto::FrameHeader fh;
        std::memcpy(&fh, fixed.data(), sizeof fh);
        if (fh.format != proto::kFormatH264) continue;   // full_frame (Y8) answers go elsewhere later
        bool key = fh.flags & proto::kFlagKeyframe;
        if (on_packet_) on_packet_(fh, data.data(), data.size(), receive_ns, ptp_ ? 0 : offset_ns_.load());
        if (!started && !key) continue;   // a decoder can only start at a keyframe
        // Without PTP a frame is only placed in time once the node's clock offset
        // has been measured (the first status answers come within ~0.1 s).
        if (!ptp_ && !clock_known_) continue;
        if (key && !started) last_seq = -1;
        started = true;

        int64_t offset = ptp_ ? 0 : offset_ns_.load();
        int64_t stamp_local = fh.stamp_ns - offset;
        int64_t t0 = realtime_ns();
        pkt->data = data.data();
        pkt->size = int(data.size());
        bool got = false;
        Frame f;
        if (avcodec_send_packet(ctx, pkt) == 0) {
            while (avcodec_receive_frame(ctx, pic) == 0) {
                got = true;
                bool color = fh.flags & proto::kFlagColor;
                f.width = pic->width;
                f.height = pic->height;
                copy_plane(pic->data[0], pic->linesize[0], pic->width, pic->height, f.y);
                if (color) {
                    copy_plane(pic->data[1], pic->linesize[1], pic->width / 2, pic->height / 2, f.u);
                    copy_plane(pic->data[2], pic->linesize[2], pic->width / 2, pic->height / 2, f.v);
                }
                f.color = color;
            }
        }
        int64_t t1 = realtime_ns();
        {
            std::lock_guard<std::mutex> g(stats_lock_);
            stats_.camera_id = proto::camera_id(fh);
            stats_.frames++;
            stats_.keyframes += key;
            stats_.bytes += data.size();
            if (last_seq >= 0 && int64_t(fh.frame_seq) > last_seq + 1)
                stats_.lost += uint64_t(int64_t(fh.frame_seq) - last_seq - 1);
            stats_.arrive_ms.push_back((receive_ns - stamp_local) / 1e6);
            stats_.node_ms.push_back((fh.node_send_ns - fh.stamp_ns) / 1e6);
            if (got) {
                stats_.decode_ms.push_back((t1 - t0) / 1e6);
                stats_.decoded_ms.push_back((t1 - stamp_local) / 1e6);
                stats_.width = f.width;
                stats_.height = f.height;
                stats_.color = f.color;
            } else {
                stats_.decode_fail++;
            }
        }
        last_seq = int64_t(fh.frame_seq);
        if (!got) continue;
        f.camera_id = proto::camera_id(fh);
        f.frame_seq = fh.frame_seq;
        f.node_stamp_ns = fh.stamp_ns;
        f.stamp_ns = stamp_local;
        f.exposure_ns = fh.exposure_ns;
        f.line_time_ns = fh.line_time_ns;
        f.sensor_width = fh.sensor_width;
        f.sensor_height = fh.sensor_height;
        f.keyframe = key;
        f.receive_ns = receive_ns;
        f.decoded_ns = t1;
        if (on_frame_) on_frame_(std::move(f));
    }
    av_frame_free(&pic);
    av_packet_free(&pkt);
    avcodec_free_context(&ctx);
    return true;
}

}  // namespace mocap::rx
