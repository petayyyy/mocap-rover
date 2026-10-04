// mocap_camstat -- receive the camera nodes' H.264 streams, decode them and
// measure what the tract would see.
//
//   mocap_camstat --seconds 60 192.168.10.101:5600 192.168.11.104:5600 ...
//
// Per node: frames and keyframes received, frames lost (sensor sequence
// gaps), decode failures, Mbit/s, and the latency from the start of exposure
// to (a) arrival and (b) the decoded picture, plus the node's own share
// (exposure -> send, from the header).  Cross-machine latencies need a
// common clock: with --clock ptp the stamps are taken as they are (PTP keeps
// REALTIME common); otherwise the node's offset is estimated from the round
// trip of status commands (NTP estimator, smallest round trip wins).

#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/frame.h>
}

#include <atomic>
#include <cstring>
#include <algorithm>
#include <iomanip>
#include <sstream>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "../common/clock.hpp"
#include "../common/protocol.hpp"

namespace proto = mocap::proto;

namespace {

struct Stats {
    std::string addr, camera_id, hello, last_status;
    uint64_t frames = 0, keyframes = 0, bytes = 0, lost = 0, decode_fail = 0, decoded = 0;
    int64_t last_seq = -1, offset_ns = 0, offset_unc_ns = -1;
    std::vector<double> arrive_ms, decoded_ms, node_ms, decode_ms;
    int width = 0, height = 0;
    std::string error;
};

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

bool write_all(int fd, const std::vector<uint8_t>& m) {
    size_t off = 0;
    while (off < m.size()) {
        ssize_t w = send(fd, m.data() + off, m.size() - off, MSG_NOSIGNAL);
        if (w <= 0) return false;
        off += size_t(w);
    }
    return true;
}

int connect_to(const std::string& addr) {
    auto colon = addr.rfind(':');
    std::string host = addr.substr(0, colon), port = colon == std::string::npos ? "5600" : addr.substr(colon + 1);
    addrinfo hints{}, *res = nullptr;
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    if (getaddrinfo(host.c_str(), port.c_str(), &hints, &res)) return -1;
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (connect(fd, res->ai_addr, res->ai_addrlen)) {
        close(fd);
        fd = -1;
    }
    freeaddrinfo(res);
    int one = 1;
    if (fd >= 0) setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
    return fd;
}

int64_t json_int(const std::string& j, const std::string& key) {
    auto p = j.find("\"" + key + "\":");
    if (p == std::string::npos) return 0;
    return std::stoll(j.substr(p + key.size() + 3));
}

struct Message {
    proto::MessageHeader h;
    proto::FrameHeader fh;
    std::string json;
    std::vector<uint8_t> data;
    int64_t receive_ns;
};

bool read_message(int fd, Message& m) {
    if (!read_exact(fd, &m.h, sizeof m.h) || m.h.magic != proto::kMagic) return false;
    std::vector<uint8_t> fixed(m.h.header_len);
    if (m.h.header_len && !read_exact(fd, fixed.data(), fixed.size())) return false;
    m.json.assign(m.h.json_len, '\0');
    if (m.h.json_len && !read_exact(fd, m.json.data(), m.json.size())) return false;
    m.data.resize(m.h.data_len);
    if (m.h.data_len && !read_exact(fd, m.data.data(), m.data.size())) return false;
    m.receive_ns = mocap::realtime_ns();
    if (m.h.type == proto::kMsgFrame && fixed.size() >= sizeof(proto::FrameHeader))
        std::memcpy(&m.fh, fixed.data(), sizeof m.fh);
    return true;
}

void run_node(Stats& s, double seconds, bool ptp, std::atomic<bool>& stop) {
    int fd = connect_to(s.addr);
    if (fd < 0) {
        s.error = "connect failed";
        return;
    }
    // Clock offset by round trips of status commands, before frames are timed.
    Message m;
    if (!ptp) {
        int64_t best_rtt = INT64_MAX;
        for (int i = 0; i < 15; ++i) {
            int64_t t0 = mocap::realtime_ns();
            write_all(fd, proto::encode_json(proto::kMsgCommand, "{\"cmd\":\"status\"}"));
            while (read_message(fd, m) && m.h.type != proto::kMsgAck) {
                if (m.h.type == proto::kMsgHello) s.hello = m.json;
            }
            int64_t t1 = mocap::realtime_ns();
            int64_t node = json_int(m.json, "realtime_ns");
            if (node && t1 - t0 < best_rtt) {
                best_rtt = t1 - t0;
                s.offset_ns = node - (t0 + t1) / 2;   // node clock minus ours
                s.offset_unc_ns = best_rtt / 2;
            }
        }
    }
    const AVCodec* codec = avcodec_find_decoder(AV_CODEC_ID_H264);
    AVCodecContext* ctx = avcodec_alloc_context3(codec);
    ctx->flags |= AV_CODEC_FLAG_LOW_DELAY;
    ctx->thread_count = 1;
    avcodec_open2(ctx, codec, nullptr);
    AVPacket* pkt = av_packet_alloc();
    AVFrame* frame = av_frame_alloc();
    int64_t end = mocap::realtime_ns() + int64_t(seconds * 1e9);
    bool started = false;
    while (!stop && mocap::realtime_ns() < end && read_message(fd, m)) {
        if (m.h.type == proto::kMsgHello) s.hello = m.json;
        if (m.h.type == proto::kMsgStatus) s.last_status = m.json;
        if (m.h.type != proto::kMsgFrame || m.fh.format != proto::kFormatH264) continue;
        bool key = m.fh.flags & proto::kFlagKeyframe;
        if (!started && !key) continue;      // a decoder can only start at a keyframe
        started = true;
        s.camera_id = proto::camera_id(m.fh);
        s.frames++;
        s.keyframes += key;
        s.bytes += m.data.size();
        if (s.last_seq >= 0 && int64_t(m.fh.frame_seq) > s.last_seq + 1)
            s.lost += uint64_t(int64_t(m.fh.frame_seq) - s.last_seq - 1);
        s.last_seq = int64_t(m.fh.frame_seq);
        int64_t stamp_local = m.fh.stamp_ns - s.offset_ns;
        s.arrive_ms.push_back((m.receive_ns - stamp_local) / 1e6);
        s.node_ms.push_back((m.fh.node_send_ns - m.fh.stamp_ns) / 1e6);
        int64_t t0 = mocap::realtime_ns();
        pkt->data = m.data.data();
        pkt->size = int(m.data.size());
        bool got = false;
        if (avcodec_send_packet(ctx, pkt) == 0) {
            while (avcodec_receive_frame(ctx, frame) == 0) {
                got = true;
                s.width = frame->width;
                s.height = frame->height;
            }
        }
        int64_t t1 = mocap::realtime_ns();
        if (got) {
            s.decoded++;
            s.decode_ms.push_back((t1 - t0) / 1e6);
            s.decoded_ms.push_back((t1 - stamp_local) / 1e6);
        } else {
            s.decode_fail++;
        }
    }
    av_frame_free(&frame);
    av_packet_free(&pkt);
    avcodec_free_context(&ctx);
    close(fd);
}

std::string stat(const std::vector<double>& v) {
    std::ostringstream o;
    o << std::fixed << std::setprecision(1) << "P50 " << mocap::percentile(v, 50) << "  P95 "
      << mocap::percentile(v, 95) << "  max " << (v.empty() ? 0.0 : *std::max_element(v.begin(), v.end()))
      << " ms";
    return o.str();
}

}  // namespace

int main(int argc, char** argv) {
    double seconds = 30;
    bool ptp = false;
    std::vector<std::string> nodes;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--seconds" && i + 1 < argc) seconds = std::stod(argv[++i]);
        else if (a == "--clock" && i + 1 < argc) ptp = std::string(argv[++i]) == "ptp";
        else nodes.push_back(a);
    }
    if (nodes.empty()) {
        std::cerr << "usage: mocap_camstat [--seconds N] [--clock ptp|estimate] host:port ...\n";
        return 2;
    }
    std::vector<Stats> stats(nodes.size());
    std::vector<std::thread> threads;
    std::atomic<bool> stop{false};
    for (size_t i = 0; i < nodes.size(); ++i) {
        stats[i].addr = nodes[i];
        threads.emplace_back(run_node, std::ref(stats[i]), seconds, ptp, std::ref(stop));
    }
    for (auto& t : threads) t.join();
    for (const Stats& s : stats) {
        std::cout << "== " << s.addr << " " << s.camera_id << (s.error.empty() ? "" : "  ERROR " + s.error) << "\n";
        if (!s.error.empty()) continue;
        std::cout << std::fixed << std::setprecision(1)
                  << "  frames         " << s.frames << " (" << s.frames / seconds << " fps), keyframes "
                  << s.keyframes << ", lost " << s.lost << ", decoded " << s.decoded << ", decode failures "
                  << s.decode_fail << ", " << s.width << "x" << s.height << "\n"
                  << "  link           " << s.bytes * 8 / seconds / 1e6 << " Mbit/s\n"
                  << "  clock          " << (ptp ? "PTP (stamps as they are)"
                                                 : "node " + std::to_string(s.offset_ns / 1e6) + " ms vs this host, +-" +
                                                       std::to_string(s.offset_unc_ns / 1e6) + " ms")
                  << "\n"
                  << "  node exp->send " << stat(s.node_ms) << "\n"
                  << "  exp->arrival   " << stat(s.arrive_ms) << "\n"
                  << "  decode         " << stat(s.decode_ms) << "\n"
                  << "  exp->decoded   " << stat(s.decoded_ms) << "\n"
                  << "  last status    " << s.last_status << "\n";
    }
    return 0;
}
