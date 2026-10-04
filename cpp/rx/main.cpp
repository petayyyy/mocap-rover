// mocap_rx -- the laptop's receiving path: every camera node's H.264 stream
// in, decoded frames out.
//
//   mocap_rx [--ptp] [--shm] [--record DIR] [--seconds N] [--stats S] host:port ...
//
// One receiver per node (own thread, own decoder): six cameras decode on six
// cores in parallel, ~1.5 ms per 1640x1232 frame.  Decoded frames go to
// shared memory (--shm, /dev/shm/mocap_<camera_id>) for other processes --
// the Python web page, the tract while it is ported -- and, in-process, to
// the C++ tract when it lands here.  --record keeps the H.264 as it came.
// Every --stats seconds one line per camera: fps, lost, Mbit/s and the
// latency from exposure to arrival and to the decoded picture.

#include <signal.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "../common/clock.hpp"
#include "node_receiver.hpp"
#include "recorder.hpp"
#include "shm_ring.hpp"

using namespace mocap::rx;

namespace {

std::atomic<bool> g_stop{false};
void on_signal(int) { g_stop = true; }

std::string p50_95(const std::vector<double>& v) {
    char b[64];
    std::snprintf(b, sizeof b, "%5.1f/%5.1f", mocap::percentile(v, 50), mocap::percentile(v, 95));
    return b;
}

}  // namespace

int main(int argc, char** argv) {
    bool ptp = false, shm = false;
    std::string record;
    double seconds = 0, stats_s = 2;
    std::vector<std::string> nodes;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--ptp") ptp = true;
        else if (a == "--shm") shm = true;
        else if (a == "--record" && i + 1 < argc) record = argv[++i];
        else if (a == "--seconds" && i + 1 < argc) seconds = std::stod(argv[++i]);
        else if (a == "--stats" && i + 1 < argc) stats_s = std::stod(argv[++i]);
        else if (a.rfind("--", 0) == 0) {
            std::cerr << "unknown option " << a << "\n";
            return 2;
        } else nodes.push_back(a);
    }
    if (nodes.empty()) {
        std::cerr << "usage: mocap_rx [--ptp] [--shm] [--record DIR] [--seconds N] [--stats S] host:port ...\n";
        return 2;
    }
    signal(SIGINT, on_signal);
    signal(SIGTERM, on_signal);
    signal(SIGPIPE, SIG_IGN);

    std::unique_ptr<Recorder> recorder = record.empty() ? nullptr : std::make_unique<Recorder>(record);
    std::mutex rings_lock;
    std::map<std::string, std::unique_ptr<ShmRing>> rings;
    auto publish = [&](Frame&& f) {
        if (!shm) return;
        ShmRing* ring;
        {
            std::lock_guard<std::mutex> g(rings_lock);
            auto& r = rings[f.camera_id];
            if (!r) r = std::make_unique<ShmRing>(f.camera_id);
            ring = r.get();
        }
        ring->publish(f);   // one writer per camera: its own receiver thread
    };
    NodeReceiver::PacketFn keep;
    if (recorder)
        keep = [&](const mocap::proto::FrameHeader& h, const uint8_t* d, size_t n, int64_t rx_ns, int64_t offset) {
            recorder->write(h, d, n, rx_ns, offset);
        };
    std::vector<std::unique_ptr<NodeReceiver>> receivers;
    for (const auto& addr : nodes) receivers.push_back(std::make_unique<NodeReceiver>(addr, ptp, publish, keep));

    int64_t start = mocap::realtime_ns(), next = start + int64_t(stats_s * 1e9);
    while (!g_stop && (seconds <= 0 || mocap::realtime_ns() - start < int64_t(seconds * 1e9))) {
        usleep(50'000);
        if (mocap::realtime_ns() < next) continue;
        next += int64_t(stats_s * 1e9);
        for (auto& r : receivers) {
            NodeStats s = r->take_stats();
            std::printf("%-12s %-20s %s %4dx%-4d %s fps %5.1f lost %3lu fail %lu %5.1f Mbit/s | exp->arrive %s  ->decoded %s ms | decode %s | clock %+.2f ms (+-%.2f)\n",
                        s.camera_id.empty() ? "?" : s.camera_id.c_str(), s.addr.c_str(), s.connected ? "up  " : "DOWN",
                        s.width, s.height, s.color ? "col " : "grey", s.frames / stats_s, (unsigned long)s.lost,
                        (unsigned long)s.decode_fail, s.bytes * 8 / stats_s / 1e6, p50_95(s.arrive_ms).c_str(),
                        p50_95(s.decoded_ms).c_str(), p50_95(s.decode_ms).c_str(), s.offset_ns / 1e6,
                        s.offset_unc_ns / 1e6);
        }
        std::fflush(stdout);
    }
    receivers.clear();
    return 0;
}
