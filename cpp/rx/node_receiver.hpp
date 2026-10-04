// One camera node: connection, H.264 decoding, clock, statistics.
//
// A reader thread reads messages and decodes every frame as it arrives (one
// decoder per camera, so six cameras decode on six cores in parallel); a
// small pinger thread sends a "status" command every few seconds, and the
// answers keep the node-to-host clock offset (NTP estimator: the round trip
// with the smallest delay in the last 10 s wins).  Under PTP the stamps are
// already on a common scale and the offset is taken as 0.
//
// The connection is re-established if the node goes away; the decoder starts
// again at the next keyframe.
#pragma once

#include <atomic>
#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

#include "../common/protocol.hpp"
#include "frame.hpp"

namespace mocap::rx {

struct NodeStats {
    std::string addr, camera_id;
    bool connected = false;
    uint64_t frames = 0, keyframes = 0, lost = 0, decode_fail = 0, bytes = 0, reconnects = 0;
    int64_t offset_ns = 0, offset_unc_ns = -1;
    int width = 0, height = 0;
    bool color = false;
    std::vector<double> arrive_ms, decoded_ms, decode_ms, node_ms;   // since the last take
    std::string last_status;
};

class NodeReceiver {
public:
    using FrameFn = std::function<void(Frame&&)>;
    // Raw H.264 access unit with its header, for recording: data, size,
    // receive time and the node-minus-host clock offset then in use.
    using PacketFn = std::function<void(const proto::FrameHeader&, const uint8_t*, size_t, int64_t, int64_t)>;

    NodeReceiver(std::string addr, bool ptp, FrameFn on_frame, PacketFn on_packet = nullptr);
    ~NodeReceiver();

    // Statistics since the previous call (the sample vectors are moved out).
    NodeStats take_stats();
    // Send a JSON command (e.g. {"cmd":"configure","exposure_us":900}).
    bool command(const std::string& json);

private:
    void run();
    void pinger();
    bool session(int fd);
    void on_ack(const std::string& json, int64_t receive_ns);

    std::string addr_;
    bool ptp_;
    FrameFn on_frame_;
    PacketFn on_packet_;
    std::atomic<bool> stop_{false};
    std::atomic<int> fd_{-1};
    std::mutex send_lock_;
    std::mutex stats_lock_;
    NodeStats stats_;
    // NTP samples: (receive time, round trip, offset)
    std::mutex clock_lock_;
    std::deque<std::tuple<int64_t, int64_t, int64_t>> samples_;
    std::atomic<int64_t> ping_sent_ns_{0};
    std::atomic<int64_t> offset_ns_{0};
    std::atomic<bool> clock_known_{false};
    std::thread thread_, ping_thread_;
};

}  // namespace mocap::rx
