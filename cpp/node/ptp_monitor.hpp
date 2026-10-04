// PTP health on the node: polls pmc for offsetFromMaster and the port state,
// so every frame and every status says how well this node's clock follows the
// master.  The frame stamps themselves do not depend on it: they are taken on
// REALTIME, which phc2sys keeps on the PTP time scale.
//
// pmc needs root to bind its socket next to ptp4l's; the node runs it through
// "sudo -n" (one NOPASSWD line for /usr/sbin/pmc, see setup_cm4_node.sh).
#pragma once

#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>

namespace mocap {

class PtpMonitor {
public:
    explicit PtpMonitor(double period_s = 5.0, std::string pmc = "/usr/sbin/pmc",
                        std::string uds = "/var/run/ptp4l");
    ~PtpMonitor();

    int64_t offset_ns() const { return offset_ns_; }   // INT64_MIN when unknown
    std::string state() const;                          // MASTER, SLAVE, ... or unavailable
    double age_s() const;                               // since the last good answer

private:
    void loop();
    void poll_once();

    double period_s_;
    std::string pmc_, uds_;
    std::atomic<bool> stop_{false};
    std::atomic<int64_t> offset_ns_;
    std::atomic<int64_t> updated_ns_{0};
    mutable std::mutex lock_;
    std::string state_ = "unknown";
    std::thread thread_;
};

}  // namespace mocap
