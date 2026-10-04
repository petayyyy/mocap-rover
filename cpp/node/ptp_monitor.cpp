#include "ptp_monitor.hpp"

#include <unistd.h>

#include <cstdio>
#include <limits>
#include <regex>

#include "../common/clock.hpp"

namespace mocap {

PtpMonitor::PtpMonitor(double period_s, std::string pmc, std::string uds)
    : period_s_(period_s), pmc_(std::move(pmc)), uds_(std::move(uds)),
      offset_ns_(std::numeric_limits<int64_t>::min()) {
    thread_ = std::thread(&PtpMonitor::loop, this);
}

PtpMonitor::~PtpMonitor() {
    stop_ = true;
    if (thread_.joinable()) thread_.join();
}

std::string PtpMonitor::state() const {
    std::lock_guard<std::mutex> g(lock_);
    return state_;
}

double PtpMonitor::age_s() const {
    int64_t u = updated_ns_;
    return u ? (monotonic_ns() - u) / 1e9 : -1.0;
}

void PtpMonitor::loop() {
    while (!stop_) {
        poll_once();
        for (int i = 0; i < int(period_s_ * 10) && !stop_; ++i) usleep(100'000);
    }
}

void PtpMonitor::poll_once() {
    std::string cmd = "sudo -n " + pmc_ + " -u -b 0 -s " + uds_ +
                      " 'GET CURRENT_DATA_SET' 'GET PORT_DATA_SET' 2>/dev/null";
    FILE* p = popen(cmd.c_str(), "r");
    if (!p) {
        std::lock_guard<std::mutex> g(lock_);
        state_ = "unavailable";
        return;
    }
    std::string out;
    char buf[512];
    while (fgets(buf, sizeof buf, p)) out += buf;
    pclose(p);
    static const std::regex off_re(R"(offsetFromMaster\s+(-?\d+(?:\.\d+)?))");
    static const std::regex state_re(R"(portState\s+(\w+))");
    std::smatch m;
    std::lock_guard<std::mutex> g(lock_);
    if (out.find_first_not_of(" \t\r\n") == std::string::npos) {
        state_ = "unavailable";   // no ptp4l, or no sudo rule for pmc
        offset_ns_ = std::numeric_limits<int64_t>::min();
        return;
    }
    if (std::regex_search(out, m, off_re)) offset_ns_ = int64_t(std::stod(m[1]));
    state_ = std::regex_search(out, m, state_re) ? std::string(m[1]) : "no_answer";   // ptp4l not running
    updated_ns_ = monotonic_ns();
}

}  // namespace mocap
