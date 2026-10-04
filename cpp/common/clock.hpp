// Clocks and small statistics shared by the node and the tools.
#pragma once

#include <algorithm>
#include <cstdint>
#include <ctime>
#include <vector>

namespace mocap {

inline int64_t clock_ns(clockid_t id) {
    timespec ts{};
    clock_gettime(id, &ts);
    return int64_t(ts.tv_sec) * 1'000'000'000 + ts.tv_nsec;
}

inline int64_t boottime_ns() { return clock_ns(CLOCK_BOOTTIME); }
inline int64_t monotonic_ns() { return clock_ns(CLOCK_MONOTONIC); }
inline int64_t realtime_ns() { return clock_ns(CLOCK_REALTIME); }

// REALTIME - MONOTONIC: V4L2 (unicam, the encoder) stamps buffers on
// MONOTONIC; PTP (phc2sys) keeps REALTIME common across the nodes.
inline int64_t realtime_minus_monotonic_ns() {
    int64_t m0 = monotonic_ns();
    int64_t r = realtime_ns();
    int64_t m1 = monotonic_ns();
    return r - (m0 + m1) / 2;
}

// REALTIME - BOOTTIME, read between two BOOTTIME samples so the error is half
// the gap.  libcamera stamps frames on BOOTTIME; PTP (phc2sys) keeps REALTIME
// common across the nodes, so a stamp moves to the common scale by this offset.
inline int64_t realtime_minus_boottime_ns() {
    int64_t b0 = boottime_ns();
    int64_t r = realtime_ns();
    int64_t b1 = boottime_ns();
    return r - (b0 + b1) / 2;
}

// Percentile of a copy (values are small vectors of a status period).
inline double percentile(std::vector<double> v, double q) {
    if (v.empty()) return 0.0;
    size_t k = std::min(v.size() - 1, size_t(q / 100.0 * double(v.size() - 1) + 0.5));
    std::nth_element(v.begin(), v.begin() + long(k), v.end());
    return v[k];
}

}  // namespace mocap
