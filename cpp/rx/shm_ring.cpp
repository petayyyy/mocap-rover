#include "shm_ring.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <cstring>
#include <stdexcept>

#include "../common/clock.hpp"

namespace mocap::rx {

ShmRing::ShmRing(const std::string& camera_id, int slots, int max_width, int max_height) {
    name_ = "/mocap_" + camera_id;
    path_ = "/dev/shm" + name_;
    size_t y = size_t(max_width) * size_t(max_height), c = y / 4;
    size_t stride = (kShmSlotHeaderBytes + y + 2 * c + 4095) / 4096 * 4096;
    size_ = kShmHeaderBytes + stride * size_t(slots);
    int fd = shm_open(name_.c_str(), O_CREAT | O_RDWR, 0644);
    if (fd < 0 || ftruncate(fd, off_t(size_))) throw std::runtime_error("shm " + name_ + ": " + std::strerror(errno));
    void* p = mmap(nullptr, size_, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    if (p == MAP_FAILED) throw std::runtime_error("mmap " + name_);
    base_ = static_cast<uint8_t*>(p);
    std::memset(base_, 0, kShmHeaderBytes);
    header_ = reinterpret_cast<ShmHeader*>(base_);
    header_->version = kShmVersion;
    header_->slots = uint32_t(slots);
    header_->max_width = uint32_t(max_width);
    header_->max_height = uint32_t(max_height);
    header_->slot_stride = stride;
    std::strncpy(header_->camera_id, camera_id.c_str(), sizeof header_->camera_id - 1);
    for (int i = 0; i < slots; ++i) reinterpret_cast<ShmSlot*>(base_ + kShmHeaderBytes + stride * size_t(i))->seq_lock = 0;
    __atomic_store_n(&header_->magic, kShmMagic, __ATOMIC_RELEASE);   // readers start when this appears
}

ShmRing::~ShmRing() {
    if (base_) munmap(base_, size_);
    shm_unlink(name_.c_str());
}

void ShmRing::publish(const Frame& f) {
    if (f.width > int(header_->max_width) || f.height > int(header_->max_height)) return;
    uint64_t n = __atomic_load_n(&header_->write_count, __ATOMIC_RELAXED);
    uint8_t* slot = base_ + kShmHeaderBytes + header_->slot_stride * (n % header_->slots);
    auto* s = reinterpret_cast<ShmSlot*>(slot);
    uint64_t lock = __atomic_load_n(&s->seq_lock, __ATOMIC_RELAXED);
    __atomic_store_n(&s->seq_lock, lock + 1, __ATOMIC_RELEASE);   // odd: writing
    __atomic_thread_fence(__ATOMIC_RELEASE);
    s->frame_seq = f.frame_seq;
    s->stamp_ns = f.stamp_ns;
    s->node_stamp_ns = f.node_stamp_ns;
    s->exposure_ns = f.exposure_ns;
    s->line_time_ns = f.line_time_ns;
    s->width = uint32_t(f.width);
    s->height = uint32_t(f.height);
    s->sensor_width = uint32_t(f.sensor_width);
    s->sensor_height = uint32_t(f.sensor_height);
    s->color = f.color;
    s->keyframe = f.keyframe;
    s->receive_ns = f.receive_ns;
    s->decoded_ns = f.decoded_ns;
    uint8_t* px = slot + kShmSlotHeaderBytes;
    std::memcpy(px, f.y.data(), f.y.size());
    if (f.color) {
        size_t c = size_t(header_->max_width) * size_t(header_->max_height) / 4;
        std::memcpy(px + size_t(header_->max_width) * header_->max_height, f.u.data(), f.u.size());
        std::memcpy(px + size_t(header_->max_width) * header_->max_height + c, f.v.data(), f.v.size());
    }
    s->published_ns = realtime_ns();
    __atomic_store_n(&s->seq_lock, lock + 2, __ATOMIC_RELEASE);   // even: done
    __atomic_store_n(&header_->write_count, n + 1, __ATOMIC_RELEASE);
}

}  // namespace mocap::rx
