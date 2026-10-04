// How soon does a raw frame reach userspace straight from the CSI receiver
// (unicam), without libcamera?  Sets the sensor through its V4L2 subdevice,
// streams /dev/video0 and prints the delay from the buffer timestamp to DQBUF,
// the frame rate, and how long a 2 MB copy out of the buffer takes.
#include <fcntl.h>
#include <linux/v4l2-subdev.h>
#include <linux/videodev2.h>
#include <poll.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <vector>

static long long now_ns(clockid_t c) { timespec t; clock_gettime(c, &t); return t.tv_sec * 1000000000LL + t.tv_nsec; }
static int ctl(int fd, unsigned id, int v) { v4l2_control c{id, v}; return ioctl(fd, VIDIOC_S_CTRL, &c); }
static long long get_ext(int fd, unsigned id) {
    v4l2_ext_control c{}; c.id = id; v4l2_ext_controls cs{}; cs.which = V4L2_CTRL_WHICH_CUR_VAL; cs.count = 1; cs.controls = &c;
    if (ioctl(fd, VIDIOC_G_EXT_CTRLS, &cs)) return -1; return c.value64 ? c.value64 : c.value;
}
int main(int argc, char** argv) {
    double fps = argc > 1 ? atof(argv[1]) : 50; int exp_us = argc > 2 ? atoi(argv[2]) : 800; int secs = argc > 3 ? atoi(argv[3]) : 10;
    const int W = 1640, H = 1232;
    int sd = open("/dev/v4l-subdev0", O_RDWR);
    v4l2_subdev_format f{}; f.which = V4L2_SUBDEV_FORMAT_ACTIVE; f.pad = 0;
    f.format.width = W; f.format.height = H; f.format.code = MEDIA_BUS_FMT_SBGGR8_1X8; f.format.field = V4L2_FIELD_NONE;
    if (ioctl(sd, VIDIOC_SUBDEV_S_FMT, &f)) perror("subdev S_FMT");
    long long pixel_rate = get_ext(sd, V4L2_CID_PIXEL_RATE), hblank = 0;
    v4l2_control hb{V4L2_CID_HBLANK, 0}; ioctl(sd, VIDIOC_G_CTRL, &hb); hblank = hb.value;
    double line_ns = 1e9 * double(W + hblank) / double(pixel_rate);
    int frame_lines = int(1e9 / fps / line_ns + 0.5);
    if (ctl(sd, V4L2_CID_VBLANK, frame_lines - H)) perror("VBLANK");
    if (ctl(sd, V4L2_CID_EXPOSURE, std::max(1, int(exp_us * 1000.0 / line_ns)))) perror("EXPOSURE");
    ctl(sd, V4L2_CID_ANALOGUE_GAIN, 200);
    printf("pixel_rate %lld, hblank %lld, line %.1f ns, frame_lines %d -> %.2f fps, exposure %d lines\n",
           pixel_rate, hblank, line_ns, frame_lines, 1e9 / (frame_lines * line_ns), int(exp_us * 1000.0 / line_ns));
    int vd = open("/dev/video0", O_RDWR);
    v4l2_format vf{}; vf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE; vf.fmt.pix.width = W; vf.fmt.pix.height = H;
    vf.fmt.pix.pixelformat = V4L2_PIX_FMT_SBGGR8; vf.fmt.pix.field = V4L2_FIELD_NONE;
    if (ioctl(vd, VIDIOC_S_FMT, &vf)) { perror("S_FMT"); return 1; }
    int stride = int(vf.fmt.pix.bytesperline);
    v4l2_requestbuffers rb{}; rb.count = 4; rb.type = V4L2_BUF_TYPE_VIDEO_CAPTURE; rb.memory = V4L2_MEMORY_MMAP;
    if (ioctl(vd, VIDIOC_REQBUFS, &rb)) { perror("REQBUFS"); return 1; }
    std::vector<unsigned char*> maps(rb.count);
    for (unsigned i = 0; i < rb.count; ++i) {
        v4l2_buffer b{}; b.type = rb.type; b.memory = V4L2_MEMORY_MMAP; b.index = i; ioctl(vd, VIDIOC_QUERYBUF, &b);
        maps[i] = (unsigned char*)mmap(nullptr, b.length, PROT_READ, MAP_SHARED, vd, b.m.offset); ioctl(vd, VIDIOC_QBUF, &b);
    }
    int t = V4L2_BUF_TYPE_VIDEO_CAPTURE; if (ioctl(vd, VIDIOC_STREAMON, &t)) { perror("STREAMON"); return 1; }
    std::vector<double> lat, cp; std::vector<unsigned char> copy(size_t(stride) * H);
    long long t_end = now_ns(CLOCK_MONOTONIC) + secs * 1000000000LL; int n = 0; unsigned last_seq = 0, missed = 0; unsigned flags = 0;
    while (now_ns(CLOCK_MONOTONIC) < t_end) {
        pollfd p{vd, POLLIN, 0}; poll(&p, 1, 1000);
        v4l2_buffer b{}; b.type = rb.type; b.memory = V4L2_MEMORY_MMAP;
        if (ioctl(vd, VIDIOC_DQBUF, &b)) continue;
        long long now = now_ns(CLOCK_MONOTONIC), ts = b.timestamp.tv_sec * 1000000000LL + b.timestamp.tv_usec * 1000LL;
        if (n > 5) lat.push_back((now - ts) / 1e6);
        long long c0 = now_ns(CLOCK_MONOTONIC); memcpy(copy.data(), maps[b.index], copy.size()); cp.push_back((now_ns(CLOCK_MONOTONIC) - c0) / 1e6);
        if (n && b.sequence > last_seq + 1) missed += b.sequence - last_seq - 1; last_seq = b.sequence; flags = b.flags;
        ioctl(vd, VIDIOC_QBUF, &b); ++n;
    }
    std::sort(lat.begin(), lat.end()); std::sort(cp.begin(), cp.end());
    printf("%d frames in %d s = %.1f fps, missed %u; ts source %s; stamp->DQBUF min %.2f P50 %.2f P95 %.2f ms; 2 MB copy P50 %.2f ms\n",
           n, secs, double(n) / secs, missed,
           (flags & V4L2_BUF_FLAG_TSTAMP_SRC_MASK) == V4L2_BUF_FLAG_TSTAMP_SRC_SOE ? "start of exposure/frame (SOE)" : "end of frame (EOF)",
           lat.front(), lat[lat.size() / 2], lat[lat.size() * 95 / 100], cp[cp.size() / 2]);
    ioctl(vd, VIDIOC_STREAMOFF, &t);
}
