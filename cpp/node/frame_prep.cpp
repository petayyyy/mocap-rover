#include "frame_prep.hpp"

#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cstring>

namespace mocap {

namespace {

constexpr int kBands = 4;

// Row bands of [0, rows) with even boundaries (the Bayer phase is per 2 rows).
template <class F>
void bands(int rows, F&& fn) {
    const int step = rows / kBands / 2 * 2;
    cv::parallel_for_(cv::Range(0, kBands), [&](const cv::Range& r) {
        for (int b = r.start; b < r.end; ++b) fn(b * step, b == kBands - 1 ? rows : (b + 1) * step);
    }, kBands);
}

void copy_rows(const cv::Mat& from, uint8_t* to, int to_stride) {
    bands(from.rows, [&](int y0, int y1) {
        for (int y = y0; y < y1; ++y)
            std::memcpy(to + size_t(y) * size_t(to_stride), from.ptr(y), size_t(from.cols) * from.elemSize());
    });
}

// One Bayer cell row pair (B G / G R) -> BT.601 luma of the cell's colour
// (B, mean G, R) -- the weights of the full demosaic -- and, if asked, its
// full-range chroma after white balance (gains in 1/256).
void cells_row(const uint8_t* __restrict r0, const uint8_t* __restrict r1, int cw, uint8_t* __restrict gray,
               uint8_t* __restrict cb, uint8_t* __restrict cr, int red_q8, int blue_q8) {
    if (gray)
        for (int x = 0; x < cw; ++x) {
            int b = r0[2 * x], g2x = r0[2 * x + 1] + r1[2 * x], r = r1[2 * x + 1];
            gray[x] = uint8_t((77 * r + 75 * g2x + 29 * b + 128) >> 8);
        }
    if (!cb) return;
    // Inputs are 0..255 after the white-balance clip, so the results stay in
    // 0..255 without clamping (the loop vectorises).
    for (int x = 0; x < cw; ++x) {
        int g = (r0[2 * x + 1] + r1[2 * x] + 1) >> 1;
        int b = std::min(255, (r0[2 * x] * blue_q8) >> 8), r = std::min(255, (r1[2 * x + 1] * red_q8) >> 8);
        cb[x] = uint8_t((-43 * r - 85 * g + 128 * b + 32768) >> 8);
        cr[x] = uint8_t((128 * r - 107 * g - 21 * b + 32768) >> 8);
    }
}

void resize_into(const cv::Mat& from, cv::Size size, cv::Mat& scratch, uint8_t* to, int to_stride) {
    if (from.size() == size) {
        copy_rows(from, to, to_stride);
        return;
    }
    bool down = size.width <= from.cols / 2 || size.height <= from.rows / 2;
    cv::resize(from, scratch, size, 0, 0, down ? cv::INTER_AREA : cv::INTER_LINEAR);
    copy_rows(scratch, to, to_stride);
}

}  // namespace

void FramePrep::full_gray(const uint8_t* raw, int raw_stride, int width, int height, cv::Mat& gray) {
    cv::Mat src(height, width, CV_8UC1, const_cast<uint8_t*>(raw), size_t(raw_stride));
    gray.create(height, width, CV_8UC1);
    bands(height, [&](int y0, int y1) {
        int m0 = std::max(0, y0 - 2), m1 = std::min(height, y1 + 2);
        cv::Mat band, out;
        src.rowRange(m0, m1).copyTo(band);
        cv::cvtColor(band, out, cv::COLOR_BayerRG2GRAY);
        out.rowRange(y0 - m0, y1 - m0).copyTo(gray.rowRange(y0, y1));
    });
}

void FramePrep::process(const uint8_t* raw, int raw_stride, int width, int height, const OutputFormat& out,
                        const EncoderPlanes& p, cv::Mat* preview, int preview_width) {
    cv::Mat src(height, width, CV_8UC1, const_cast<uint8_t*>(raw), size_t(raw_stride));
    const cv::Size out_size(out.width, out.height);
    const int cw = width / 2, ch = height / 2;
    const bool full = out.width == width && out.height == height;
    const bool small = out.width <= cw && out.height <= ch;   // at or below one pixel per Bayer cell
    const bool demosaic = !small;
    const bool cells = small || out.color || preview != nullptr;
    const bool cells_gray = small || preview != nullptr;   // full-size colour takes Y from the demosaic

    if (cells_gray) cells_gray_.create(ch, cw, CV_8UC1);
    if (out.color) {
        cb_.create(ch, cw, CV_8UC1);
        cr_.create(ch, cw, CV_8UC1);
    }
    if (demosaic && !full) full_.create(height, width, CV_8UC1);
    cv::Mat y_direct(out.height, out.width, CV_8UC1, p.y, size_t(p.y_stride));

    // One pass over the raw frame: each band is read from the (uncached)
    // camera buffer once and everything is made from the cached copy.
    bands(height, [&](int y0, int y1) {
        int m0 = std::max(0, y0 - 2), m1 = std::min(height, y1 + 2);   // margins keep band edges exact
        cv::Mat band;
        src.rowRange(m0, m1).copyTo(band);
        if (demosaic) {
            cv::Mat g;
            cv::cvtColor(band, g, cv::COLOR_BayerRG2GRAY);
            g.rowRange(y0 - m0, y1 - m0).copyTo((full ? y_direct : full_).rowRange(y0, y1));
        }
        if (cells) {
            for (int y = y0; y < y1; y += 2) {
                const uint8_t* r0 = band.ptr(y - m0);
                const uint8_t* r1 = band.ptr(y - m0 + 1);
                int cy = y / 2;
                cells_row(r0, r1, cw, cells_gray ? cells_gray_.ptr(cy) : nullptr, out.color ? cb_.ptr(cy) : nullptr,
                          out.color ? cr_.ptr(cy) : nullptr, int(out.red_gain * 256), int(out.blue_gain * 256));
            }
        }
    });

    // Luminance at the stream size.
    if (small) resize_into(cells_gray_, out_size, scaled_, p.y, p.y_stride);
    else if (!full) resize_into(full_, out_size, scaled_, p.y, p.y_stride);

    // Chroma: U, V are half size in YUV420 -- one value per Bayer cell at full size.
    if (out.color) {
        const cv::Size c_size(out.width / 2, out.height / 2);
        resize_into(cb_, c_size, chroma_scratch_, p.u, p.c_stride);
        resize_into(cr_, c_size, chroma_scratch_, p.v, p.c_stride);
    }

    if (preview) {
        cv::Mat small_view;
        if (out.color) {
            cv::Mat ycrcb, planes[3] = {cells_gray_, cr_, cb_};
            cv::merge(planes, 3, ycrcb);
            cv::cvtColor(ycrcb, small_view, cv::COLOR_YCrCb2BGR);
        } else {
            small_view = cells_gray_;
        }
        double k = std::min(1.0, double(preview_width) / small_view.cols);
        cv::resize(small_view, *preview, cv::Size(int(small_view.cols * k + 0.5), int(small_view.rows * k + 0.5)),
                   0, 0, cv::INTER_AREA);
    }
}

}  // namespace mocap
