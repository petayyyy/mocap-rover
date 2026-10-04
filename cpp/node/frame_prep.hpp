// Raw SBGGR8 frame -> the picture the encoder takes, written straight into its
// input buffer (YUV420: Y, then U and V at half size).
//
// The output is always the WHOLE sensor frame, resized -- never a crop -- so
// the 160-degree field of view is kept at every resolution; grey or colour.
//
//   Y, full size          demosaic in row bands straight into the encoder buffer
//   Y, half size or less  2x2 cell mean (B+G+G+R)/4, no demosaic; area resize below half
//   Y, between            full demosaic, then a bilinear step down
//   U, V (colour)         from the Bayer cells: YUV420 chroma is half size, which is
//                         exactly one cell at full size -- no full colour demosaic
//
// Everything is made in one pass: each band of the raw frame is read once.
//
// The camera and encoder buffers are DMA memory (uncached): each is read or
// written once, in bands on four cores.  The chroma planes stay at 128 for
// grey (the encoder prefills them).
//
// SBGGR8 is OpenCV's *BayerRG* pattern: OpenCV names the pattern by the second
// row.  (BayerBG swaps red and blue; checked on a synthetic mosaic.)
#pragma once

#include <opencv2/core.hpp>

#include <cstdint>

namespace mocap {

struct OutputFormat {
    int width = 1640;
    int height = 1232;
    bool color = false;
    // White balance for the chroma: red and blue relative to green.  The sensor
    // has no ISP in this path, so without it everything leans green.  Luminance
    // is not balanced (the marker path reads Y only).
    float red_gain = 1.0f, blue_gain = 1.0f;
};

struct EncoderPlanes {
    uint8_t* y = nullptr;
    int y_stride = 0;
    uint8_t* u = nullptr;
    uint8_t* v = nullptr;
    int c_stride = 0;
};

class FramePrep {
public:
    // Fills the encoder planes; if preview is non-null, also leaves a copy at
    // most preview_width wide there (grey CV_8UC1 or BGR CV_8UC3).
    void process(const uint8_t* raw, int raw_stride, int width, int height, const OutputFormat& out,
                 const EncoderPlanes& planes, cv::Mat* preview, int preview_width = 820);

    // Full-size lossless grey of a raw frame (calibration frames), ~10 ms.
    static void full_gray(const uint8_t* raw, int raw_stride, int width, int height, cv::Mat& gray);

private:
    cv::Mat full_, scaled_, cells_gray_, cb_, cr_, chroma_scratch_;
};

}  // namespace mocap
