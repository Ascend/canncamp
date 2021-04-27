//
// Created by HwHiAiUser on 4/23/21.
//

#pragma once

#include <iostream>
#include <vector>

struct Point {
    std::uint32_t x;
    std::uint32_t y;
};

struct DetectionResult {
    Point lt;   //The coordinate of left top point
    Point rb;   //The coordinate of the right bottom point
    std::string result_text;  // Face:xx%
};

#define RGBF32_CHAN_SIZE(width, height) ((width) * (height) * 4)

#define RGBU8_IMAGE_SIZE(width, height) ((width) * (height) * 3)