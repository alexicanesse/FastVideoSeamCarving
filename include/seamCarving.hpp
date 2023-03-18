/*
 * This file is part of seamCarving which is released under GNU General Public License v3.0.
 * See file LICENSE or go to https://github.com/alexicanesse/ChineseCheckers/blob/main/LICENSE for full license details.
 * Copyright 2022 - ENS de Lyon
 */

#ifndef INCLUDE_SEAMCARVING_HPP_
#define INCLUDE_SEAMCARVING_HPP_

/* C++ libraries */
#include <opencv2/opencv.hpp>

class seamCarving {
 public:
    cv::Mat3b image_;
    cv::Mat1f image_grad_mag_;

    bool loadImage(const std::string &link);
    void showImage(const cv::Mat &image, const std::string &title);

    void computeGradMagImage();
};

#endif /* INCLUDE_SEAMCARVING_HPP_ */