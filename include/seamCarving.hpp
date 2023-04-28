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
    int duration_ = 0;

    /* Video */
    std::vector<cv::Mat3b> video_;
    std::vector<cv::Mat1b> gray_video_;
    cv::Mat1f content_grad_mag_;
    float fps_;

    int width_;
    int height_;
    int width_full_;
    int height_full_;
    int frames_;

    bool loadContent(const std::string &link);
    template <typename T>
    void saveContent(const std::string &link, const std::vector<cv::Mat_<T>> &video);

    template <typename T>
    void showcontent(const std::vector<cv::Mat_<T>> &video,
                     const std::string &title);

    void computeGradMagContent();

    void resizeContent(double horizontal_factor, double vertical_factor);

    std::vector<int> findVerticalSeamContent();
    void removeVerticalSeamsContent(int k);
    std::vector<int> findHorizontalSeamContent();
    void removeHorizontalSeamsContent(int k);
};

#endif /* INCLUDE_SEAMCARVING_HPP_ */