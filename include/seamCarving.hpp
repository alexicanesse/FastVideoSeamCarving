/*
 * This file is part of seamCarving which is released under GNU General Public License v3.0.
 * See file LICENSE or go to https://github.com/alexicanesse/FastVideoSeamCarving/blob/main/LICENSE for full license details.
 * Copyright 2022 - ENS de Lyon
 */

#ifndef INCLUDE_SEAMCARVING_HPP_
#define INCLUDE_SEAMCARVING_HPP_

/* C++ libraries */
#include <opencv2/opencv.hpp>

class seamCarving {
  public:
    /* Video */
    std::vector<cv::Mat3b> video_;            /* The video we are working on */
    std::vector<cv::Mat1b> gray_video_;       /* The gray-scaled and sub-sampled video */
    cv::Mat1f              content_grad_mag_; /* The energy map */

  private:
    float fps_;       /* Number of frames per second in the video */
    int width_;       /* Width of the sub-sampled video in pixels */
    int height_;      /* Height of the sub-sampled video in pixels */
    int width_full_;  /* Width of the  video in pixels */
    int height_full_; /* Height of the video in pixels */
    int frames_;      /* Number of frames in the video */

  private:
    /* Finds vertical seams     */
    std::vector<int> findVerticalSeamContent();
    /* Removes vertical seams   */
    void removeVerticalSeamsContent(int k);
    /* Finds horizontal seams   */
    std::vector<int> findHorizontalSeamContent();
    /* Removes horizontal seams */
    void removeHorizontalSeamsContent(int k);

  public:
    /* Loads the video          */
    bool loadContent(const std::string &link);
    /* Show a video or an image */
    template <typename T>
    void showcontent(const std::vector<cv::Mat_<T>> &video,
                     const std::string &title);
    /* Save an image or a video */
    template <typename T>
    void saveContent(const std::string &link, const std::vector<cv::Mat_<T>> &video);

    /* Computes the energy map  */
    void computeGradMagContent();
    /* Wrapper function the seam carving algorithm */
    void resizeContent(double horizontal_factor, double vertical_factor);
};

#endif /* INCLUDE_SEAMCARVING_HPP_ */