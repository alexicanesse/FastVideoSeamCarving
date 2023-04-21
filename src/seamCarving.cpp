/*
 * This file is part of seamCarving which is released under GNU General Public License v3.0.
 * See file LICENSE or go to https://github.com/alexicanesse/ChineseCheckers/blob/main/LICENSE for full license details.
 * Copyright 2022 - ENS de Lyon
 */


/* seamCarving.hpp */
#include "seamCarving.hpp"

/* C++ libraries */
#include <thread>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <boost/timer/progress_display.hpp>

int main() {
    cv::setNumThreads(8);

    seamCarving sc;

    sc.loadContent("tests/f2_480p.m4v");
    //sc.showcontent(sc.video_, "Before");

    auto start = std::chrono::high_resolution_clock::now();
    sc.resizeContent(1, .85);
    //sc.showcontent(std::vector<cv::Mat1f>(1, sc.content_grad_mag_), "title");
    //sc.showcontent(sc.gray_video_, "title");
    sc.showcontent(sc.video_, "After");


    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    sc.showcontent(sc.video_, "After");

    return 0;
}

void seamCarving::resizeContent(double horizontal_factor, double vertical_factor) {
    if (vertical_factor < 1) {
        removeHorizontalSeamsContent(height_ - height_ * vertical_factor);
        cv::Rect roi(0, 0, width_, height_);
        for (auto &image : video_)
            image = image(roi);
    }

    if (horizontal_factor < 1) {
        removeVerticalSeamsContent(width_ - width_ * horizontal_factor);
        cv::Rect roi(0, 0, width_, height_);
        for (auto &image : video_)
            image = image(roi);
    }
}

bool seamCarving::loadContent(const std::string &link) {
    cv::VideoCapture capture(link);
    cv::Mat3b frame;

    fps_ = capture.get(cv::CAP_PROP_FPS);
    video_.resize(capture.get(cv::CAP_PROP_FRAME_COUNT));

    if(!capture.isOpened()) {
        std::cerr << "Error when reading the file\n";
        return false;
    }

    cv::namedWindow( "test", 1);

    for (int i = 0; i < video_.size(); ++i) {
        capture >> frame;
        if(frame.empty()) {
            video_.resize(i);
            break;
        }

        frame.copyTo(video_[i]);
    }

    /* Convert the image to grayscale */
    gray_video_.resize(video_.size());
    for (int i = 0; i < gray_video_.size(); ++i) {
        cv::cvtColor(video_[i], gray_video_[i], cv::COLOR_BGR2GRAY);
    }

    width_  = video_[0].cols;
    height_ = video_[0].rows;
    frames_ = video_.size();

    return true;
}

template <typename T>
void seamCarving::showcontent(const std::vector<cv::Mat_<T>> &video,
                              const std::string &title) {

    if (video.empty())
        return;

    int delay = 1000./fps_;

    auto start1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < video.size() - 1; ++i) {
        start1 = std::chrono::high_resolution_clock::now();
        cv::imshow(title, video[i]);

        cv::waitKey(1); /* waits to display frame */
        std::this_thread::sleep_for(std::chrono::milliseconds((delay - std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start1).count())));
    }
    cv::imshow(title, video[video.size() - 1]);
    cv::waitKey(); /* wait for a keypress before closing the window */
}

void seamCarving::computeGradMagContent() {
    cv::Mat temp_grad_mag = cv::Mat1f::zeros(height_, width_);
    content_grad_mag_     = cv::Mat1f::zeros(height_, width_);
    for (int i = 0; i < frames_; ++i) {
        /* Compute gradients using Sobel operator */
        cv::Mat1f grad_x, grad_y;
        cv::Sobel(gray_video_[i], grad_x, CV_32F, 1, 0);
        cv::Sobel(gray_video_[i], grad_y, CV_32F, 0, 1);

        /* Compute gradient magnitude */
        cv::magnitude(grad_x, grad_y, temp_grad_mag);

        /* Normalize the gradient magnitude image and store it */
        cv::normalize(temp_grad_mag, temp_grad_mag, 0, 1, cv::NORM_MINMAX);

        cv::max(content_grad_mag_, temp_grad_mag, content_grad_mag_);

    }
}

std::vector<int> seamCarving::findVerticalSeamContent() {
    cv::Mat1i backtrack = cv::Mat1i::zeros(cv::Size(width_, height_));

    computeGradMagContent();

    /* Initialize rolling array */
    std::vector<float> prev_row(width_);
    std::vector<float> curr_row(width_);
    content_grad_mag_.row(0).copyTo(curr_row);

    int idx;
    float min_energy, temp_energy;
    for (int i = 1; i < height_; ++i) {
        /* Swap rows */
        swap(prev_row, curr_row);

        /* Left edge */
        if (prev_row[0] <= prev_row[1] + abs(content_grad_mag_(i - 1, 0) - content_grad_mag_(i, 1))) {
            curr_row[0] = prev_row[0] + content_grad_mag_(i, 0) ;
            backtrack(i, 0) = 0;
        } else {
            curr_row[0] = prev_row[1] + content_grad_mag_(i, 0) + abs(content_grad_mag_(i - 1, 0) - content_grad_mag_(i, 1));
            backtrack(i, 0) = 1;
        }

        for (int j = 1; j < width_; ++j) {
            min_energy = prev_row[j - 1]
                           + abs(content_grad_mag_(i, j - 1) - content_grad_mag_(i, j + 1))
                           + abs(content_grad_mag_(i - 1, j) - content_grad_mag_(i, j - 1));
            backtrack(i, j) = j - 1;

            temp_energy = prev_row[j]
                            + abs(content_grad_mag_(i, j - 1) - content_grad_mag_(i, j + 1));
            if (temp_energy < min_energy) {
                min_energy = temp_energy;
                backtrack(i, j) = j;
            }

            if (j + 1 != width_) {
                temp_energy = prev_row[j + 1]
                              + abs(content_grad_mag_(i, j - 1) - content_grad_mag_(i, j + 1))
                              + abs(content_grad_mag_(i - 1, j) - content_grad_mag_(i, j + 1));
                if (temp_energy < min_energy) {
                    min_energy = temp_energy;
                    backtrack(i, j) = j + 1;
                }
            }

            curr_row[j] = min_energy + content_grad_mag_(i, j);
        }
    }

    /* Find the position of the smallest elements in the last row of the cumulative energy */
    int j = 0;
    for (int i = 0; i < width_; ++i)
        if (curr_row[i] < curr_row[j]) j = i;

    std::vector<int> seam(height_);
    seam[height_ - 1] = j;
    for (int i = height_ - 2; i >= 0; --i)
        seam[i] = backtrack(i, seam[i + 1]);

    return seam;
}

void seamCarving::removeVerticalSeamsContent(int k) {
    std::vector<std::vector<int>> seams_to_remove;

    boost::timer::progress_display pd(k);
    for (int step = 0; step < k; ++step) {
        ++pd;

        std::vector<int> seam = findVerticalSeamContent();
        seams_to_remove.push_back(seam);

        /* Used to fill empty space left. */
        cv::Mat dummy(1, 1, CV_8UC3, cv::Vec3b(0, 0, 0));

        /* Remove the seam. */
        cv::parallel_for_(cv::Range(0, frames_), [&](const cv::Range& range) {
            for (int f = range.start; f < range.end; ++f) {
                int j;
                cv::Mat new_row;

                for (int i = 0; i < height_; ++i) {
                    j = seam[i];
                    if (j + 1 != width_) {
                        gray_video_[f].row(i).colRange(j+ 1, width_).copyTo(new_row);
                        new_row.copyTo(gray_video_[f].row(i).colRange(j, width_ - 1));
                    }
                }
            }
        });

        --width_;

        cv::Rect roi(0, 0, width_, height_);
        for (auto &gray_image : gray_video_)
            gray_image = gray_image(roi);
    }

    /* Remove seams in the colored video. */
    cv::parallel_for_(cv::Range(0, frames_), [&](const cv::Range& range) {
        for (int f = range.start; f < range.end; ++f) {
            int j;
            cv::Mat new_row;

            int width = video_[f].cols;
            for (const auto &seam : seams_to_remove) {
                for (int i = 0; i < height_; ++i) {
                    j = seam[i];
                    if (j + 1 != width) {
                        video_[f].row(i).colRange(j + 1, width).copyTo(new_row);
                        new_row.copyTo(video_[f].row(i).colRange(j, width - 1));
                    }
                    video_[f](i, width - 1) = {0, 0, 0};
                }
                --width;
            }
        }
    });
}

std::vector<int> seamCarving::findHorizontalSeamContent() {
    cv::Mat1i backtrack = cv::Mat1i::zeros(cv::Size(width_, height_));

    computeGradMagContent();

    /* Initialize rolling array */
    std::vector<float> prev_col(height_);
    std::vector<float> curr_col(height_);
    content_grad_mag_.col(0).copyTo(curr_col);

    int idx;
    float min_energy, temp_energy;
    for (int j = 1; j < width_; ++j) {
        /* Swap columns */
        swap(prev_col, curr_col);

        /* Left edge */
        if (prev_col[0] <= prev_col[1] + abs(content_grad_mag_(0, j - 1) - content_grad_mag_(1, j))) {
            curr_col[0] = prev_col[0] + content_grad_mag_(0, j);
            backtrack(0, j) = 0;
        } else {
            curr_col[0] = prev_col[1] + abs(content_grad_mag_(0, j - 1) - content_grad_mag_(1, j));
            backtrack(0, j) = 1;
        }

        for (int i = 1; i < height_; ++i) {
            min_energy = prev_col[i - 1]
                            + abs(content_grad_mag_(i - 1, j) - content_grad_mag_(i, j - 1))
                            + abs(content_grad_mag_(i - 1, j) - content_grad_mag_(i + 1, j));
            backtrack(i, j) = i - 1;

            temp_energy = prev_col[i]
                          + abs(content_grad_mag_(i - 1, j) - content_grad_mag_(i + 1, j));
            if (temp_energy < min_energy) {
                min_energy = temp_energy;
                backtrack(i, j) = i;
            }

            if (i + 1 != height_) {
                temp_energy = prev_col[i + 1]
                              + abs(content_grad_mag_(i, j - 1) - content_grad_mag_(i + 1, j))
                              + abs(content_grad_mag_(i - 1, j) - content_grad_mag_(i + 1, j));
                if (temp_energy < min_energy) {
                    min_energy = temp_energy;
                    backtrack(i, j) = i + 1;
                }
            }

            curr_col[i] = min_energy + content_grad_mag_(i, j);
        }
    }

    /* Find the position of the smallest elements in the last column of the cumulative energy */
    int i = 0;
    for (int j = 0; j < height_; ++j) {
        if (curr_col[j] < curr_col[i])
            i = j;
    }

    std::vector<int> seam(width_);
    for (int j = width_ - 1; j >= 0; --j) {
        seam[j] = i;
        i = backtrack(i, j);
    }

    return seam;
}

void seamCarving::removeHorizontalSeamsContent(int k) {
    std::vector<std::vector<int>> seams_to_remove;

    boost::timer::progress_display pd(k);
    for (int step = 0; step < k; ++step) {
        ++pd;

        std::vector<int> seam = findHorizontalSeamContent();
        seams_to_remove.push_back(seam);

        /* Used to fill empty space left. */
        cv::Mat dummy(1, 1, CV_8UC3, cv::Vec3b(0, 0, 0));

        /* Remove the seam. */
        cv::parallel_for_(cv::Range(0, frames_), [&](const cv::Range& range) {
            for (int f = range.start; f < range.end; ++f) {
                int i;
                cv::Mat new_column;
                cv::Mat1f new_grad_x_column;
                for (int j = 0; j < width_; ++j) {
                    i = seam[j];
                    if (i + 1 != height_) {
                        gray_video_[f].rowRange(i + 1, height_).col(j).copyTo(new_column);
                        new_column.copyTo(gray_video_[f].rowRange(i, height_ - 1).col(j));
                    }
                    gray_video_[f](height_ - 1, j) = {0};
                }
            }
        });

        --height_;

        cv::Rect roi(0, 0, width_, height_);
        for (auto &gray_image : gray_video_)
            gray_image = gray_image(roi);
    }

    /* Remove seams in the colored video. */
    cv::parallel_for_(cv::Range(0, frames_), [&](const cv::Range& range) {
        for (int f = range.start; f < range.end; ++f) {
            int i;
            cv::Mat new_column;
            int height = video_[f].rows;
            for (const auto &seam : seams_to_remove) {
                for (int j = 0; j < width_; ++j) {
                    i = seam[j];
                    if (i + 1 != height) {
                        video_[f].rowRange(i + 1, height).col(j).copyTo(new_column);
                        new_column.copyTo(video_[f].rowRange(i, height - 1).col(j));
                    }
                }
                --height;
            }
        }
    });
}