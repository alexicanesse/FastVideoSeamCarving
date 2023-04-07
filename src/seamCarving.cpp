/*
 * This file is part of seamCarving which is released under GNU General Public License v3.0.
 * See file LICENSE or go to https://github.com/alexicanesse/ChineseCheckers/blob/main/LICENSE for full license details.
 * Copyright 2022 - ENS de Lyon
 */


/* seamCarving.hpp */
#include "seamCarving.hpp"

/* C++ libraries */
#include <opencv2/opencv.hpp>
#include <chrono>
#include <boost/timer/progress_display.hpp>

int main() {
    cv::setNumThreads(8);

    seamCarving sc;

    sc.loadImage("im.jpg");

    sc.showImage(sc.image_, "Image");

    auto start = std::chrono::high_resolution_clock::now();
    sc.resize(0.5, 0.95);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "Time taken to resize the image: " << duration.count() << " ms" << std::endl;

    sc.showImage(sc.image_, "Image");
    sc.computeGradMagImage();
    sc.showImage(sc.image_grad_mag_, "Magnitude");

    return 0;
}

void seamCarving::resize(double horizontal_factor, double vertical_factor) {
    if (vertical_factor < 1) {
        removeHorizontalSeams(height_ - height_ * vertical_factor);
        cv::Rect roi(0, 0, width_, height_);
        image_ = image_(roi);
    }

    if (horizontal_factor < 1) {
        removeVerticalSeams(width_ - width_ * horizontal_factor);
        cv::Rect roi(0, 0, width_, height_);
        image_ = image_(roi);
    }
}

bool seamCarving::loadImage(const std::string &link) {
    image_ = cv::imread(link, cv::IMREAD_COLOR);

    /* Check if the image was loaded successfully */
    if (image_.empty()) {
        std::cerr << "Could not read the image file." << std::endl;
        return false;
    }

    width_  = image_.cols;
    height_ = image_.rows;

    return true;
}

void seamCarving::showImage(const cv::Mat &image, const std::string &title) {
    if (!image.empty())
        cv::imshow(title, image);

    /* Wait for a key press. */
    cv::waitKey(0);
}

void seamCarving::computeGradMagImage() {
    /* Convert the image to grayscale */
    cv::cvtColor(image_, gray_image_, cv::COLOR_BGR2GRAY);

    /* Compute gradients using Sobel operator */
    cv::Mat1f grad_x, grad_y;
    cv::Sobel(gray_image_, grad_x, CV_32F, 1, 0);
    cv::Sobel(gray_image_, grad_y, CV_32F, 0, 1);

    /* Compute gradient magnitude */
    cv::Mat1f grad_mag;
    cv::magnitude(grad_x, grad_y, grad_mag);

    /* Normalize the gradient magnitude image and store it */
    cv::normalize(grad_mag, image_grad_mag_, 0, 1, cv::NORM_MINMAX);
}

std::vector<int> seamCarving::findVerticalSeam() {
    cv::Mat1i backtrack = cv::Mat1i::zeros(cv::Size(width_, height_));

    computeGradMagImage();

    int idx;
    double min_energy;
    for (int i = 1; i < height_; ++i) {
        /* Left edge */
        if (image_grad_mag_(i - 1, 0) <= image_grad_mag_(i - 1, 1)) {
            image_grad_mag_(i, 0) += image_grad_mag_(i - 1, 0);
            backtrack(i, 0) = 0;
        } else {
            image_grad_mag_(i, 0) += image_grad_mag_(i - 1, 1);
            backtrack(i, 0) = 1;
        }

        for (int j = 1; j < width_; ++j) {
            min_energy = image_grad_mag_(i - 1, j - 1);
            backtrack(i, j) = j - 1;
            for (int jk = j; jk < j + 2 && jk < width_; ++jk) {
                if (image_grad_mag_(i - 1, jk) <= min_energy) {
                    min_energy = image_grad_mag_(i - 1, jk);
                    backtrack(i, j) = jk;
                }
            }
            image_grad_mag_(i, j) += min_energy;
        }
    }

    /* Find the position of the smallest elements in the last row of the cumulative energy */
    int j = 0;
    for (int i = 0; i < width_; ++i) {
        if (image_grad_mag_(height_ - 1, i) < image_grad_mag_(height_ - 1, j))
            j = i;
    }

    std::vector<int> seam(height_);
    for (int i = height_ - 1; i >= 0; --i) {
        seam[i] = j;
        j = backtrack(i, j);
    }

    return seam;
}

void seamCarving::removeVerticalSeams(int k) {
    boost::timer::progress_display pd(k);
    for (int step = 0; step < k; ++step) {
        ++pd;

        std::vector<int> seam = findVerticalSeam();

        /* Used to fill empty space left. */
        cv::Mat dummy(1, image_.cols - width_ + 1, CV_8UC3, cv::Vec3b(0, 0, 0));

        /* Remove the seam. */
        cv::parallel_for_(cv::Range(0, height_), [&](const cv::Range& range) {
            for (int i = range.start; i < range.end; ++i) {
                cv::Mat new_row;
                int j = seam[i];
                if (j + 1 != width_) {
                    image_.row(i).colRange(j + 1, width_).copyTo(new_row);
                    new_row.copyTo(image_.row(i).colRange(j, width_ - 1));
                }
                image_(i, width_ - 1) = {0, 0, 0};
            }
        });

        --width_;
    }
}

std::vector<int> seamCarving::findHorizontalSeam() {
    cv::Mat1i backtrack = cv::Mat1i::zeros(cv::Size(width_, height_));

    computeGradMagImage();

    double min_energy;
    for (int j = 1; j < width_; ++j) {
        /* Left edge */
        if (image_grad_mag_(0, j - 1) <= image_grad_mag_(1, j - 1)) {
            image_grad_mag_(0, j) += image_grad_mag_(0, j - 1);
            backtrack(0, j) = 0;
        } else {
            image_grad_mag_(0, j) += image_grad_mag_(1, j - 1);
            backtrack(0, j) = 1;
        }
        for (int i = 1; i < height_; ++i) {
            min_energy = image_grad_mag_(i - 1, j - 1);
            backtrack(i, j) = i - 1;
            for (int ik = i; ik < i + 2 && ik < height_; ++ik) {
                if (image_grad_mag_(ik, j - 1) <= min_energy) {
                    min_energy = image_grad_mag_(ik, j - 1);
                    backtrack(i, j) = ik;
                }
            }
            image_grad_mag_(i, j) += min_energy;
        }
    }

    /* Find the position of the smallest elements in the last column of the cumulative energy */
    int i = 0;
    for (int j = 0; j < height_; ++j) {
        if (image_grad_mag_(height_ - 1, j) < image_grad_mag_(height_ - 1, j))
            i = j;
    }

    std::vector<int> seam(width_);
    for (int j = width_ - 1; j >= 0; --j) {
        seam[j] = i;
        i = backtrack(i, j);
    }

    return seam;
}

void seamCarving::removeHorizontalSeams(int k) {
    boost::timer::progress_display pd(k);
    for (int step = 0; step < k; ++step) {
        ++pd;

        std::vector<int> seam = findHorizontalSeam();

        /* Used to fill empty space left. */
        cv::Mat dummy(image_.rows - height_ + 1, 1, CV_8UC3, cv::Vec3b(0, 0, 0));

        /* Remove the seam. */
        cv::parallel_for_(cv::Range(0, width_), [&](const cv::Range& range) {
            for (int j = range.start; j < range.end; ++j) {
                cv::Mat new_column;
                int i = seam[j];
                if (i + 1 != height_) {
                    image_.rowRange(1, height_).col(j).copyTo(new_column);
                    new_column.copyTo(image_.rowRange(1, height_).col(j));
                }
                image_(height_ - 1, j) = {0, 0, 0};
            }
        });

        --height_;
    }
}