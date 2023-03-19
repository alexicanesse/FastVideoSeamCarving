/*
 * This file is part of seamCarving which is released under GNU General Public License v3.0.
 * See file LICENSE or go to https://github.com/alexicanesse/ChineseCheckers/blob/main/LICENSE for full license details.
 * Copyright 2022 - ENS de Lyon
 */


/* seamCarving.hpp */
#include "seamCarving.hpp"

/* C++ libraries */
#include <opencv2/opencv.hpp>
#include <boost/timer/timer.hpp>
#include <boost/timer/progress_display.hpp>

int main() {
    seamCarving sc;

    sc.loadImage("im.jpg");

    sc.showImage(sc.image_, "Image");

    sc.resize(0.95, 0.5);

    sc.showImage(sc.image_, "Image");
    sc.computeGradMagImage();
    sc.showImage(sc.image_grad_mag_, "Magnitude");

    return 0;
}

void seamCarving::resize(double horizontal_factor, double vertical_factor) {
    int r = image_.rows;
    int c = image_.cols;

    if (horizontal_factor < 1)
        removeHorizontalSeams(r - r * horizontal_factor);
    else
        addHorizontalSeams(r * horizontal_factor - r);

    if (vertical_factor < 1)
        removeVerticalSeams(c - c * vertical_factor);
    else
        addVerticalSeams(c * vertical_factor - c);
}

bool seamCarving::loadImage(const std::string &link) {
    image_ = cv::imread(link, cv::IMREAD_COLOR);

    /* Check if the image was loaded successfully/ */
    if (image_.empty()) {
        std::cerr << "Could not read the image file." << std::endl;
        return false;
    }

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
    cv::Mat1b gray_image;
    cv::cvtColor(image_, gray_image, cv::COLOR_BGR2GRAY);

    /* Compute gradients using Sobel operator */
    cv::Mat1f grad_x, grad_y;
    cv::Sobel(gray_image, grad_x, CV_32F, 1, 0);
    cv::Sobel(gray_image, grad_y, CV_32F, 0, 1);

    /* Compute gradient magnitude */
    cv::Mat1f grad_mag;
    cv::magnitude(grad_x, grad_y, grad_mag);

    /* Normalize the gradient magnitude image */
    cv::normalize(grad_mag, image_grad_mag_, 0, 1, cv::NORM_MINMAX);
}

cv::Mat1s seamCarving::findVerticalSeam() {
    int r = image_.rows;
    int c = image_.cols;
    cv::Mat1i backtrack = cv::Mat1i::zeros(image_.size());

    computeGradMagImage();

    int idx;
    double min_energy;
    for (int i = 1; i < r; ++i) {
        /* Left edge */
        if (image_grad_mag_(i - 1, 0) <= image_grad_mag_(i - 1, 1)) {
            image_grad_mag_(i, 0) += image_grad_mag_(i - 1, 0);
            backtrack(i, 0) = 0;
        } else {
            image_grad_mag_(i, 0) += image_grad_mag_(i - 1, 1);
            backtrack(i, 0) = 1;
        }
        for (int j = 1; j < c; ++j) {
            min_energy = image_grad_mag_(i - 1, j - 1);
            backtrack(i, j) = j - 1;
            for (int jk = j; jk < j + 2 && jk < c; ++jk) {
                if (image_grad_mag_(i - 1, jk) <= min_energy) {
                    min_energy = image_grad_mag_(i - 1, jk);
                    backtrack(i, j) = jk;
                }
            }
            image_grad_mag_(i, j) += min_energy;
        }
    }
    return backtrack;
}

void seamCarving::removeVerticalSeams(int k) {
    boost::timer::progress_display pd(k);
    for (int step = 0; step < k; ++step) {
        ++pd;

        int r = image_.rows;
        int c = image_.cols;

        /* Create a mask to mark pixels for deletion */
        cv::Mat1b mask = cv::Mat::ones(image_.size(), CV_8U);

        auto backtrack = findVerticalSeam();

        /* Find the position of the smallest element in the last row of M */
        int j = 0;
        for (int i = 0; i < c; ++i) {
            if (image_grad_mag_(r - 1, i) < image_grad_mag_(r - 1, j))
                j = i;
        }

        /* Find the seam */
        for (int i = r - 1; i >= 0; --i) {
            mask(i, j) = 0;
            j = backtrack(i, j);
        }

        /* Delete all the pixels marked False in the mask */
        cv::Mat3b new_image(r, c - 1);
        for (int i = 0; i < r; ++i) {
            int b = 0;
            for (int j = 0; j < c; ++j) {
                if (mask(i, j))
                    new_image(i, j - b) = image_(i, j);
                else
                    ++b;
            }
        }

        image_ = new_image;
    }
}

void seamCarving::addVerticalSeams(int k) {
    cv::Mat kernel = cv::Mat::ones(3, 3, CV_32F) / 9.0;

    boost::timer::progress_display pd(k);
    for (int step = 0; step < k; ++step) {
        ++pd;

        cv::Mat3b filteredImage;
        cv::filter2D(image_, filteredImage, -1, kernel);

        int r = image_.rows;
        int c = image_.cols;

        /* Create a mask to mark pixels for deletion */
        cv::Mat1b mask = cv::Mat::ones(image_.size(), CV_8U);

        auto backtrack = findVerticalSeam();
        /* Find the position of the smallest elements in the last row of the cumulative energy */
        int j = 0;
        for (int i = 0; i < c; ++i) {
            if (image_grad_mag_(r - 1, i) < image_grad_mag_(r - 1, j))
                j = i;
        }

        /* Find the seam */
        for (int i = r - 1; i >= 0; --i) {
            mask(i, j) = 0;
            j = backtrack(i, j);
        }

        /* Duplicate pixels using the mask */
        cv::Mat3b new_image(r, c + 1);
        for (int i = 0; i < r; ++i) {
            int b = 0;
            for (int j = 0; j < c; ++j) {
                new_image(i, j + b) = image_(i, j);
                if (!mask(i, j)) {
                    new_image(i, j + ++b) = filteredImage(i, j);
                }
            }
        }
        image_ = new_image;
    }
}

cv::Mat1s seamCarving::findHorizontalSeam() {
    int r = image_.rows;
    int c = image_.cols;
    cv::Mat1i backtrack = cv::Mat1i::zeros(image_.size());

    computeGradMagImage();

    double min_energy;
    for (int j = 1; j < c; ++j) {
        /* Left edge */
        if (image_grad_mag_(0, j - 1) <= image_grad_mag_(1, j - 1)) {
            image_grad_mag_(0, j) += image_grad_mag_(0, j - 1);
            backtrack(0, j) = 0;
        } else {
            image_grad_mag_(0, j) += image_grad_mag_(1, j - 1);
            backtrack(0, j) = 1;
        }
        for (int i = 1; i < r; ++i) {
            min_energy = image_grad_mag_(i - 1, j - 1);
            backtrack(i, j) = i - 1;
            for (int ik = i; ik < i + 2 && ik < r; ++ik) {
                if (image_grad_mag_(ik, j - 1) <= min_energy) {
                    min_energy = image_grad_mag_(ik, j - 1);
                    backtrack(i, j) = ik;
                }
            }
            image_grad_mag_(i, j) += min_energy;
        }
    }
    return backtrack;
}

void seamCarving::removeHorizontalSeams(int k) {
    boost::timer::progress_display pd(k);
    for (int step = 0; step < k; ++step) {
        ++pd;

        int r = image_.rows;
        int c = image_.cols;

        /* Create a mask to mark pixels for deletion */
        cv::Mat1b mask = cv::Mat::ones(image_.size(), CV_8U);

        auto backtrack = findHorizontalSeam();

        /* Find the position of the smallest element in the last row of M */
        int i = 0;
        for (int j = 0; j < r; ++j) {
            if (image_grad_mag_(j, c - 1) < image_grad_mag_(i, c - 1))
                i = j;
        }

        /* Find the seam */
        for (int j = c - 1; j >= 0; --j) {
            mask(i, j) = 0;
            i = backtrack(i, j);
        }

        /* Delete all the pixels marked False in the mask */
        cv::Mat3b new_image(r - 1, c);
        for (int j = 0; j < c; ++j) {
            int b = 0;
            for (int i = 0; i < r; ++i) {
                if (mask(i, j))
                    new_image(i - b, j) = image_(i, j);
                else
                    ++b;
            }
        }

        image_ = new_image;
    }
}

void seamCarving::addHorizontalSeams(int k) {
    cv::Mat kernel = cv::Mat::ones(3, 3, CV_32F) / 9.0;

    boost::timer::progress_display pd(k);
    for (int step = 0; step < k; ++step) {
        ++pd;

        cv::Mat3b filteredImage;
        cv::filter2D(image_, filteredImage, -1, kernel);

        int r = image_.rows;
        int c = image_.cols;

        /* Create a mask to mark pixels for deletion */
        cv::Mat1b mask = cv::Mat::ones(image_.size(), CV_8U);

        auto backtrack = findVerticalSeam();
        /* Find the position of the smallest elements in the last row of the cumulative energy */
        int j = 0;
        for (int i = 0; i < c; ++i) {
            if (image_grad_mag_(r - 1, i) < image_grad_mag_(r - 1, j))
                j = i;
        }

        /* Find the seam */
        for (int i = r - 1; i >= 0; --i) {
            mask(i, j) = 0;
            j = backtrack(i, j);
        }

        /* Duplicate pixels using the mask */
        cv::Mat3b new_image(r, c + 1);
        for (int i = 0; i < r; ++i) {
            int b = 0;
            for (int j = 0; j < c; ++j) {
                new_image(i, j + b) = image_(i, j);
                if (!mask(i, j)) {
                    new_image(i, j + ++b) = filteredImage(i, j);
                }
            }
        }
        image_ = new_image;
    }
}