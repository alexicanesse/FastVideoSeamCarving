/*
 * This file is part of seamCarving which is released under GNU General Public License v3.0.
 * See file LICENSE or go to https://github.com/alexicanesse/ChineseCheckers/blob/main/LICENSE for full license details.
 * Copyright 2022 - ENS de Lyon
 */


/* seamCarving.hpp */
#include "seamCarving.hpp"

/* C++ libraries */
#include <opencv2/opencv.hpp>

int main() {
    seamCarving sc;

    sc.loadImage("im.jpg");

    sc.computeGradMagImage();

    sc.showImage(sc.image_, "Image");
    sc.showImage(sc.image_grad_mag_, "Magnitude");

    return 0;
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