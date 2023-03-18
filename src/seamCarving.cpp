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

    sc.loadImage("im.png");
    sc.showImage();

    /* Wait for a key press. */
    cv::waitKey(0);

    return 0;
}

bool seamCarving::loadImage(const std::string &link) {
    this->image = cv::imread(link, cv::IMREAD_COLOR);

    /* Check if the image was loaded successfully/ */
    if (image.empty()) {
        std::cerr << "Could not read the image file." << std::endl;
        return false;
    }

    return true;
}

void seamCarving::showImage() {
    if (!image.empty())
        cv::imshow("Image", image);
}