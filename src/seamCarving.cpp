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

/* The following pragma are used to removed deprecation warning from boost
 * header files. Using them avoid to remove this warning from the entire project.
 */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include <boost/program_options.hpp>
#pragma GCC diagnostic pop

int main(int argc, char** argv) {
    boost::program_options::options_description desc("Allowed options");
    desc.add_options()
            ("help,h", "print the help message")
            ("input,i", boost::program_options::value<std::string>(), "set the input file")
            ("output,o", boost::program_options::value<std::string>(), "set the output file")
            ("show_result,r", "show the output file")
            ("show_energy_map,e", "show the energy map");

    boost::program_options::variables_map vm;
    boost::program_options::store(boost::program_options::parse_command_line(argc, argv, desc), vm);
    try{
        boost::program_options::notify(vm);
    }
    catch (std::exception& e) {
        std::cout << "Error: " << e.what() << std::endl;
        std::cout << desc << std::endl;
        return 1;
    }

    boost::program_options::store(boost::program_options::parse_command_line(argc, argv, desc), vm);

    if (vm.count("help")) {
        std::cout << desc << "\n";
        return 1;
    }

    std::string input_file;
    if (vm.count("input")) {
        input_file = vm["input"].as<std::string>();
    } else{
        std::cerr << "--input option is requiered";
        return 0;
    }

    std::string output_file;
    if (vm.count("output")) {
        output_file = vm["output"].as<std::string>();
    } else{
        std::cerr << "--output option is requiered";
        return 0;
    }

    bool show_result     = vm.count("show_result");
    bool show_energy_map = vm.count("show_energy_map");

    cv::setNumThreads(8);

    seamCarving sc;
    sc.loadContent(input_file);

    if (show_energy_map) {
        sc.computeGradMagContent();
        sc.showcontent(std::vector<cv::Mat1f>(1, sc.content_grad_mag_), "Energy map");
    }

    auto start = std::chrono::high_resolution_clock::now();
    sc.resizeContent(0.98, .98);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Time taken : " << duration.count() << "ms\n";

    sc.saveContent(output_file, sc.video_);

    if (show_result)
        sc.showcontent(sc.video_, "Result");

    return 0;
}

void seamCarving::resizeContent(double horizontal_factor, double vertical_factor) {
    if (vertical_factor < 1) {
        removeHorizontalSeamsContent(height_full_ - height_full_ * vertical_factor);
        cv::Rect roi(0, 0, width_full_, height_full_);
        for (auto &image : video_)
            image = image(roi);
    }

    if (horizontal_factor < 1) {
        removeVerticalSeamsContent(width_full_ - width_full_ * horizontal_factor);
        cv::Rect roi(0, 0, width_full_, height_full_);
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
        resize(gray_video_[i], gray_video_[i], cv::Size(gray_video_[i].cols/2, gray_video_[i].rows/2));
    }

    width_  = gray_video_[0].cols;
    height_ = gray_video_[0].rows;
    width_full_  = video_[0].cols;
    height_full_ = video_[0].rows;
    frames_ = video_.size();

    return true;
}

template <typename T>
void seamCarving::saveContent(const std::string &link, const std::vector<cv::Mat_<T>> &video) {
    if (video.size() == 1) {
        cv::normalize(video[0], video[0], 0, 255, cv::NORM_MINMAX);
        cv::imwrite(link, video[0]);
        return;
    }

    int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
    cv::Size frame_size = video[0].size();

    cv::VideoWriter video_obj(link, fourcc, fps_, frame_size, true);

    if (!video_obj.isOpened()) {
        std::cerr << "Could not open the output video file for write.\n";
        return;
    }

    for (int i = 0; i < frames_; i++)
        video_obj.write(video[i]);

    video_obj.release();
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
    cv::Mat temp_grad_mag        = cv::Mat1f::zeros(height_, width_);
    cv::Mat content_grad_mag_max = cv::Mat1f::zeros(height_, width_);
    cv::Mat temporal             = cv::Mat1f::zeros(height_, width_);

    std::vector<cv::Mat> all_nrj(frames_);

    std::mutex mutex;
    cv::parallel_for_(cv::Range(0, frames_), [&](const cv::Range& range) {
        for (int f = range.start; f < range.end; ++f) {
            /* Compute gradients using Sobel operator */
            cv::Mat1f grad_x, grad_y;
            cv::Sobel(gray_video_[f], grad_x, CV_32F, 1, 0);
            cv::Sobel(gray_video_[f], grad_y, CV_32F, 0, 1);

            /* Compute gradient magnitude */
            mutex.lock();
            cv::magnitude(grad_x, grad_y, temp_grad_mag);

            temp_grad_mag.copyTo(all_nrj[f]);
            cv::max(content_grad_mag_max, temp_grad_mag, content_grad_mag_max);
            mutex.unlock();
        }
    });

    if (frames_ > 1) {
        for (int f = 0; f < frames_ - 1; ++f) {
            temp_grad_mag = all_nrj[f + 1] - all_nrj[f];
            cv::normalize(temp_grad_mag, temp_grad_mag, 0, 1, cv::NORM_MINMAX);
            cv::max(temporal, temp_grad_mag, temporal);
        }
    }

    cv::normalize(content_grad_mag_max, content_grad_mag_max, 0, 1, cv::NORM_MINMAX);
    cv::normalize(temporal, temporal, 0, 1, cv::NORM_MINMAX);

    content_grad_mag_max += temporal;
    content_grad_mag_ = content_grad_mag_max;
    cv::normalize(content_grad_mag_, content_grad_mag_, 0, 1, cv::NORM_MINMAX);
}

std::vector<int> seamCarving::findVerticalSeamContent() {
    cv::Mat1i backtrack = cv::Mat1i::zeros(cv::Size(width_, height_));

    computeGradMagContent();

    /* Initialize rolling array */
    std::vector<float> rolling_array(width_);
    content_grad_mag_.row(0).copyTo(rolling_array);

    float prev_row_j;
    float min_energy, temp_energy;
    for (int i = 1; i < height_; ++i) {
        /* Left edge */
        if (rolling_array[0] <= rolling_array[1] + abs(content_grad_mag_(i - 1, 0) - content_grad_mag_(i, 1))) {
            temp_energy = rolling_array[0] + content_grad_mag_(i, 0) ;
            backtrack(i, 0) = 0;
        } else {
            temp_energy = rolling_array[1] + content_grad_mag_(i, 0) + abs(content_grad_mag_(i - 1, 0) - content_grad_mag_(i, 1));
            backtrack(i, 0) = 1;
        }

        prev_row_j = rolling_array[0]; /* Store previous row value for the rolling array */
        rolling_array[0] = temp_energy;

        for (int j = 1; j < width_; ++j) {
            min_energy = prev_row_j
                           + abs(content_grad_mag_(i, j - 1) - content_grad_mag_(i, j + 1))
                           + abs(content_grad_mag_(i - 1, j) - content_grad_mag_(i, j - 1));
            backtrack(i, j) = j - 1;

            temp_energy = rolling_array[j]
                            + abs(content_grad_mag_(i, j - 1) - content_grad_mag_(i, j + 1));
            if (temp_energy < min_energy) {
                min_energy = temp_energy;
                backtrack(i, j) = j;
            }

            if (j + 1 != width_) {
                temp_energy = rolling_array[j + 1]
                              + abs(content_grad_mag_(i, j - 1) - content_grad_mag_(i, j + 1))
                              + abs(content_grad_mag_(i - 1, j) - content_grad_mag_(i, j + 1));
                if (temp_energy < min_energy) {
                    min_energy = temp_energy;
                    backtrack(i, j) = j + 1;
                }
            }

            prev_row_j = rolling_array[j]; /* Store previous row value for the rolling array */
            rolling_array[j] = min_energy + content_grad_mag_(i, j);
        }
    }

    /* Find the position of the smallest elements in the last row of the cumulative energy */
    int j = std::min_element(rolling_array.begin(), rolling_array.end()) - rolling_array.begin();

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
        width_full_ -= 2;

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
                    j = 2*seam[i];
                    if (j + 1 != width) {
                        video_[f].row(2 * i).colRange(j + 2, width).copyTo(new_row);
                        new_row.copyTo(video_[f].row(2 * i).colRange(j, width - 2));
                        video_[f].row(2 * i + 1).colRange(j + 2, width).copyTo(new_row);
                        new_row.copyTo(video_[f].row(2 * i + 1).colRange(j, width - 2));
                    }
                    video_[f](i, width - 1) = {0, 0, 0};
                }
                width -= 2;
            }
        }
    });
}

std::vector<int> seamCarving::findHorizontalSeamContent() {
    cv::Mat1i backtrack = cv::Mat1i::zeros(cv::Size(width_, height_));

    computeGradMagContent();

    /* Initialize rolling array */
    std::vector<float> rolling_array(height_);
    content_grad_mag_.col(0).copyTo(rolling_array);

    float prev_col_i;
    float min_energy, temp_energy;
    for (int j = 1; j < width_; ++j) {
        /* Left edge */
        if (rolling_array[0] <= rolling_array[1] + abs(content_grad_mag_(0, j - 1) - content_grad_mag_(1, j))) {
            temp_energy = rolling_array[0] + content_grad_mag_(0, j);
            backtrack(0, j) = 0;
        } else {
            temp_energy = rolling_array[1] + abs(content_grad_mag_(0, j - 1) - content_grad_mag_(1, j));
            backtrack(0, j) = 1;
        }

        prev_col_i = rolling_array[0]; /* Store previous col value for the rolling array */
        rolling_array[0] = temp_energy;

        for (int i = 1; i < height_; ++i) {
            min_energy = prev_col_i
                            + abs(content_grad_mag_(i - 1, j) - content_grad_mag_(i, j - 1))
                            + abs(content_grad_mag_(i - 1, j) - content_grad_mag_(i + 1, j));
            backtrack(i, j) = i - 1;

            temp_energy = rolling_array[i]
                          + abs(content_grad_mag_(i - 1, j) - content_grad_mag_(i + 1, j));
            if (temp_energy < min_energy) {
                min_energy = temp_energy;
                backtrack(i, j) = i;
            }

            if (i + 1 != height_) {
                temp_energy = rolling_array[i + 1]
                              + abs(content_grad_mag_(i, j - 1) - content_grad_mag_(i + 1, j))
                              + abs(content_grad_mag_(i - 1, j) - content_grad_mag_(i + 1, j));
                if (temp_energy < min_energy) {
                    min_energy = temp_energy;
                    backtrack(i, j) = i + 1;
                }
            }

            prev_col_i = rolling_array[i]; /* Store previous col value for the rolling array */
            rolling_array[i] = min_energy + content_grad_mag_(i, j);
        }
    }

    /* Find the position of the smallest elements in the last column of the cumulative energy */
    int i = std::min_element(rolling_array.begin(), rolling_array.end()) - rolling_array.begin();

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
        height_full_ -= 2;

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
                    i = 2*seam[j];
                    if (i + 2 != height) {
                        video_[f].rowRange(i + 2, height).col(2 * j).copyTo(new_column);
                        new_column.copyTo(video_[f].rowRange(i, height - 2).col(2 * j));
                        video_[f].rowRange(i + 2, height).col(2 * j + 1).copyTo(new_column);
                        new_column.copyTo(video_[f].rowRange(i, height - 2).col(2 * j + 1));
                    }
                }
                height -= 2;
            }
        }
    });
}