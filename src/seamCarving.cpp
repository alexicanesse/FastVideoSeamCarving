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
    /********
     * Handle command line options.
     ********/

    /* Define command line options using the boost library */
    boost::program_options::options_description desc("Allowed options");
    desc.add_options()
            ("help,h", "print the help message")
            ("input,i", boost::program_options::value<std::string>(), "set the input file")
            ("output,o", boost::program_options::value<std::string>(), "set the output file")
            ("h_scale", boost::program_options::value<float>(), "set the horizontal scaling factor (must be <= 1 and >= 0)")
            ("v_scale", boost::program_options::value<float>(), "set the vertical scaling factor (must be <= 1 and >= 0)")
            ("show_result,r", "show the output file")
            ("show_energy_map,e", "show the energy map");

    /* Parse command line arguments and store them in a variable map */
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

    /* If the help option is used, print the help and exit */
    if (vm.count("help")) {
        std::cout << desc << "\n";
        return 1;
    }

    /* Get output file from the command line arguments, or print an error message if it's missing */
    std::string input_file;
    if (vm.count("input")) {
        input_file = vm["input"].as<std::string>();
    } else{
        std::cerr << "--input option is requiered";
        return 1;
    }


    std::string output_file;
    if (vm.count("output")) {
        output_file = vm["output"].as<std::string>();
    } else{
        std::cerr << "--output option is requiered";
        return 1;
    }

    /* Get horizontal scaling factor from the command line arguments, or set it to 1 if it's missing or invalid */
    float h_scale = 1;
    if (vm.count("h_scale")) {
        if (vm["h_scale"].as<float>() <= 1 and vm["h_scale"].as<float>() >= 0)
            h_scale = vm["h_scale"].as<float>();
        else
            std::cout << "Invalide h_scale argument ignored.\n";
    }

    /* Get vertical scaling factor from the command line arguments, or set it to 1 if it's missing or invalid */
    float v_scale = 1;
    if (vm.count("v_scale")) {
        if (vm["v_scale"].as<float>() <= 1 and vm["v_scale"].as<float>() >= 0)
            v_scale = vm["v_scale"].as<float>();
        else
            std::cout << "Invalide v_scale argument ignored.\n";
    }

    /* Get and set show_result and show_energy_map values */
    bool show_result     = vm.count("show_result");
    bool show_energy_map = vm.count("show_energy_map");

    /* Sets the number of threads to be used by OpenCV */
    cv::setNumThreads(8);

    /* Creates an instance of the seamCarving class, which contains the algorithms to perform seam carving */
    seamCarving sc;
    /* Loads the content (an image or video) from the input file specified in the command line. */
    sc.loadContent(input_file);

    /* If the user specified the option to show the energy map, computes the gradient magnitude of the content and shows it. */
    if (show_energy_map) {
        sc.computeGradMagContent();
        sc.showcontent(std::vector<cv::Mat1f>(1, sc.content_grad_mag_), "Energy map");
    }

    /* Measures the time it takes to resize the content using seam carving. */
    auto start = std::chrono::high_resolution_clock::now();
    /* Actually resizes the content */
    sc.resizeContent(h_scale, v_scale);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    /* Prints the time taken to resize the content. */
    std::cout << "Time taken : " << duration.count() << "ms\n";

    /* Saves the result (image or video) to the output file specified in the command line. */
    sc.saveContent(output_file, sc.video_);

    /* If the user specified the option to show the result, displays it. */
    if (show_result)
        sc.showcontent(sc.video_, "Result");

    Returns 0 to indicate successful execution.
    return 0;
}

void seamCarving::resizeContent(double horizontal_factor, double vertical_factor) {
    if (vertical_factor < 1) {
        std::cout << "Vertical:\n";
        removeHorizontalSeamsContent(height_ - height_ * vertical_factor);

        /* Remove the black bars */
        cv::Rect roi(0, 0, width_full_, height_full_);
        for (auto &image : video_)
            image = image(roi);
    }

    if (horizontal_factor < 1) {
        std::cout << "Horizontal:\n";
        removeVerticalSeamsContent(width_ - width_ * horizontal_factor);

        /* Remove the black bars */
        cv::Rect roi(0, 0, width_full_, height_full_);
        for (auto &image : video_)
            image = image(roi);
    }
}

bool seamCarving::loadContent(const std::string &link) {
    /* Open the video file */
    cv::VideoCapture capture(link);
    cv::Mat3b frame;

    /* Get the fps of the video and initialize the video container */
    fps_ = capture.get(cv::CAP_PROP_FPS);
    if ((int) capture.get(cv::CAP_PROP_FRAME_COUNT) >= 1)
        video_.resize(capture.get(cv::CAP_PROP_FRAME_COUNT));
    else
        video_.resize(1);

    /* Check if the video file was successfully opened */
    if(!capture.isOpened()) {
        std::cerr << "Error when reading the file\n";
        return false;
    }

    /* Create a window for displaying the frames (for testing purposes) */
    cv::namedWindow( "test", 1);

    /* Read each frame from the video file and add it to the video container */
    for (int i = 0; i < video_.size(); ++i) {
        capture >> frame;

        /* If there are no more frames to read, resize the video container and exit the loop */
        if(frame.empty()) {
            video_.resize(i);
            break;
        }

        frame.copyTo(video_[i]);
    }

    /* Create a grayscale version of each frame in the video container */
    gray_video_.resize(video_.size());
    for (int i = 0; i < gray_video_.size(); ++i) {
        cv::cvtColor(video_[i], gray_video_[i], cv::COLOR_BGR2GRAY);
        /* Downsample the grayscale image to improve performance */
        resize(gray_video_[i], gray_video_[i], cv::Size(gray_video_[i].cols/2, gray_video_[i].rows/2));
    }

    /* Set the width, height, and number of frames of the video */
    width_  = gray_video_[0].cols;
    height_ = gray_video_[0].rows;
    width_full_  = video_[0].cols;
    height_full_ = video_[0].rows;
    frames_ = video_.size();

    return true;
}

template <typename T>
void seamCarving::saveContent(const std::string &link, const std::vector<cv::Mat_<T>> &video) {
    /* If there is only one frame, save it as an image */
    if (video.size() == 1) {
        cv::normalize(video[0], video[0], 0, 255, cv::NORM_MINMAX);
        cv::imwrite(link, video[0]);
        return;
    }

    /* Initialize a video writer object with the given properties */
    int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
    cv::Size frame_size = video[0].size();

    cv::VideoWriter video_obj(link, fourcc, fps_, frame_size, true);

    /* Check if the video writer object was successfully initialized */
    if (!video_obj.isOpened()) {
        std::cerr << "Could not open the output video file for write.\n";
        return;
    }

    /* Write each frame to the output file */
    for (int i = 0; i < frames_; i++)
        video_obj.write(video[i]);

    /* Release the video writer object */
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

        cv::waitKey(1); /* Waits to display frame */
        std::this_thread::sleep_for(std::chrono::milliseconds((delay - std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start1).count())));
    }
    cv::imshow(title, video[video.size() - 1]);
    cv::waitKey(); /* Wait for a keypress before closing the window */
}

void seamCarving::computeGradMagContent() {
    /* Creates buffer matrices */
    cv::Mat temp_grad_mag        = cv::Mat1f::zeros(height_, width_);
    cv::Mat content_grad_mag_max = cv::Mat1f::zeros(height_, width_);
    cv::Mat temporal             = cv::Mat1f::zeros(height_, width_);

    /* Will store the gradient magnitude for each video frame (used for temporal derivative */
    std::vector<cv::Mat> all_nrj(frames_);

    /* Will synchronize access to shared variables between threads. */
    std::mutex mutex;
    /* Distributes the loop over the threads */
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

    /* calculates the temporal gradient if it is a video */
    if (frames_ > 1) {
        for (int f = 0; f < frames_ - 1; ++f) {
            temp_grad_mag = all_nrj[f + 1] - all_nrj[f];
            cv::normalize(temp_grad_mag, temp_grad_mag, 0, 1, cv::NORM_MINMAX);
            cv::max(temporal, temp_grad_mag, temporal);
        }
    }

    /* Normalize the results to be able to compare them */
    cv::normalize(content_grad_mag_max, content_grad_mag_max, 0, 1, cv::NORM_MINMAX);
    cv::normalize(temporal, temporal, 0, 1, cv::NORM_MINMAX);

    /* Combine the results into the final energy map */
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

        /* Distributes the loop over the threads */
        cv::parallel_for_(cv::Range(0, frames_), [&](const cv::Range& range) {
            /* Remove the seam */
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

    /* Distributes the loop over the threads */
    cv::parallel_for_(cv::Range(0, frames_), [&](const cv::Range& range) {
        /* Remove seams in the colored video */
        for (int f = range.start; f < range.end; ++f) {
            int j;
            cv::Mat new_row;

            int width = video_[f].cols;
            for (const auto &seam : seams_to_remove) {
                for (int i = 0; i < height_; ++i) {
                    j = 2*seam[i];
                    if (j + 2 != width) {
                        video_[f].row(2 * i).colRange(j + 2, width).copyTo(new_row);
                        new_row.copyTo(video_[f].row(2 * i).colRange(j, width - 2));
                        video_[f].row(2 * i + 1).colRange(j + 2, width).copyTo(new_row);
                        new_row.copyTo(video_[f].row(2 * i + 1).colRange(j, width - 2));
                    }
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

        /* Distributes the loop over the threads */
        cv::parallel_for_(cv::Range(0, frames_), [&](const cv::Range& range) {
            /* Remove the seam */
            for (int f = range.start; f < range.end; ++f) {
                int i;
                cv::Mat new_column;
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

    /* Distributes the loop over the threads */
    cv::parallel_for_(cv::Range(0, frames_), [&](const cv::Range& range) {
        /* Remove seams in the colored video. */
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