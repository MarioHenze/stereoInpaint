#include <algorithm>
#include <filesystem>
#include <future>
#include <iostream>
#include <mutex>
#include <vector>

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/matx.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/stereo.hpp>

#include "costvolume.h"

std::mutex m_cout_mutex;

void show_horizontal(std::vector<std::reference_wrapper<cv::Mat>> const & images,
                     std::string const & window_name = "images",
                     int const timeout = 0)
{
    if (images.empty())
        return;

    auto width{0};
    auto height{0};
    auto type{images.front().get().type()};

    for (auto const & image: images)
    {
        //assert(image.get().dims == 2);
        assert(image.get().type() == type);

        width += image.get().cols;
        height = std::max(height, image.get().rows);
    }

    cv::Mat composite(cv::Size(width, height), type);

    int anchor{0};
    for (auto const & image: images)
    {
        cv::Rect const roi(anchor,
                           0,
                           image.get().cols,
                           image.get().rows);
        image.get().copyTo(composite(roi));
        anchor += image.get().cols;
    }

    // limit to display width
    cv::resize(composite,
               composite,
               {},
               1. / images.size(),
               1. / images.size());
    cv::namedWindow(window_name, cv::WINDOW_NORMAL);
    cv::imshow(window_name, composite);
    cv::resizeWindow(window_name, width, height);
    cv::waitKey(timeout);
}

void compute_disparity(const cv::Mat & from,
                       const cv::Mat & to,
                       cv::Mat & disparity)
{
    auto sm = cv::stereo::QuasiDenseStereo::create({from.cols, from.rows});

    sm->process(from, to);
    disparity = sm->getDisparity();
}

/**
 * @brief morph_by_disparity morphs valid pixels into masked regions
 * @param source partial occluded data from one view
 * @param source_mask the occlusion mask for the source view
 * @param target partial occluded data from another view
 * @param target_mask occlusion mask for the target view
 * @param disp_map disparity map which translates pixel positions from the
 * source view into the target view
 * @return a pair with first containing the target image with additional pixels
 * of the source image, and second containing the modified mask
 */
std::pair<cv::Mat, cv::Mat> morph_by_disparity(cv::Mat const &source,
                                               cv::Mat const &source_mask,
                                               cv::Mat const &target,
                                               cv::Mat const &target_mask,
                                               cv::Mat const &disp_map)
{
    assert(!source.empty());
    assert(!source_mask.empty());
    assert(!target.empty());
    assert(!target_mask.empty());
    assert(!disp_map.empty());

    assert(source.size() == source_mask.size());
    assert(source_mask.size() == target.size());
    assert(target.size() == target_mask.size());
    assert(target_mask.size() == disp_map.size());

    assert(disp_map.type() == CV_32S);

    cv::Mat target_remapped = target.clone();
    cv::Mat target_mask_remapped = target_mask.clone();

    #pragma omp parallel for
    for (int scanline = 0; scanline < source.cols; ++scanline) {
        for (int pixel = 0; pixel < source.rows; ++pixel) {
            // Our source is invalid at the location
            if (source_mask.at<uint8_t>(pixel, scanline) > 0)
                continue;

            auto const disp_at_location = disp_map.at<int32_t>(pixel, scanline);
            auto const mapped_pixel = pixel + disp_at_location;

            // The target location lies outside of the frame
            if (mapped_pixel >= target_mask.rows || mapped_pixel < 0)
                continue;

            // The target already contains valid data at the remapped point
            if (target_mask.at<uint8_t>(mapped_pixel, scanline) == 0)
                continue;

// As our preconditions hold true, e.g. we have a valid source pixel
// and a masked target, we can copy our pixel
#pragma omp critical
            {
                target_remapped.at<cv::Vec3b>(mapped_pixel, scanline)
                    = source.at<cv::Vec3b>(pixel, scanline);
                // Zap mask bits out of our initial target mask
                target_mask_remapped.at<uint8_t>(mapped_pixel, scanline) = 0;
            }
        }
    }

    return std::make_pair(target_remapped, target_mask_remapped);
}

int main(int argc, char *argv[])
{
    const std::string options{
        "{help h usage ? |  | Print this message.}"
        "{gui | | Show input and resulting mask as window.}"
        "{slice | | Save cost volume slices}"
        "{disparity | | Save disparity maps}"
        "{force | | Force writing over existing files}"
        "{@left | | left input image}"
        "{@right | | rigth input image}"
        "{@left_mask | | left mask image}"
        "{@right_mask | | rigth mask image}"};
    cv::CommandLineParser parser{argc, argv, options};

    if (parser.has("h")) {
        parser.printMessage();
        return 0;
    }

    if (!parser.has("@left") || !parser.has("@right")
        || !parser.has("@left_mask") || !parser.has("@right_mask")) {
        std::cerr << "Not all images were specified!";
        return -1;
    }

    if (!std::filesystem::exists(parser.get<std::string>("@left"))
        || !std::filesystem::exists(parser.get<std::string>("@right"))
        || !std::filesystem::exists(parser.get<std::string>("@left_mask"))
        || !std::filesystem::exists(parser.get<std::string>("@right_mask"))) {
        std::cerr << "Not all images exist";
        return -1;
    }

    cv::Mat left = cv::imread(parser.get<std::string>("@left"));
    cv::Mat right = cv::imread(parser.get<std::string>("@right"));
    cv::Mat left_mask = cv::imread(parser.get<std::string>("@left_mask"));
    cv::Mat right_mask = cv::imread(parser.get<std::string>("@right_mask"));

    if (left.empty() || right.empty() || left_mask.empty()
        || right_mask.empty()) {
        std::cerr << "Could not read all images and masks!";
        return -1;
    }

    // As the mask are stored as png, we need to convert them to grayscale
    if (left_mask.type() == CV_8UC3) {
        cv::cvtColor(left_mask, left_mask, cv::COLOR_BGR2GRAY);
    }

    if (right_mask.type() == CV_8UC3) {
        cv::cvtColor(right_mask, right_mask, cv::COLOR_BGR2GRAY);
    }

    // After rectification the timestamp location are not correlated anymore and
    // therefore we have different cost volumes, depending on the mapping
    // direction
    CostVolume cost_to_right(left.cols, left.rows, 80);
    CostVolume cost_to_left(right.cols, right.rows, 80);

    {
    std::cout << "Calculating to right cost volume" << std::endl;
    //cost_to_right.calculate(left, right, left_mask, right_mask, 9);
    auto cost_to_right_future = std::async(std::launch::async,
                                           &CostVolume::calculate,
                                           &cost_to_right,
                                           std::ref(left),
                                           std::ref(right),
                                           std::ref(left_mask),
                                           std::ref(right_mask),
                                           9);
    std::cout << "Calculating to left cost volume" << std::endl;
    //cost_to_left.calculate(right, left, right_mask, left_mask, 9);
    auto cost_to_left_future = std::async(std::launch::async,
                                          &CostVolume::calculate,
                                          &cost_to_left,
                                          std::ref(right),
                                          std::ref(left),
                                          std::ref(right_mask),
                                          std::ref(left_mask),
                                          9);
    }

    if (parser.has("slice")) {
        std::cout << "Slicing cost volume\n";
        // seperate slices for every pair into own folder to prevent mismatch
        auto const path = std::filesystem::path(
                              parser.get<std::string>("@left"))
                              .remove_filename()
                              .string();
        if (path.empty())
            const_cast<std::string &>(path) = "./";

        auto const name = std::filesystem::path(
                              parser.get<std::string>("@left"))
                              .filename()
                              .replace_extension()
                              .string()
                          + std::filesystem::path(
                                parser.get<std::string>("@right"))
                                .filename()
                                .replace_extension()
                                .string();
        assert(!name.empty());

        auto const folder_path = std::filesystem::path(path + name + "/");
        std::filesystem::create_directory(folder_path);

        for (int i = 0; i < cost_to_right.slice_count(); ++i) {
            cv::Mat const slice = cost_to_right.slice(i);
            cv::imwrite(folder_path.string() + "to_r" + std::to_string(i)
                            + ".PNG",
                        slice);
        }

        for (int i = 0; i < cost_to_left.slice_count(); ++i) {
            cv::Mat const slice = cost_to_left.slice(i);
            cv::imwrite(folder_path.string() + "to_l" + std::to_string(i)
                            + ".PNG",
                        slice);
        }
    }

    cv::Mat disparity_to_right;
    cv::Mat disparity_to_left;
    { // Parallel computation of the disparity maps
        std::cout << "Calculate disparity map" << std::endl;
        auto to_right_future = std::async(std::launch::async,
                                          &CostVolume::calculate_disparity_map,
                                          &cost_to_right);
        auto to_left_future = std::async(std::launch::async,
                                         &CostVolume::calculate_disparity_map,
                                         &cost_to_left);
        disparity_to_right = to_right_future.get().clone();
        disparity_to_left = to_left_future.get().clone();
    }

    if (parser.has("disparity")) {
        cv::imwrite("disparity_to_right_"
                        + std::filesystem::path(
                              parser.get<std::string>("@left"))
                              .filename()
                              .string() + ".PNG",
                    disparity_to_right);
        cv::imwrite("disparity_to_left_"
                        + std::filesystem::path(
                              parser.get<std::string>("@right"))
                              .filename()
                              .string() + ".PNG",
                    disparity_to_left);
    }

    cv::Mat remapped_left;
    cv::Mat remapped_left_mask;
    cv::Mat remapped_right;
    cv::Mat remapped_right_mask;
    { // Parallel remapping according to the disparity maps
        std::cout << "Remapping image" << std::endl;
        //        auto to_right_future = std::async(std::launch::async,
        //                                          &morph_by_disparity,
        //                                          std::ref(left),
        //                                          std::ref(left_mask),
        //                                          std::ref(right),
        //                                          std::ref(right_mask),
        //                                          std::ref(disparity_to_right));
        //        auto to_left_future = std::async(std::launch::async,
        //                                         &morph_by_disparity,
        //                                         std::ref(right),
        //                                         std::ref(right_mask),
        //                                         std::ref(left),
        //                                         std::ref(left_mask),
        //                                         std::ref(disparity_to_left));

        //        auto const to_right_pair = to_right_future.get();
        //        auto const to_left_pair = to_left_future.get();

        auto const to_right_pair = morph_by_disparity(left,
                                                      left_mask,
                                                      right,
                                                      right_mask,
                                                      disparity_to_right);
        auto const to_left_pair = morph_by_disparity(right,
                                                     right_mask,
                                                     left,
                                                     left_mask,
                                                     disparity_to_left);

        remapped_right = to_right_pair.first.clone();
        remapped_right_mask = to_right_pair.second.clone();
        remapped_left = to_left_pair.first.clone();
        remapped_left_mask = to_left_pair.second.clone();
    }

    cv::threshold(remapped_left_mask,
                  remapped_left_mask,
                  0.5,
                  255,
                  cv::THRESH_BINARY);
    cv::threshold(remapped_right_mask,
                  remapped_right_mask,
                  0.5,
                  255,
                  cv::THRESH_BINARY);

    auto const prefix = "remapped_";
    cv::imwrite(prefix
                    + std::filesystem::path(parser.get<std::string>("@left"))
                          .filename()
                          .string() + ".PNG",
                remapped_left);
    cv::imwrite(prefix
                    + std::filesystem::path(
                          parser.get<std::string>("@left_mask"))
                          .filename()
                          .string() + ".PNG",
                remapped_left_mask);
    cv::imwrite(prefix
                    + std::filesystem::path(parser.get<std::string>("@right"))
                          .filename()
                          .string() + ".PNG",
                remapped_right);
    cv::imwrite(prefix
                    + std::filesystem::path(
                          parser.get<std::string>("@right_mask"))
                          .filename()
                          .string() + ".PNG",
                remapped_right_mask);

    //auto const disparity_map_to_right = cost_to_right.calculate_disparity_map();


//    auto const remapped_right = morph_by_disparity(left,
//                                                   left_mask,
//                                                   right,
//                                                   right_mask,
//                                                   disparity_map_to_right);
//    cv::imwrite("remap.jpg", remapped_right.first);

    /*
    cv::Mat disparity_left;
    cv::Mat disparity_right;

    auto lr_disp_future = std::async(std::launch::async,
                                     compute_disparity,
                                     left,
                                     right,
                                     std::ref(disparity_left));
    auto rl_disp_future = std::async(std::launch::async,
                                     compute_disparity,
                                     right,
                                     left,
                                     std::ref(disparity_right));

    lr_disp_future.get();
    rl_disp_future.get();

    show_horizontal({disparity_left, disparity_right});*/

    return 0;
}
