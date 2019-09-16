#include <algorithm>
#include <filesystem>
#include <future>
#include <iostream>
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

    for (int scanline = 0; scanline < source.cols; ++scanline) {
        for (int pixel = 0; pixel < source.rows; ++pixel) {
            // Our source is invalid at the location
            if (source_mask.at<uint8_t>(pixel, scanline) > 0)
                continue;

            auto const disp_at_location = disp_map.at<int32_t>(pixel, scanline);
            auto const mapped_pixel = pixel + disp_at_location;
            // Our target already contains valid data at the remapped point
            if (target_mask.at<uint8_t>(mapped_pixel, scanline) == 0)
                continue;

            // As our preconditions hold true, e.g. we have a valid source pixel
            // and a masked target, we can copy our pixel
            target_remapped.at<cv::Vec3b>(mapped_pixel, scanline)
                = source.at<cv::Vec3b>(pixel, scanline);
            // Zap mask bits out of our initial target mask
            target_mask_remapped.at<uint8_t>(mapped_pixel, scanline) = 0;
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
        "{force | | Force writing over existing files}"
        "{@left | | left input image}"
        "{@right | | rigth input image}"
        "{@left_mask | | left mask image}"
        "{@right_mask | | rigth mask image}"
    };
    cv::CommandLineParser parser{argc, argv, options};

    if (parser.has("h"))
    {
        parser.printMessage();
        return 0;
    }

    if (
            !parser.has("@left") || !parser.has("@right") ||
            !parser.has("@left_mask") || !parser.has("@right_mask")
       )
    {
        std::cerr << "Not all images were specified!";
        return -1;
    }

    if (
            !std::filesystem::exists(parser.get<std::string>("@left")) ||
            !std::filesystem::exists(parser.get<std::string>("@right")) ||
            !std::filesystem::exists(parser.get<std::string>("@left_mask")) ||
            !std::filesystem::exists(parser.get<std::string>("@right_mask"))
       )
    {
        std::cerr << "Not all images exist";
        return -1;
    }

    cv::Mat left = cv::imread(parser.get<std::string>("@left"));
    cv::Mat right = cv::imread(parser.get<std::string>("@right"));
    cv::Mat left_mask = cv::imread(parser.get<std::string>("@left_mask"));
    cv::Mat right_mask = cv::imread(parser.get<std::string>("@right_mask"));

    if (
            left.empty() ||
            right.empty() ||
            left_mask.empty() ||
            right_mask.empty()
       )
    {
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


    CostVolume cost_volume(left.cols, left.rows, 10);
    cost_volume.calculate(left, right, left_mask, right_mask, 9);

    if (parser.has("slice")) {
        // seperate slices for every pair into own folder to prevent mismatch
        auto const path
            = std::filesystem::path(parser.get<std::string>("@left"))
                  .remove_filename().string();
        if (path.empty())
            const_cast<std::string &>(path) = "./";

        auto const name
            = std::filesystem::path(parser.get<std::string>("@left"))
                  .filename().replace_extension().string()
            + std::filesystem::path(parser.get<std::string>("@right"))
                  .filename().replace_extension().string();
        assert(!name.empty());

        auto const folder_path = std::filesystem::path(path + name + "/");
        std::filesystem::create_directory(folder_path);

        for (int i = 0; i < cost_volume.slice_count(); ++i) {
            cv::Mat const slice
                = cost_volume.slice(i);
            cv::imwrite(folder_path.string() + std::to_string(i) + ".PNG",
                        slice);
        }
    }

    auto const disparity_map = cost_volume.calculate_disparity_map();
    //cv::imshow("disp_map", disparity_map);
    //cv::waitKey();
    //cv::imwrite("disparity_map.png", disparity_map);

    auto const remapped_right = morph_by_disparity(left,
                                                   left_mask,
                                                   right,
                                                   right_mask,
                                                   disparity_map);
    cv::imwrite("remap.jpg", remapped_right.first);

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
