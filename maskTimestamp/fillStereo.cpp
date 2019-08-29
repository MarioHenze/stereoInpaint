#include <algorithm>
#include <filesystem>
#include <future>
#include <iostream>
#include <vector>

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
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

    if (parser.has("slice")) {
        CostVolume cost_volume(cv::Rect(0,0,left.cols, left.rows), 10);
        cost_volume.calculate(left, right, left_mask, right_mask);

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

        for (size_t i = 0; i < cost_volume.slice_count(); ++i) {
            cv::Mat const slice = cost_volume.slice(i);
            cv::imwrite(folder_path.string() + std::to_string(i) + ".PNG",
                        slice);
        }
    }

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

    show_horizontal({disparity_left, disparity_right});

    return 0;
}
