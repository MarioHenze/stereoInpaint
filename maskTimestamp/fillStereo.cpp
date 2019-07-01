#include <algorithm>
#include <filesystem>
#include <iostream>
#include <vector>

#include <opencv2/calib3d.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

void show_horizontal(std::vector<std::reference_wrapper<cv::Mat>> const images,
                     std::string const window_name = "images",
                     int const timeout = 0)
{
    if (!images.size())
        return;

    auto width{0};
    auto height{0};
    auto type{images.front().get().type()};

    for (auto const & image: images)
    {
        assert(image.get().dims == 2);
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
    cv::waitKey(timeout);
}

int main(int argc, char *argv[])
{
    const std::string options{
        "{help h usage ? |  | Print this message.}"
        "{gui | | Show input and resulting mask as window.}"
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

    show_horizontal({left, right, left_mask, right_mask});

    {// Find feature points in image for correspondance
        auto feature_detector = cv::AKAZE::create();
        cv::Mat inverted_mask_left;
        cv::Mat inverted_mask_right;
        cv::bitwise_not(left_mask, inverted_mask_left);
        cv::bitwise_not(right_mask, inverted_mask_right);

        std::vector<cv::KeyPoint> left_keypoints;
        std::vector<cv::KeyPoint> right_keypoints;
        cv::Mat left_descriptors;
        cv::Mat right_descriptors;

        feature_detector->detectAndCompute(left,
                                           inverted_mask_left,
                                           left_keypoints,
                                           left_descriptors);
        feature_detector->detectAndCompute(right,
                                           inverted_mask_right,
                                           right_keypoints,
                                           right_descriptors);
        if (left_descriptors.empty())
            cv::error(0,
                      "left descriptor empty",
                      __FUNCTION__,
                      __FILE__,
                      __LINE__);
        if (right_descriptors.empty())
            cv::error(0,
                      "right descriptor empty",
                      __FUNCTION__,
                      __FILE__,
                      __LINE__);

        std::vector<std::vector<cv::DMatch>> matches;
        auto matcher = cv::BFMatcher{};
        matcher.knnMatch(left_descriptors, right_descriptors, matches, 2);

        matches.erase(
                    std::remove_if(
                        matches.begin(),
                        matches.end(),
                        [](std::vector<cv::DMatch> match){
                            assert(match.size() == 2);
                            return match.front().distance >
                                .7f * match.back().distance;
                    }),
                matches.end());

        std::vector<cv::Point2f> left_points;
        std::vector<cv::Point2f> right_points;

        for (auto const & match : matches)
        {
            assert(match.size() == 2);

            auto left_index = match.front().queryIdx;
            auto right_index = match.back().queryIdx;

            left_points.emplace_back(left_keypoints.at(left_index).pt);
            right_points.emplace_back(right_keypoints.at(right_index).pt);
        }

        cv::Mat homography = cv::findHomography(left_points,
                                                right_points,
                                                cv::RANSAC);

        cv::Mat left_rectified;
        cv::undistort(left, left_rectified, homography, {});

        cv::imshow("rect", left_rectified);
        cv::waitKey(0);
    }

    auto sm = cv::StereoBM::create();

    cv::Mat disparity_map;
    cv::Mat left_mono;
    cv::Mat right_mono;

    cv::cvtColor(left, left_mono, cv::COLOR_BGR2GRAY);
    cv::cvtColor(right, right_mono, cv::COLOR_BGR2GRAY);
    sm->compute(left_mono, right_mono, disparity_map);

    cv::imshow("Disparity", disparity_map);
    cv::waitKey();

    return 0;
}
