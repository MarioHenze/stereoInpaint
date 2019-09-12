#include <algorithm>
#include <filesystem>
#include <iostream>
#include <iterator>
#include <map>
#include <string>

#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/utility.hpp>

void showImg(cv::Mat const & img)
{
    cv::imshow("Image", img);
    cv::waitKey();
}

cv::Mat threshold_hsv(const cv::Mat & in,
                      const cv::Scalar & color,
                      const cv::Scalar & variation)
{
    cv::Mat hsv_in;
    cv::Mat ret;
    cv::cvtColor(in, hsv_in, cv::COLOR_BGR2HSV);

    cv::inRange(hsv_in, color - variation, color + variation, ret);

    return ret;
}

cv::Mat expand_morphological(const cv::Mat & in,
                             const int repetitions = 1)
{
    cv::Mat morphKernel = getStructuringElement(cv::MorphShapes::MORPH_RECT,
                                                cv::Size(3 , 3));

    cv::Mat ret{in};

    for (int i = 0; i < repetitions; ++i) {
        morphologyEx(in,
                     ret,
                     cv::MorphTypes::MORPH_DILATE,
                     morphKernel);
    }

    return ret;
}

int main(int argc, char *argv[])
{
    constexpr std::pair<int, int> origin {1715, 1695};
    constexpr std::pair<int, int> extend{2495 - origin.first,
                                                  1785 - origin.second};

    const std::string options{
        "{help h usage ? |  | Print this message.}"
        "{gui | | Show input and resulting mask as window.}"
        "{force | | Force writing over existing files}"
        "{@input | | Image to search for timestamps}"
        "{@output | | Image to save the mask to}"
    };
    cv::CommandLineParser parser{argc, argv, options};

    if (parser.has("h"))
    {
        parser.printMessage();
        return 0;
    }

    if (!parser.has("@input") || !parser.has("@output"))
    {
        std::cout << "Input or output image was not specified!";
        return -1;
    }

    cv::Mat in_image = cv::imread(parser.get<std::string>("@input"));
    if (!in_image.data)
    {
        std::cout << "Image could not be loaded!";
        return -1;
    }

    if (!parser.has("force") &&
        std::filesystem::exists(parser.get<std::string>("@output")))
    {
        std::cout << "File with name "
                  << parser.get<std::string>("@output")
                  << " already exists. Force override to continue!";
        return -1;
    }

    // Cut out usual timestamp region
    cv::Mat in_image_cropped = in_image;/*({origin.first,
                                         origin.second,
                                         extend.first,
                                         extend.second});*/

    if(parser.has("gui"))
        showImg(in_image_cropped);


    cv::Mat mask = threshold_hsv(in_image_cropped,
                                 cv::Scalar{12, 245, 245},
                                 cv::Scalar{12, 10, 10});

    if(parser.has("gui"))
        showImg(mask);

    cv::Mat filled = expand_morphological(mask, 4);

    if(parser.has("gui"))
        showImg(filled);

    /*// Write out mask image
    cv::Mat full_mask{in_image.size(), CV_8UC1};
    mask.copyTo(full_mask({origin.first,
                           origin.second,
                           extend.first,
                           extend.second}));*/

    if(parser.has("gui"))
        showImg(filled);//full_mask);

    cv::imwrite(parser.get<std::string>("@output"), filled);//full_mask);

    return 0;
}
