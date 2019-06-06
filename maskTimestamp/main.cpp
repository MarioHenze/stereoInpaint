#include <algorithm>
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

int main(int argc, char *argv[])
{
    constexpr std::pair<unsigned int, unsigned int> origin {1715, 1695};
    constexpr std::pair<
            unsigned int,
            unsigned int>
            extend {2495 - origin.first,1785 - origin.second};

    //constexpr double area_upper_threshold{7000};
    //constexpr double area_lower_threshold{100};

    const std::string options{
        "{help h usage ? |  | Print this message.}"
        "{gui | | Show input and resulting mask as window.}"
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

    // Cut out usual timestamp region
    cv::Mat in_image_cropped = in_image({origin.first,
                                         origin.second,
                                         extend.first,
                                         extend.second});

    if(parser.has("gui"))
        showImg(in_image_cropped);

    // Convert to grayscale for gradient computation
    cv::Mat in_grayscale;
    cv::cvtColor(in_image_cropped, in_grayscale, cv::COLOR_RGB2GRAY);

    if(parser.has("gui"))
        showImg(in_grayscale);

    // Derive morphological gradient image
    cv::Mat in_grad;
    cv::Mat morphKernel = cv::getStructuringElement(
                cv::MorphShapes::MORPH_ELLIPSE,
                cv::Size(3,3));
    cv::morphologyEx(
                in_grayscale,
                in_grad,
                cv::MorphTypes::MORPH_GRADIENT,
                morphKernel);

    if(parser.has("gui"))
        showImg(in_grad);

    // Convert to binary threshold image for morphological filtering
    cv::Mat in_bw;
    cv::threshold(in_grad,
                  in_bw,
                  0.,
                  255.,
                  cv::THRESH_BINARY | cv::THRESH_OTSU);

    if(parser.has("gui"))
        showImg(in_bw);

    // Fill holes and horizontal and vertical "gaps"
    cv::Mat in_connected;
    morphKernel = getStructuringElement(cv::MorphShapes::MORPH_RECT,
                                        cv::Size(3 , 3));
    morphologyEx(in_bw,
                 in_connected,
                 cv::MorphTypes::MORPH_CLOSE,
                 morphKernel);

    morphKernel = getStructuringElement(cv::MorphShapes::MORPH_RECT,
                                        cv::Size(9 , 1));
    morphologyEx(in_connected,
                 in_connected,
                 cv::MorphTypes::MORPH_CLOSE,
                 morphKernel);

    morphKernel = getStructuringElement(cv::MorphShapes::MORPH_RECT,
                                        cv::Size(1 , 9));
    morphologyEx(in_connected,
                 in_connected,
                 cv::MorphTypes::MORPH_CLOSE,
                 morphKernel);

    if(parser.has("gui"))
        showImg(in_connected);

//    cv::Mat in_inv;
//    cv::bitwise_not(in_connected, in_inv);
//    std::vector<std::vector<cv::Point>> contours;
//    cv::findContours(in_inv,
//                     contours,
//                     cv::RetrievalModes::RETR_LIST,
//                     cv::ContourApproximationModes::CHAIN_APPROX_SIMPLE);

//    if(parser.has("gui"))
//        showImg(in_inv);

//    // Filter out all invalid contours and hopefully the background
//    auto iter = std::remove_if(contours.begin(),
//                   contours.end(),
//                   [] (decltype(contours)::value_type contour)
//                   {
//                       return area_lower_threshold > cv::contourArea(contour) ||
//                        area_upper_threshold < cv::contourArea(contour);
//                   });
//    contours.erase(iter, contours.cend());

//    cv::cvtColor(in_inv, in_inv, cv::ColorConversionCodes::COLOR_GRAY2RGB);
//    cv::drawContours(in_inv, contours, -1, {0,255,0}, cv::FILLED);

//    if(parser.has("gui"))
//        showImg(in_inv);

    // Write out mask image
    cv::Mat mask{in_image.size(), CV_8UC1};
    in_connected.copyTo(mask({origin.first,
                              origin.second,
                              extend.first,
                              extend.second}));

    if(parser.has("gui"))
        showImg(mask);

    std::string filename = parser.get<std::string>("@input");

    std::istringstream iss(filename);
    std::vector<std::string> results(std::istream_iterator<std::string>{iss},
                                     std::istream_iterator<std::string>());

    cv::imwrite(parser.get<std::string>("@output"), mask);

    return 0;
}
