#include <iostream>

#include <opencv2/core/utility.hpp>

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
        std::cout << "Not all images were specified!";
        return -1;
    }

    return 0;
}
