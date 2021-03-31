#include "yolo3.h"
#include "yoloconfig.h"

#include <iostream>

cv::String weights_path = "/Users/justinbellucci/GitHub/yolo3-object-detection/yolov3.weights";
cv::String config_path = "/Users/justinbellucci/GitHub/yolo3-object-detection/yolov3.cfg";
cv::String classNames_path = "/Users/justinbellucci/GitHub/yolo3-object-detection/coco.names";

const char* keys =
"{help h usage ? | | Usage examples: \n\t\t./yolo3_detector --video=run_sm.mp4}"
"{video v       |<none>| input video   }"
;

std::string outputFile;
cv::String mediaPath;
bool isVideo = false;

int main(int argc, char** argv)
{
    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("Use this script to run object detection using YOLO3 in OpenCV.");
    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }

    if(parser.has("video"))
    {
        isVideo = true;
        mediaPath = parser.get<cv::String>("video");
    }
    else
        isVideo = false;

}

