#include "yolo3.h"
#include "yoloconfig.h"

#include <iostream>

const char* keys =
"{help h usage ?    |      | Usage examples: \n\t\t./yolo3_detector --video=/my_video.mp4 --config=/yolov3.cfg --weights=/yolov3.weights --names=/coco.names}"
"{video v           |<none>| Input video   }"
"{config c          |<none>| Path to yolov3.cfg file.   }"
"{weights w         |<none>| Path to yolov3.weights file.   }"
"{names n           |<none>| Path to coco.names file.   }"
;

cv::String weights_path;
cv::String config_path;
cv::String classNames_path;

std::string outputFile;
cv::String mediaPath;
bool isVideo = false;

int main(int argc, char** argv)
{
    cv::CommandLineParser parser(argc, argv, keys);
    config_path = parser.get<cv::String>("config");
    weights_path = parser.get<cv::String>("weights");
    classNames_path = parser.get<cv::String>("names");

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

    YoloConfig::FrameProcessingData frameProcData;
    frameProcData.confThreshold = 0.5;
    frameProcData.nmsThreshold = 0.4;
    frameProcData.inpHeight = 416;
    frameProcData.inpWidth = 416;

    Yolo3 yolo3(frameProcData);
    yolo3.run(weights_path, config_path, classNames_path, mediaPath, isVideo);

    return 0;
}

