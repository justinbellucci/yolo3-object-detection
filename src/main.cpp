#include "yolo3.h"
#include "yoloconfig.h"

#include <filesystem>
#include <iostream>

const char* keys =
"{help h usage ?    |      | Usage examples: \n\t\t./yolo3_detector --video=/my_video.mp4 }"
"{video v           |<none>| Input video   }"
;

cv::String weights_path;
cv::String config_path;
cv::String classNames_path;
std::string outputFile;
cv::String mediaPath;

cv::String cwd;
bool isVideo = false;

int main(int argc, char** argv)
{
    cv::CommandLineParser parser(argc, argv, keys);
    // get main directory for project - up one from build
    std::filesystem::path p = std::filesystem::current_path();
    cwd = p.parent_path();

    config_path = cwd + "/yolov3.cfg";
    weights_path = cwd + "/yolov3.weights";
    classNames_path = cwd + "/coco.names";

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

