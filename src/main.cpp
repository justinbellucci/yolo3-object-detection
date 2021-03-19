#include "yolo3.h"
#include "yoloconfig.h"

#include <iostream>

cv::String model_path = "/Users/justinbellucci/GitHub/yolo3-object-detection/yolov3.weights";
cv::String config_path = "/Users/justinbellucci/GitHub/yolo3-object-detection/yolov3.cfg";
cv::String classNames_path = "/Users/justinbellucci/GitHub/yolo3-object-detection/coco.names";

int main()
{
    
    // instantiate struct object to hold frame processing data
    YoloConfig::FrameProcessingData data;
    data.inpWidth = 416;
    data.inpHeight = 416;

    // run main process
    Yolo3 yolo3(data);
    yolo3.run(model_path, config_path, classNames_path);
}