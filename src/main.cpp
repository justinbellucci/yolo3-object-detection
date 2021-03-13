#include "yolo3.h"

#include <iostream>

cv::String model_path = "/Users/justinbellucci/GitHub/yolo3-object-detection/yolov3.weights";
cv::String config_path = "/Users/justinbellucci/GitHub/yolo3-object-detection/yolov3.cfg";

int main()
{
    // do some testing here
    Yolo3 yolo3;
    yolo3.run(model_path, config_path);
    //yolo3.testCamera();
}