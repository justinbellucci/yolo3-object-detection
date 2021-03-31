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

int main(int argc, char** argv)
{
    
}

