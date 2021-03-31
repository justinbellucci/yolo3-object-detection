#include "yolo3.h"

#include <memory>

// constructor
Yolo3::Yolo3(YoloConfig::FrameProcessingData &data){
    _capture = std::make_unique<cv::VideoCapture>();
    _video = std::make_unique<cv::VideoWriter>();
    _frame = std::make_unique<cv::Mat>();
    _blob = std::make_unique<cv::Mat>();
    _frameProcData = std::move(data);
}

// destructor
// set up thread barrier before object is destroyed
Yolo3::~Yolo3() = default;

// --- class methods ---
void Yolo3::run(cv::String &weights_path, cv::String &config_path, cv::String &classNames_path, cv::String &inputMedia)
{
    
}

// load the class names from the coco.names file
void Yolo3::loadClassNames(const std::string &namesPath)
{
    // create a filestream object
    std::ifstream ifsObj(namesPath.c_str());
    std::string line;
    while(std::getline(ifsObj, line))
    {
        _classNames.push_back(line);
    }
    std::cout << "Class names file successfully loaded." << std::endl;
}
