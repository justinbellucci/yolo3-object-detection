#ifndef YOLO3_H
#define YOLO3_H

#include "model.h"
#include "framequeue.h"
#include "yoloconfig.h"
#include "preprocessor.h"

#include <iostream>
#include <memory>
#include <vector>
#include <thread>
#include <algorithm>
#include <string>
#include <fstream>
#include <queue>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

class Yolo3
{
public:
    // constructor/destructor
    Yolo3() = delete;
    Yolo3(struct YoloConfig::FrameProcessingData &data);
    ~Yolo3();

    // public methods
    void run(cv::String &model_path, cv::String &config_path, cv::String &classNames_path);

private:
    std::unique_ptr<Model> _model; // yolo3 dnn model
    std::unique_ptr<cv::VideoCapture> _capture; // video capture object
    std::unique_ptr<FrameQueue<cv::Mat>> _frames; // FrameQueue object
    std::unique_ptr<FrameQueue<cv::Mat>> _processedFrames; 
    std::unique_ptr<FrameQueue<std::vector<cv::Mat>>> _predictions;
    
    std::vector<std::thread> threads; // create a thread vector
    std::vector<std::string> _classNames; 

    YoloConfig::FrameProcessingData _frameProcData;

    // private methods
    void startCaptureFramesThread();
    // void startProcessFramesThread();

    void captureFrames();
    void processFrames();

    void loadClassNames(std::string &path);
};

#endif