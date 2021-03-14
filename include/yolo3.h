#ifndef YOLO3_H
#define YOLO3_H

#include "model.h"
#include "framequeue.h"

#include <iostream>
#include <memory>
#include <vector>
#include <thread>
#include <algorithm>

#include <queue>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
class Yolo3
{
public:
    // constructor/destructor
    Yolo3();
    ~Yolo3();

    // public methods
    void run(cv::String model_path, cv::String config_path);

private:
    std::unique_ptr<Model> _model; // yolo3 dnn model
    std::unique_ptr<cv::VideoCapture> _capture; // video capture object
    std::unique_ptr<FrameQueue<cv::Mat>> _frames; // FrameQueue object

    std::vector<std::thread> threads; // create a thread vector

    // private functions
    void startCaptureFramesThread();

    void CaptureFrames();
};

#endif