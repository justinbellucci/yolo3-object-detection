#ifndef YOLO3_H
#define YOLO3_H

#include "yoloconfig.h"
#include "model.h"

#include <iostream>
#include <memory>
#include <vector>
#include <algorithm>
#include <string>
#include <fstream>
#include <sstream>
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
    void run(cv::String &weightsPath, cv::String &configPath, std::string &classNamesPath, cv::String &mediaPath, bool &isVideo);

private:
    std::unique_ptr<cv::VideoCapture> _capturer; // video capture object
    std::unique_ptr<cv::VideoWriter> _video; // 
    std::unique_ptr<Model> _model;

    YoloConfig::FrameProcessingData _frameProcData;
    std::vector<std::string> _classNames;
    std::string _outputFile;

    cv::Mat _frame;
    cv::Mat _blob;
    // private methods
    void loadClassNames(const cv::String &classNamesPath);
    std::vector<cv::String> getOutputsNames(cv::dnn::Net &net);
    void postprocess(cv::Mat &frame, const std::vector<cv::Mat> &outs);
    void drawPredictions(int classId, float conf, int left, int top, int right, int bottom, cv::Mat &frame);
};

#endif