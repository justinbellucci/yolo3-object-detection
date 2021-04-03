#ifndef MODEL_H
#define MODEL_H

#include "yoloconfig.h"

#include <memory>
#include <iostream>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

class Model 
{
public:
    // constructor 
    Model() = delete;
    // destructor
    ~Model();
    // copy constructor
    Model(const Model &source) = delete;
    // copy assignment operator
    Model &operator=(const Model &source) = delete;
    // // move constructor
    Model(Model &&source) noexcept; 
    // // move assignment operator
    Model &operator=(Model &&source) noexcept;

    static Model initialize(const cv::String &configPath, const cv::String &weightsPath, float confThreshold = 0.5, float nmsThreshold = 0.4,
                            cv::dnn::Target backend = cv::dnn::DNN_TARGET_CPU);

    cv::dnn::Net processFrames(cv::Mat &frame, struct YoloConfig::FrameProcessingData &data);

    
private:
    // constructor - class object protected from creation outside scope
    Model(cv::dnn::Net &net, float confThreshold, float nmsThreshold);

    std::unique_ptr<cv::dnn::Net> _net;

    float _confThreshold; // Confidence threshold
    float _nmsThreshold; // Non-maximum suppression threshold
};

#endif