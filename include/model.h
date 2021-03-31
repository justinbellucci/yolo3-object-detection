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

    static Model initialize(const cv::String &model, const cv::String &config, float confThreshold = 0.5, float nmsThreshold = 0.4,
                            cv::dnn::Backend backend = cv::dnn::DNN_BACKEND_DEFAULT, cv::dnn::Target target = cv::dnn::DNN_TARGET_CPU);

    void processFrames(cv::Mat &frame, struct YoloConfig::FrameProcessingData &data);
    std::vector<cv::Mat> forward();

    void postProcessFrames(cv::Mat &frame, const std::vector<cv::Mat> &outs, std::vector<std::string> &names);
    void drawPreds(int classId, float conf, int left, int top, int right, int bottom, cv::Mat &frame, std::vector<std::string> &names);

    std::vector<cv::String> getOutputNames();
    
private:
    // constructor - class object protected from creation outside scope
    Model(cv::dnn::Net &net, float confThreshold, float nmsThreshold);

    std::unique_ptr<cv::dnn::Net> _net;

    float _confThreshold; // Confidence threshold
    float _nmsThreshold; // Non-maximum suppression threshold
};

#endif