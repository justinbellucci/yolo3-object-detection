#ifndef MODEL_H
#define MODEL_H

#include "yoloconfig.h"
#include "preprocessor.h"

#include <memory>

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/core/async.hpp>

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
                            const cv::String &framework = "", cv::dnn::Backend backend = cv::dnn::DNN_BACKEND_OPENCV,
                            cv::dnn::Target target = cv::dnn::DNN_TARGET_CPU);

    void processFrames(cv::Mat &frame, struct YoloConfig::FrameProcessingData &data);
    void forward();

    std::vector<cv::String> getOutputNames(cv::dnn::Net &net);
    
private:
    // constructor - class object protected from creation outside scope
    Model(cv::dnn::Net &net, float confThreshold, float nmsThreshold, Preprocessor &preproc);

    std::unique_ptr<cv::dnn::Net> _net;
    std::unique_ptr<Preprocessor> _preProcessor;

    float _confThreshold; // Confidence threshold
    float _nmsThreshold; // Non-maximum suppression threshold
};

#endif