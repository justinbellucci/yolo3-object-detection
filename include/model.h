#ifndef MODEL_H
#define MODEL_H

#include <memory>

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>

class Model 
{
public:
    // constructor moved to private
    Model() = delete;
    // destructor
    ~Model();
    // copy constructor
    Model(const Model &source) = delete;
    // copy assignment operator
    Model &operator=(const Model &source) = delete;
    // move constructor
    Model(Model &&source);
    // move assignment operator
    Model &operator=(Model &&source);

    static Model initialize(const cv::String &model, const cv::String &config, float confThreshold = 0.5, float nmsThreshold = 0.4,
                            const cv::String &framework = "", cv::dnn::Backend backend = cv::dnn::DNN_BACKEND_DEFAULT,
                            cv::dnn::Target target = cv::dnn::DNN_TARGET_CPU);

private:
    // constructor
    Model(cv::dnn::Net &net, float confThreshold, float nmsThreshold);
    
    std::unique_ptr<cv::dnn::Net> _net;
    // _preprocessor;
    // _postprocessor;

    float _confThreshold; // Confidence threshold
    float _nmsThreshold; // Non-maximum suppression threshold
};

#endif