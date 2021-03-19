#ifndef PREPROCESSOR_H
#define PREPROCESSOR_H

#include "yoloconfig.h"

#include "opencv2/opencv.hpp"

class Preprocessor 
{
public:
    // constructor / destructor
    Preprocessor() = default;
    ~Preprocessor() = default;

    // Preprocessor(const Preprocessor& source) = delete;
    Preprocessor(Preprocessor&& source) noexcept;

    // Preprocessor& operator=(const Preprocessor& source) = delete;
    // Preprocessor& operator=(Preprocessor&& source) noexcept = delete;

    void process(cv::Mat &frame, cv::dnn::Net &net, YoloConfig::FrameProcessingData &data);
};

#endif 
