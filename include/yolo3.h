#ifndef YOLO3_H
#define YOLO3_H

#include "model.h"

#include <iostream>
#include <memory>
#include <vector>
#include <thread>

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
    int testCamera();
    
    void run(cv::String model_path, cv::String config_path);

private:
    std::unique_ptr<Model> _model;
};

#endif