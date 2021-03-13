#ifndef YOLO3_H
#define YOLO3_H

#include <iostream>

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

private:
    
};

#endif