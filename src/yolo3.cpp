#include "yolo3.h"

Yolo3::Yolo3(){}

Yolo3::~Yolo3(){}

int Yolo3::testCamera()
{
    cv::VideoCapture capture(0);

    if(!capture.isOpened())
    {
        std::cout << "Error opening video stream!" << std::endl;
        return -1;
    }

    while(1)
    {
        cv::Mat frame;

        capture >> frame;
        flip(frame, frame, 1);

        if(frame.empty())
        {
            break;
        }
        cv::imshow("Test frame", frame);

        char c = (char)cv::waitKey(1); 
        if(c==27)
        {
            break;
        }
    }
    capture.release();
    return 0;
}