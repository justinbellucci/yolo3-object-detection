#include "yolo3.h"
#include <memory>

// constructor
Yolo3::Yolo3(){
    _capture = std::make_unique<cv::VideoCapture>();
    _frames = std::make_unique<FrameQueue<cv::Mat>>();
}

// destructor
// set up thread barrier before object is destroyed
Yolo3::~Yolo3(){
    std::for_each(threads.begin(), threads.end(), [](std::thread &t) {
        t.join();
    });
}

// --- class methods ---
void Yolo3::run(cv::String model_path, cv::String config_path)
{
    Model model = Model::initialize(cv::String(model_path), cv::String(config_path));
    _model = std::make_unique<Model>(std::move(model));

    // TODO: Add functionality for other video types
    // open VideoCapture object with webcam (0)
    _capture->open(0);
    if(!_capture->isOpened())
    {
        std::cout << "Error opening video!" << std::endl;
    }
    // start frames capturing
    startCaptureFramesThread();
    CaptureFrames();
}

// start a new thread for frame capturing
void Yolo3::startCaptureFramesThread()
{
    threads.emplace_back(std::thread(&Yolo3::CaptureFrames, this));
}

void Yolo3::CaptureFrames()
{
    cv::Mat frame;
    while(true)
    {
        *_capture >> frame;

        if(!frame.empty())
        {
            _frames->pushFrame(frame.clone());
        } else 
        {
            break;
        }
    
    }
}