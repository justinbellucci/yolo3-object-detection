#include "yolo3.h"

#include <memory>

// constructor
Yolo3::Yolo3(YoloConfig::FrameProcessingData &data){
    _model = nullptr;
    _capture = std::make_unique<cv::VideoCapture>();
    _frames = std::make_unique<FrameQueue<cv::Mat>>();
    _processedFrames = std::make_unique<FrameQueue<cv::Mat>>();
    _predictions = std::make_unique<FrameQueue<std::vector<cv::Mat>>>();
    _frameProcData = std::move(data);
}

// destructor
// set up thread barrier before object is destroyed
Yolo3::~Yolo3(){
    std::for_each(threads.begin(), threads.end(), [](std::thread &t) {
        t.join();
    });
}

// --- class methods ---
void Yolo3::run(cv::String &model_path, cv::String &config_path, cv::String &classNames_path)
{
    // initialize model
    Model model = Model::initialize(cv::String(model_path), cv::String(config_path));
    _model = std::make_unique<Model>(std::move(model));

    // load class names
    loadClassNames(classNames_path);

    // TODO: Add functionality for other video types
    // open VideoCapture object with webcam (0)
    _capture->open(0);
    if(!_capture->isOpened())
    {
        std::cout << "Error opening video!" << std::endl;
    }

    // capture and process frames
    startCaptureFramesThread();
    processFrames();
}

// start a new thread for frame capturing
void Yolo3::startCaptureFramesThread()
{
    threads.emplace_back(std::thread(&Yolo3::captureFrames, this));
}


void Yolo3::captureFrames()
{
    cv::Mat frame;
    while(true)
    {
        *_capture >> frame;

        if(!frame.empty())
        {
            _frames->push(frame.clone());
        } 
        else 
        {
            break;
        }
    }
}

void Yolo3::processFrames()
{
    // process frames
    while(true)
    {
        cv::Mat frame;
        // get the next frame 
        if(!_frames->isEmpty())
        {
            frame = _frames->get();
        }

        // // process the frame
        if(!frame.empty())
        {
            _model->processFrames(frame, _frameProcData);
            // add processed frames to queue
            _processedFrames->push(frame);
            _model->forward(); // forward pass through the network
        }
        // // get the predictions of the model from the futures 
        
    }
}

void Yolo3::loadClassNames(std::string &path) 
{
    // create filestream object
    std::ifstream ifsObj;
    ifsObj.open(path);
    if(!ifsObj.is_open())
    {
        CV_Error(cv::Error::StsError, "COCO.names file at " + path + " not found.");
    }
    std::string line;
    while(std::getline(ifsObj, line))
    {
        _classNames.emplace_back(line);
    }
    std::cout << "Class names file successfully loaded." << std::endl;
    ifsObj.close();
}
