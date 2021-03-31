#include "yolo3.h"

#include <memory>

// constructor
Yolo3::Yolo3(YoloConfig::FrameProcessingData &data){
    _capturer = std::make_unique<cv::VideoCapture>();
    _video = std::make_unique<cv::VideoWriter>();
    _frameProcData = std::move(data);
}

// destructor
// set up thread barrier before object is destroyed
Yolo3::~Yolo3() = default;

// --- class methods ---
void Yolo3::run(cv::String &weightsPath, cv::String &configPath, cv::String &classNamesPath, cv::String &mediaPath, bool &isVideo)
{
    // TODO: Move this somewhere else
    std::string outputFile = "yolo_out_cpp.avi";

    // load the class names from the coco.names file
    loadClassNames(classNamesPath);
    // load the network
    cv::dnn::Net net = cv::dnn::readNetFromDarknet(configPath, weightsPath); 
    _net = std::make_unique<cv::dnn::Net>(std::move(net));
    _net->setPreferableBackend(cv::dnn::DNN_TARGET_CPU);

    // open video file or webcam
    try
    {
        if(isVideo)
        {
            // Open the video file
            std::ifstream ifile(mediaPath);
            if (!ifile) throw("error");
            _capturer->open(mediaPath);
            mediaPath.replace(mediaPath.end()-4, mediaPath.end(), "_yolo_out_cpp.avi");
            outputFile = mediaPath;
        }
        // open the webcam
        else
            _capturer->open(0);
    }
    catch(...)
    {
        std::cout << "Could not open the video or webcam stream." << std::endl;
    }
    
    _video->open(outputFile, cv::VideoWriter::fourcc('M','J','P','G'), 28, cv::Size(_capturer->get(cv::CAP_PROP_FRAME_WIDTH), _capturer->get(cv::CAP_PROP_FRAME_HEIGHT)));
    // Create a new window
    static const std::string kWinName = "Deep learning object detection in OpenCV";
    cv::namedWindow(kWinName, cv::WINDOW_NORMAL);

    // main frame processing loop
    while(cv::waitKey(1) < 0)
    {
        // get the fram from the video
        *_capturer >> _frame;

        // Stop the program if reached end of video
        if (_frame.empty()) {
            std::cout << "Done processing !!!" << std::endl;
            std::cout << "Output file is stored as " << outputFile << std::endl;
            cv::waitKey(3000);
            break;
        }

        // Create a 4D blob from a frame.
        cv::dnn::blobFromImage(_frame, _blob, 1/255.0, cv::Size(_frameProcData.inpWidth, _frameProcData.inpHeight), cv::Scalar(0,0,0), true, false);
        
        //Sets the input to the network
        _net->setInput(_blob);

        // Runs the forward pass to get output of the output layers
        std::vector<cv::Mat> outs;
        _net->forward(outs, getOutputsNames(*_net));
        
        // Remove the bounding boxes with low confidence
        postprocess(_frame, outs);
        
        // Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
        std::vector<double> layersTimes;
        double freq = cv::getTickFrequency() / 1000;
        double t = net.getPerfProfile(layersTimes) / freq;
        std::string label = cv::format("Inference time for a frame : %.2f ms", t);
        cv::putText(_frame, label, cv::Point(0, 15), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255));
        
        // Write the frame with the detection boxes
        cv::Mat detectedFrame;
        _frame.convertTo(detectedFrame, CV_8U);
        _video->write(detectedFrame);
        cv::imshow(kWinName, _frame);

    }
    _capturer->release();
    _video->release();
}

// load the class names from the coco.names file
void Yolo3::loadClassNames(const std::string &namesPath)
{
    // create a filestream object
    std::ifstream ifsObj(namesPath.c_str());
    std::string line;
    while(std::getline(ifsObj, line))
    {
        _classNames.push_back(line);
    }
    std::cout << "Class names file successfully loaded." << std::endl;
}

// get the names of the output layers
std::vector<cv::String> getOutputNames(const cv::dnn::Net &net)
{
    static std::vector<cv::String> names;
    if(names.empty())
    {
        // get the indicies of the unconnected output layers
        std::vector<int> outLayers = net.getUnconnectedOutLayers();
        // get the names of the layers in the network
        std::vector<cv::String> layersNames = net.getLayerNames();
        // resize names string to size of network output layers 
        names.resize(outLayers.size());
        // map names of output layers to names vector
        for(size_t i = 0; i < outLayers.size(); ++i)
        {
            names[i] = layersNames[outLayers[i] - 1];
        }
    }
    return names;
}

// post process the frames by removing the bounding boxes with low confidence
void postprocess(cv::Mat &frame, const std::vector<cv::Mat> &outs)
{
    std::vector<int> classIDs;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    // scan through all the bounding boxes output from the network and keep the 
    // ones with high confidence scores. Assign box's class label with the class 
    // that has the highest score for that particular box.
    for(size_t i = 0; i < outs.size(); ++i)
    {
        float *data = (float*)outs[i].data;
        for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
        {
            cv::Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
            cv::Point classIdPoint;
            double confidence;
            // Get the value and location of the maximum score
            cv::minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
            if (confidence > 0.5)
            {
                int centerX = (int)(data[0] * frame.cols);
                int centerY = (int)(data[1] * frame.rows);
                int width = (int)(data[2] * frame.cols);
                int height = (int)(data[3] * frame.rows);
                int left = centerX - width / 2;
                int top = centerY - height / 2;
                
                classIDs.push_back(classIdPoint.x);
                confidences.push_back((float)confidence);
                boxes.push_back(cv::Rect(left, top, width, height));
            }
        }
    }
    // perform non-maiximum suppression to eliminat redundant overlapping 
    // boxes with low conidence scores.
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, 0.5, 0.4, indices);
    for (size_t i = 0; i < indices.size(); ++i)
    {
        int idx = indices[i];
        cv::Rect box = boxes[idx];
        // drawPredictions(classIDs[idx], confidences[idx], box.x, box.y,
        //          box.x + box.width, box.y + box.height, frame);
    }
}

// draw the predicted bounding boxes on a frame
void Yolo3::drawPredictions(int classID, float conf, int left, int top, int right, int bottom, cv::Mat &frame)
{
    // draw a rectangle displaying the bounding box
    cv::rectangle(frame, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(0,0,255), 2);
    // get the label for the class name and it's confidence score
    std::string label = cv::format("%.2f", conf);
    if (!_classNames.empty())
    {
        //CV_Assert(classId < (int)classes.size());
        label = _classNames[classID] + ":" + label;
    }
    //Display the label at the top of the bounding box
    int baseLine;
    cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.35, 1, &baseLine);
    top = cv::max(top, labelSize.height);
    cv::rectangle(frame, cv::Point(left, top - round(1.5*labelSize.height)), cv::Point(left + round(1.5*labelSize.width), top + baseLine), cv::Scalar(255, 255, 255), cv::FILLED);
    cv::putText(frame, label, cv::Point(left, top), cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0,0,0),1);

}

