#include "yolo3.h"

#include <memory>

// constructor
Yolo3::Yolo3(YoloConfig::FrameProcessingData &data){
    _capture = std::make_unique<cv::VideoCapture>();
    _video = std::make_unique<cv::VideoWriter>();
    _frame = std::make_unique<cv::Mat>();
    _blob = std::make_unique<cv::Mat>();
    _frameProcData = std::move(data);
}

// destructor
// set up thread barrier before object is destroyed
Yolo3::~Yolo3() = default;

// --- class methods ---
void Yolo3::run(cv::String &weights_path, cv::String &config_path, cv::String &classNames_path, cv::String &inputMedia)
{
    
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
void postprocess(cv::Mat& frame, const std::vector<cv::Mat> &outs)
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
            if (confidence > confThreshold)
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
    cv::dnn::NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indicies);
    for (size_t i = 0; i < indices.size(); ++i)
    {
        int idx = indices[i];
        cv::Rect box = boxes[idx];
        drawPredictions(classIds[idx], confidences[idx], box.x, box.y,
                 box.x + box.width, box.y + box.height, frame);
    }
    
}

