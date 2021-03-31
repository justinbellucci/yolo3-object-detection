#include "model.h"

#include <iostream>

// constructor
Model::Model(cv::dnn::Net &net, float confThreshold, float nmsThreshold)
{
    _net = std::make_unique<cv::dnn::Net>(std::move(net));
    _confThreshold = confThreshold;
    _nmsThreshold = nmsThreshold;

    std::cout << "CREATING instance of Model at " << this << std::endl;
}

// destructor
Model::~Model() = default;
// {
//     std::cout << "DELETING instance of Model at " << this << std::endl;
// }

// move constructor
Model::Model(Model &&source) noexcept
{
    std::cout << "MOVING (c'tor) instance " << &source << " to instance " << this << std::endl;
    _net = std::move(source._net);
    _confThreshold = source._confThreshold;
    _nmsThreshold = source._nmsThreshold;

    source._net = nullptr;
    source._confThreshold = 0.0;
    source._nmsThreshold = 0.0;
}

// move assignment operator
Model &Model::operator=(Model &&source) noexcept
{
    std::cout << "MOVING (assign) instance " << &source << " to instance " << this << std::endl;
    _net = std::move(source._net);

    source._net = nullptr;

    return *this;
}

Model Model::initialize(const cv::String &model, const cv::String &config, float confThreshold, float nmsThreshold,
                        cv::dnn::Backend backend, cv::dnn::Target target)
{
    // cv::dnn::Net net = cv::dnn::readNet(model, config);
    cv::dnn::Net net = cv::dnn::readNetFromDarknet(config, model);
    cv::dnn::Net _net = std::move(net);
    _net.setPreferableBackend(cv::dnn::DNN_TARGET_CPU);
    // instantiate preProcessor 

    return Model(_net, confThreshold, nmsThreshold);
}

void Model::processFrames(cv::Mat &frame, struct YoloConfig::FrameProcessingData &data)
{
    // _preProcessor->process(frame, *_net, data);
    static cv::Mat blob;
    cv::dnn::blobFromImage(frame, blob, 1.0, cv::Size(data.inpWidth, data.inpHeight), cv::Scalar());
    // set the input to the network
    _net->setInput(blob);
}

std::vector<cv::Mat> Model::forward() 
{
    std::vector<cv::Mat> outs; 
    _net->forward(outs, getOutputNames()); 
    return outs;
}

void Model::postProcessFrames(cv::Mat &frame, const std::vector<cv::Mat> &outs, std::vector<std::string> &names)
{
    static std::vector<int> outLayers = _net->getUnconnectedOutLayers();
    static std::string outLayerType = _net->getLayer(outLayers[0])->type;

    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    
    ////////////////////////////////////////////////////
   
    for (size_t i = 0; i < outs.size(); ++i)
    {
        // Scan through all the bounding boxes output from the network and keep only the
        // ones with high confidence scores. Assign the box's class label as the class
        // with the highest score for the box.
        float* data = (float*)outs[i].data;
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
                
                classIds.push_back(classIdPoint.x);
                confidences.push_back((float)confidence);
                boxes.push_back(cv::Rect(left, top, width, height));
            }
        }
    }
    ////////////////////////////////////////////////

    for (size_t idx = 0; idx < boxes.size(); ++idx)
    {
        cv::Rect box = boxes[idx];
        drawPreds(classIds[idx], confidences[idx], box.x, box.y, box.x + box.width, box.y + box.height, frame, names);
    }
    
}

// Draw the predicted bounding box
void Model::drawPreds(int classId, float conf, int left, int top, int right, int bottom, cv::Mat &frame, std::vector<std::string> &classes)
{
    //Draw a rectangle displaying the bounding box
    cv::rectangle(frame, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(0, 255, 0 ));
    
    //Get the label for the class name and its confidence
    std::string label = cv::format("%.2f", conf);
    if (!classes.empty())
    {
        CV_Assert(classId < (int)classes.size());
        label = classes[classId] + ":" + label;
    }
    
    //Display the label at the top of the bounding box
    int baseLine;
    cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    top = cv::max(top, labelSize.height);
    cv::rectangle(frame, cv::Point(left, top - labelSize.height), cv::Point(left + labelSize.width, top + baseLine), cv::Scalar(255, 255, 255), cv::FILLED);
    cv::putText(frame, label, cv::Point(left, top), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar());
}

std::vector<cv::String> Model::getOutputNames()
{
    static std::vector<cv::String> outNames;
    if(outNames.empty())
    {
        // outNames = net.getUnconnectedOutLayersNames();
        // get the indicies of the unconnected output layers
        std::vector<int> outLayers = _net->getUnconnectedOutLayers();
        // get the names of all the layers in the networks
        std::vector<cv::String> layersNames = _net->getLayerNames();

        // get the names of the output layers
        outNames.resize(outLayers.size());
        for(size_t i = 0; i < outLayers.size(); ++i)
        {
            outNames[i] = layersNames[outLayers[i] - 1];
        }
    }
    return outNames;
}