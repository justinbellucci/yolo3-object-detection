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
Model::~Model()
{
    std::cout << "DELETING instance of Model at " << this << std::endl;
}

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

Model Model::initialize(const cv::String &configPath, const cv::String &weightsPath, float confThreshold, float nmsThreshold,
                        cv::dnn::Target backend)
{
    cv::dnn::Net net = cv::dnn::readNetFromDarknet(configPath, weightsPath); 
    net.setPreferableBackend(backend);

    return Model(net, confThreshold, nmsThreshold);
}

cv::dnn::Net Model::processFrames(cv::Mat &frame, struct YoloConfig::FrameProcessingData &data)
{
    cv::Mat blob;
    // Create a 4D blob from a frame.
    cv::dnn::blobFromImage(frame, blob, 1/255.0, cv::Size(data.inpWidth, data.inpHeight), cv::Scalar(0,0,0), true, false);
        
    //Sets the input to the network
    _net->setInput(blob);
    return *_net;
}
