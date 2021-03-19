#include "model.h"

#include <iostream>

// constructor
Model::Model(cv::dnn::Net &net, float confThreshold, float nmsThreshold, Preprocessor &preProc)
{
    _net = std::make_unique<cv::dnn::Net>(std::move(net));
    _confThreshold = confThreshold;
    _nmsThreshold = nmsThreshold;
    _preProcessor = std::make_unique<Preprocessor>(std::move(preProc));
    
    std::cout << "CREATING instance of Model at " << this << std::endl;
}

// destructor
Model::~Model()
{
    std::cout << "DELETING instance of Model at " << this << std::endl;
}

// move constructor
Model::Model(Model &&source) 
{
    std::cout << "MOVING (c'tor) instance " << &source << " to instance " << this << std::endl;
    _net = std::move(source._net);
    _confThreshold = source._confThreshold;
    _nmsThreshold = source._nmsThreshold;
    _preProcessor = std::move(source._preProcessor);

    source._net = nullptr;
    source._confThreshold = 0.0;
    source._nmsThreshold = 0.0;
    source._preProcessor = nullptr;
}

// move assignment operator
Model &Model::operator=(Model &&source) 
{
    std::cout << "MOVING (assign) instance " << &source << " to instance " << this << std::endl;
    _net = std::move(source._net);
    _preProcessor = std::move(source._preProcessor);

    source._net = nullptr;
    source._preProcessor = nullptr;

    return *this;
}

Model Model::initialize(const cv::String &model, const cv::String &config, float confThreshold, float nmsThreshold,
                            const cv::String &framework, cv::dnn::Backend backend, cv::dnn::Target target)
{
    cv::dnn::Net net = cv::dnn::readNet(model, config, framework);
    net.setPreferableBackend(backend);
    net.setPreferableTarget(target);
    // instantiate preProcessor object of type Preprocessor
    Preprocessor preProcessor;

    return Model(net, confThreshold, nmsThreshold, preProcessor);
}

void Model::processFrames(cv::Mat &frame, struct YoloConfig::FrameProcessingData &data)
{
    _preProcessor->process(frame, *_net, data);
}

cv::AsyncArray Model::forward()
{
    return _net->forwardAsync();
}