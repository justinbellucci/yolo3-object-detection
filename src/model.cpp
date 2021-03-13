#include "model.h"

#include <iostream>

// constructor
Model::Model(cv::dnn::Net &net, float confThreshold, float nmsThreshold)
{
    _net = std::make_unique<cv::dnn::Net>(std::move(net));

    _confThreshold = confThreshold;
    _nmsThreshold = nmsThreshold;
}

// destructor
Model::~Model() = default;

// copy constructor
//Model::Model(const Model &source) {}

// copy assignment operator
//Model &Model::operator=(const Model &source) {}

// move constructor
Model::Model(Model &&source)
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
Model &Model::operator=(Model &&source) 
{
    std::cout << "MOVING (assign) instance " << &source << " to instance " << this << std::endl;
    _net = std::move(source._net);

    source._net = nullptr;

    return *this;
}

Model Model::initialize(const cv::String &model, const cv::String &config, float confThreshold, float nmsThreshold,
                            const cv::String &framework, cv::dnn::Backend backend, cv::dnn::Target target)
{
    cv::dnn::Net net = cv::dnn::readNet(model, config, framework);
    net.setPreferableBackend(backend);
    net.setPreferableTarget(target);

    return Model(net, confThreshold, nmsThreshold);
}