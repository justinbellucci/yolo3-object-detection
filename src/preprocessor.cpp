#include "preprocessor.h"

Preprocessor::Preprocessor(Preprocessor&& source) noexcept {}

void Preprocessor::process(cv::Mat &frame, cv::dnn::Net &net, YoloConfig::FrameProcessingData &data)
{
    // create a blob object
    cv::Mat blob;

    // check image input dimensions
    if(data.inpWidth <= 0)
    {
        data.inpWidth = frame.cols; // equal to frame column dims
    }
    if(data.inpHeight <= 0)
    {
        data.inpHeight = frame.rows; // equal to frame row dims
    }

    // create a blob from the frame
    cv::dnn::blobFromImage(frame, blob, 1.0, cv::Size(data.inpWidth, data.inpHeight), cv::Scalar(0,0,0));
    // set the input to the network
    net.setInput(blob);
}