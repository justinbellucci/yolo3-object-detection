#ifndef YOLOCONFIG_H
#define YOLOCONFIG_H

namespace YoloConfig 
{
struct FrameProcessingData 
{
    float confThreshold; // Confidence threshold
    float nmsThreshold;  // Non-maximum suppression threshold
    int inpWidth;  // Width of network's input image
    int inpHeight; // Height of network's input image
}; 
}
#endif