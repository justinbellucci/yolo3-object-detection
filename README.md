# YOLOv3 Object Detection using OpenCV

This project is the Capstone Project for the [Udacity C++ Nanodegree Program](https://www.udacity.com/course/c-plus-plus-nanodegree--nd213). It detects up to 80 classes of objects from a sequence of frames using the You Only Look Once (YOLOv3) Deep Neural Network, originally authored by Joseph Redmon and Ali Farhadi. YOLOv3 forwards the the whole image, or frame, at once saving valuable inference time. It divides the image into a 13x13 grid of cells. Each cell is responsible for predicting a number of bounding boxes in the image. For each bounding box, the network predicts the confidence that the bounding box encloses an object, and outputs the probability that the object represents a particular class. Using non-maximum suppression the bounding boxes with low confidence scores (0.5) are eliminated. This technique yields surprisingly fast results.

This project also aims to demonstrate the use of modern Object Oriented Programming in C++ with the features such as smart pointers, and the rule-of-five. 

<img src="data/yoloDriving.gif"/>

## Dependencies for Running Locally
The following dependencies are required to run the program locally.
* cmake >= 3.17
  * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.3 (Linux, Mac)
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
* OpenCV >= 4.5
  * The OpenCV 4.5 source code can be found [here](https://github.com/opencv/opencv/tree/4.1.0)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools](https://developer.apple.com/xcode/features/)
* NOTE: This project is tested using Mac OSX 10.15

## Build Instructions

1. Clone the repository and navigate to the downloaded folder.
```
git clone https://github.com/justinbellucci/yolo3-object-detection.git
cd yolo3-object-detection
```

2. Download the models in the main project directory.
```
sudo chmod a+x getModels.sh
./getModels.sh
```

3. Make a build directory in the top level directory:
```
mkdir build && cd build
```
4. Compile 
  ```
  cmake .. 
  make
  ```
## Running the Program
The input to this program can be either a mp4 video file or a webcam. Both methods require the yolov3.weights, yolov3.cfg and coco.names files to be placed in the main project directory. Once run, a window will open and the video will play automatically. Once finished, the program will terminate the window.
```
// run from build directory
cd build 
```
1. Video file:

    ```
    ./yolo3_detector --video=/my_video.mp4 
    ```
2. Webcam

    ```
    ./yolo3_detector 
    ```
