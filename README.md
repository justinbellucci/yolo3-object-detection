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
  * The OpenCV 4.1.0 source code can be found [here](https://github.com/opencv/opencv/tree/4.1.0)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools](https://developer.apple.com/xcode/features/)
* NOTE: This is tested using Mac OSX 10.15

## Build Instructions

1. Clone the repository and navigate to the downloaded folder.
	
	```	
		git clone https://github.com/justinbellucci/yolo3-object-detection.git
		cd yolo3-object-detection
	```
2. Download the models
    ```	
		sudo chmod a+x getModels.sh
        ./getModels.sh
	```
3. Make a build directory in the top level directory:   
    ```
        mkdir build && cd build
    ```
3. Compile 
    ```
        cmake .. 
        make
    ```
## Running the Program
The input to this program can be either a mp4 video file or a webcam. Once run, a window will open and the video will play automatically. Once finished, the program will terminate the window. 
1. To run with a video file:

    ```
    ./yolo3_detector --video=/my_video.mp4
    ```
2. To run with a webcam leave the argument blank.

    ```
    ./yolo3_detector 
    ```

## Class Structure


## Rubric Points 
### Loops, Functions, I/O

| Point                                                                                          | File       | Lines          |
|------------------------------------------------------------------------------------------------|------------|----------------|
| The project demonstrates an understanding of C++ functions and control structures.             | all        | -              |
| The project reads data from a file and process the data, or the program writes data to a file. | [main.cpp] |                |
| The project accepts user input and processes the input.                                        | [main.cpp] | 10, 21         |

### Object Oriented Programming

| Point                                                                                          | File       | Lines          |
|------------------------------------------------------------------------------------------------|------------|----------------|
| The project uses Object Oriented Programming techniques.                                       | all        | -              |
| Classes use appropriate access specifiers for class members.                                   | [main.cpp] |                |
| Class constructors utilize member initialization lists.                                        | [main.cpp] | 10, 21         |
| Classes encapsulate behavior.                                                                  |            |                |

### Memory Management

| Point                                                                                          | File       | Lines          |
|------------------------------------------------------------------------------------------------|------------|----------------|
| The project makes use of references in function declarations.                                  | all        | -              |
| The project uses destructors appropriately.                                                    | [main.cpp] |                |
| The project follows the Rule of 5.                                                             | [main.cpp] | 10, 21         |
| The project uses move semantics to move data, instead of copying it, where possible.           | [main.cpp] | 10             |                           
| The project uses smart pointers instead of raw pointers.                                       |            |                |

[main.cpp]: src/main.cpp
[yolo3.cpp]: src/yolo3.cpp
[model.cpp]: src/model.cpp

