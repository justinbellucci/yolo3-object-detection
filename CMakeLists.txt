cmake_minimum_required(VERSION 3.11.3)

project(yolo3_object_detection)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

set(OpenCV_DIR /usr/local/OpenCV_install/lib/cmake/opencv4)
find_package(OpenCV 4.5 REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(include)

link_directories(${OpenCV_LIBRARY_DIRS})
add_definitions(${OpenCV_DEFINITIONS})

# Find all executables
file(GLOB project_SRCS src/*.cpp)

# Add project executables
add_executable(yolo3_detector ${project_SRCS})
target_link_libraries(yolo3_detector ${OpenCV_LIBRARIES})
