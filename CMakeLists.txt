cmake_minimum_required(VERSION 3.11.3)

project(yolo3_object_detection)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

find_package(OpenCV 4.5 REQUIRED)

include_directories(
    include
    ${OpenCV_INCLUDE_DIRS}
)

link_directories(${OpenCV_LIBRARY_DIRS})
add_definitions(${OpenCV_DEFINITIONS})

# Find all executables
file(GLOB project_SRCS src/*.cpp)

# Add project executables
add_executable(yolo3_detector ${project_SRCS})
target_link_libraries(yolo3_detector ${OpenCV_LIBRARIES})

target_link_directories(yolo3_detector PRIVATE ${OpenCV_LIBRARY_DIRS})
target_compile_definitions(yolo3_detector PRIVATE ${OpenCV_DEFINITIONS})