cmake_minimum_required(VERSION 3.11.3)

project(yolo3_object_detection)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

set(OpenCV_DIR /usr/local/OpenCV_install/lib/cmake/opencv4)
find_package(OpenCV 4.5 REQUIRED)

# Find all executables
file(GLOB project_SRCS src/*.cpp)

# target_include_directories(
#    include
#    /usr/local/OpenCV_install/include/opencv4
# )

# Add project executables
add_executable(yolo3_detector ${project_SRCS})
target_link_libraries(yolo3_detector ${OpenCV_LIBRARIES})

target_include_directories(yolo3_detector PRIVATE
    include
    /usr/local/OpenCV_install/include/opencv4
)

target_link_directories(yolo3_detector PRIVATE ${OpenCV_LIBRARY_DIRS})
target_compile_definitions(yolo3_detector PRIVATE ${OpenCV_DEFINITIONS})