cmake_minimum_required(VERSION 3.1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

project(maskTimestamp VERSION 0.1 LANGUAGES CXX)
add_executable(
    ${PROJECT_NAME}
    "main.cpp")

add_executable(
    fillStereo
    fillStereo.cpp
    costvolume.h
    costvolume.cpp)

# Configure OpenCV dependency
find_package(OpenCV REQUIRED core imgproc highgui calib3d stereo)
include_directories( ${OpenCV_INCLUDE_DIRS} )

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBRARIES} stdc++fs)
target_link_libraries(fillStereo ${OpenCV_LIBRARIES} stdc++fs pthread)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(fillStereo OpenMP::OpenMP_CXX)
endif()
