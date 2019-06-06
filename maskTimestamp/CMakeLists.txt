cmake_minimum_required(VERSION 3.0.0)

# Configure OpenCV dependency
find_package(OpenCV REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )

project(maskTimestamp VERSION 0.1 LANGUAGES CXX)
add_executable(${PROJECT_NAME} "main.cpp")
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBRARIES})
