cmake_minimum_required(VERSION 3.10)
project(NetVLAD_HDF5_MapBuilder)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

message(STATUS "Cmake Source Dir: ${CMAKE_SOURCE_DIR}")

add_definitions(-DNETVLAD_MODEL_DIR="${CMAKE_SOURCE_DIR}/../../../../../../pose_graph/src/netvald_loop/mobilenetvlad_depth-0.35")

find_package(OpenCV REQUIRED)
find_package(HDF5 REQUIRED COMPONENTS CXX)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)

message(STATUS "Project Source Dir: ${PROJECT_SOURCE_DIR}")

include_directories(
    ${PROJECT_SOURCE_DIR}/../../../../pose_graph/src/netvald_loop
    ${OpenCV_INCLUDE_DIRS}
    ${HDF5_INCLUDE_DIRS}
    ${Python3_INCLUDE_DIRS}
)

add_executable(map_builder
    run_map_builder.cpp
    map_builder.cpp
		${PROJECT_SOURCE_DIR}/../../../../pose_graph/src/netvald_loop/netvlad_inference.cpp
)

target_link_libraries(map_builder PRIVATE
    ${OpenCV_LIBS}
    ${HDF5_LIBRARIES}
    Python3::Python
    pybind11::module
)
