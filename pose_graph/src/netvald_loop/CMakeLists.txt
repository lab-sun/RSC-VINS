cmake_minimum_required(VERSION 3.10)

project(PythonInferenceExample)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

set(Python3_INCLUDE_DIRS "/usr/include/python3.8")
set(Python3_LIBRARIES "/usr/lib/x86_64-linux-gnu/libpython3.8.so")

find_package(pybind11 REQUIRED)

find_package(OpenCV REQUIRED)

set(CMAKE_CXX_STANDARD 14)

add_executable(inference_wrapper test.cpp netvlad_inference.cpp)

target_link_libraries(inference_wrapper PRIVATE Python3::Python pybind11::module ${OpenCV_LIBS})

target_include_directories(inference_wrapper PRIVATE 
    ${Python3_INCLUDE_DIRS} 
    ${pybind11_INCLUDE_DIRS} 
    ${OpenCV_INCLUDE_DIRS})
