cmake_minimum_required(VERSION 3.10)

project(NetVladExample)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV REQUIRED)
find_package(HDF5 REQUIRED COMPONENTS CXX)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)

find_package(OpenMP REQUIRED)

# 查找 FAISS 库
find_package(FAISS REQUIRED)

include_directories(${HDF5_INCLUDE_DIRS})
include_directories(${FAISS_INCLUDE_DIRS})

message(STATUS "Project Source Dir: ${PROJECT_SOURCE_DIR}")

include_directories(
    ${PROJECT_SOURCE_DIR}/../../../pose_graph/src/netvald_loop
    ${OpenCV_INCLUDE_DIRS}
    ${HDF5_INCLUDE_DIRS}
    ${Python3_INCLUDE_DIRS}
)

set(SOURCES
    run_netvlad_initializer.cpp
    netvlad_initializer.cpp
		${PROJECT_SOURCE_DIR}/../../../pose_graph/src/netvald_loop/netvlad_inference.cpp
)

add_executable(NetVladExample ${SOURCES})


# 如果 HDF5 或 FAISS 需要额外的库，请在此处指定（例如，FAISS 可能需要指定 BLAS 和 LAPACK 库）
target_link_libraries(NetVladExample PRIVATE
    ${HDF5_LIBRARIES}
    hdf5_cpp           # 添加 C++ 接口库
    faiss
    pthread
    m
		${OpenCV_LIBS}
    ${HDF5_LIBRARIES}
    Python3::Python
    pybind11::module
		${FAISS_LIBRARIES}
    OpenMP::OpenMP_CXX
)
