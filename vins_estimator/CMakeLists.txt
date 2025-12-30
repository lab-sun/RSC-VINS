cmake_minimum_required(VERSION 2.8.3)
project(vins_estimator)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++14")
#-DEIGEN_USE_MKL_ALL")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

find_package(catkin REQUIRED COMPONENTS
    roscpp
    std_msgs
    geometry_msgs
    nav_msgs
    tf
    cv_bridge
    visualization_msgs
    sensor_msgs
    roslib
    )

find_package(OpenCV REQUIRED)

# message(WARNING "OpenCV_VERSION: ${OpenCV_VERSION}")

find_package(Ceres REQUIRED)

find_package(HDF5 REQUIRED COMPONENTS CXX)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)

find_package(OpenMP REQUIRED)
find_package(FAISS REQUIRED)

include_directories(${HDF5_INCLUDE_DIRS})
include_directories(${FAISS_INCLUDE_DIRS})

message(STATUS "Project Source Dir: ${PROJECT_SOURCE_DIR}")

# /home/zty/Desktop/LSY/catkin_ws_v1/src/VINS-Mono/vins_estimator
include_directories(
    ${PROJECT_SOURCE_DIR}/../pose_graph/src/netvald_loop
    ${HDF5_INCLUDE_DIRS}
    ${Python3_INCLUDE_DIRS}
)

include_directories(${catkin_INCLUDE_DIRS} ${CERES_INCLUDE_DIRS})

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
find_package(Eigen3)
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
)

catkin_package()

add_executable(vins_estimator
    src/estimator_node.cpp
    src/parameters.cpp
    src/estimator.cpp
    src/feature_manager.cpp
    src/factor/pose_local_parameterization.cpp
    src/factor/projection_factor.cpp
    src/factor/projection_td_factor.cpp
    src/factor/marginalization_factor.cpp
    src/utility/utility.cpp
    src/utility/visualization.cpp
    src/utility/CameraPoseVisualization.cpp
    src/initial/solve_5pts.cpp
    src/initial/initial_aligment.cpp
    src/initial/initial_sfm.cpp
    src/initial/initial_ex_rotation.cpp
    src/netvlad_initial/netvlad_initializer.cpp
    ../pose_graph/src/netvald_loop/netvlad_inference.cpp
    )


target_link_libraries(vins_estimator PRIVATE ${catkin_LIBRARIES} ${OpenCV_LIBS} ${CERES_LIBRARIES}) 

target_link_libraries(vins_estimator PRIVATE
    ${HDF5_LIBRARIES}
    hdf5_cpp           
    faiss
    pthread
    m
    ${HDF5_LIBRARIES}
    Python3::Python
    pybind11::module
		${FAISS_LIBRARIES}
    OpenMP::OpenMP_CXX
)

