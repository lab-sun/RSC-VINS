# RSC-VINS

RSC-VINS is an enhanced version of VINS-Mono that integrates **NetVLAD-based initialization** and **loop closure detection** capabilities. It is a real-time SLAM framework for **Monocular Visual-Inertial Systems** that uses an optimization-based sliding window formulation for providing high-accuracy visual-inertial odometry. 

**Repository:** [https://github.com/lab-sun/RSC-VINS.git](https://github.com/lab-sun/RSC-VINS.git)

## Key Features

RSC-VINS extends the original VINS-Mono with the following enhancements:

- **NetVLAD-based Initialization** (`vins_estimator/src/netvlad_initial`): Utilizes pre-computed NetVLAD descriptors stored in HDF5 format to provide robust initial pose estimation. The module uses FAISS for efficient similarity search and retrieves top-K similar images to initialize the system with accurate poses.

- **NetVLAD-based Loop Closure Detection** (`pose_graph/src/netvald_loop`): Implements real-time loop closure detection using NetVLAD descriptors. The module performs parallel similarity computation using OpenMP to efficiently match current keyframes with historical keyframes, enabling robust loop closure detection for pose graph optimization.

- All original VINS-Mono features: efficient IMU pre-integration with bias correction, automatic estimator initialization, online extrinsic calibration, failure detection and recovery, global pose graph optimization, map merge, pose graph reuse, online temporal calibration, and rolling shutter support.

This code runs on **Linux**, and is fully integrated with **ROS**.

## 1. Prerequisites

### 1.1 **Ubuntu** and **ROS**
- Ubuntu 16.04 or later
- ROS Kinetic or later. [ROS Installation](http://wiki.ros.org/ROS/Installation)
- Additional ROS packages:
```bash
sudo apt-get install ros-YOUR_DISTRO-cv-bridge ros-YOUR_DISTRO-tf ros-YOUR_DISTRO-message-filters ros-YOUR_DISTRO-image-transport
```

### 1.2 **Ceres Solver**
Follow [Ceres Installation](http://ceres-solver.org/installation.html), use **version 1.14.0** and remember to **sudo make install**. (There are compilation issues in Ceres versions 2.0.0 and above.)

### 1.3 **Additional Dependencies for NetVLAD Modules**
- **FAISS**: For efficient similarity search in NetVLAD initialization
- **HDF5**: For reading pre-computed NetVLAD descriptors
- **Pybind11**: For Python inference interface in loop closure detection
- **TensorFlow/PyTorch**: For NetVLAD model inference (depending on your model format)

## 2. Build RSC-VINS on ROS

Clone the repository and catkin_make:
```bash
cd ~/catkin_ws/src
git clone https://github.com/lab-sun/RSC-VINS.git
cd ../
catkin_make
source ~/catkin_ws/devel/setup.bash
```

## 3. Visual-Inertial Odometry and Pose Graph Reuse on Public datasets

Download [EuRoC MAV Dataset](http://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets). Although it contains stereo cameras, we only use one camera. We take EuRoC as the example.

### 3.1 Visual-inertial odometry and loop closure

#### 3.1.1 Open three terminals, launch the vins_estimator, rviz and play the bag file respectively. Take MH_01 for example:
```bash
roslaunch vins_estimator euroc.launch 
roslaunch vins_estimator vins_rviz.launch
rosbag play YOUR_PATH_TO_DATASET/MH_01_easy.bag 
```
(If you fail to open vins_rviz.launch, just open an empty rviz, then load the config file: file -> Open Config-> YOUR_VINS_FOLDER/config/vins_rviz_config.rviz)

#### 3.1.2 (Optional) Visualize ground truth. We write a naive benchmark publisher to help you visualize the ground truth. It uses a naive strategy to align VINS with ground truth. Just for visualization, not for quantitative comparison on academic publications.
```bash
roslaunch benchmark_publisher publish.launch  sequence_name:=MH_05_difficult
```
(Green line is VINS result, red line is ground truth). 

#### 3.1.3 (Optional) You can even run EuRoC **without extrinsic parameters** between camera and IMU. We will calibrate them online. Replace the first command with:
```bash
roslaunch vins_estimator euroc_no_extrinsic_param.launch
```
**No extrinsic parameters** in that config file. Waiting a few seconds for initial calibration. Sometimes you cannot feel any difference as the calibration is done quickly.

### 3.2 Map merge

After playing MH_01 bag, you can continue playing MH_02 bag, MH_03 bag ... The system will merge them according to the loop closure.

### 3.3 Map reuse

#### 3.3.1 Map save

Set the **pose_graph_save_path** in the config file (YOUR_VINS_FOLDER/config/euroc/euroc_config.yaml). After playing MH_01 bag, input **s** in vins_estimator terminal, then **enter**. The current pose graph will be saved. 

#### 3.3.2 Map load

Set the **load_previous_pose_graph** to 1 before doing 3.1.1. The system will load previous pose graph from **pose_graph_save_path**. Then you can play MH_02 bag. New sequence will be aligned to the previous pose graph.

## 4. AR Demo

4.1 Download the [bag file](https://www.dropbox.com/s/s29oygyhwmllw9k/ar_box.bag?dl=0), which is collected from HKUST Robotic Institute. For friends in mainland China, download from [bag file](https://pan.baidu.com/s/1geEyHNl).

4.2 Open three terminals, launch the ar_demo, rviz and play the bag file respectively.
```bash
roslaunch ar_demo 3dm_bag.launch
roslaunch ar_demo ar_rviz.launch
rosbag play YOUR_PATH_TO_DATASET/ar_box.bag 
```
We put one 0.8m x 0.8m x 0.8m virtual box in front of your view. 

## 5. Run with your device 

Suppose you are familiar with ROS and you can get a camera and an IMU with raw metric measurements in ROS topic, you can follow these steps to set up your device.

5.1 Change to your topic name in the config file. The image should exceed 20Hz and IMU should exceed 100Hz. Both image and IMU should have the accurate time stamp. IMU should contain absolute acceleration values including gravity.

5.2 Camera calibration:

We support the [pinhole model](http://docs.opencv.org/2.4.8/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html) and the [MEI model](http://www.robots.ox.ac.uk/~cmei/articles/single_viewpoint_calib_mei_07.pdf). You can calibrate your camera with any tools you like. Just write the parameters in the config file in the right format. If you use rolling shutter camera, please carefully calibrate your camera, making sure the reprojection error is less than 0.5 pixel.

5.3 **Camera-IMU extrinsic parameters**:

If you have seen the config files for EuRoC and AR demos, you can find that we can estimate and refine them online. If you familiar with transformation, you can figure out the rotation and position by your eyes or via hand measurements. Then write these values into config as the initial guess. Our estimator will refine extrinsic parameters online. If you don't know anything about the camera-IMU transformation, just ignore the extrinsic parameters and set the **estimate_extrinsic** to **2**, and rotate your device set at the beginning for a few seconds. When the system works successfully, we will save the calibration result. you can use these result as initial values for next time. An example of how to set the extrinsic parameters is in [extrinsic_parameter_example](https://github.com/HKUST-Aerial-Robotics/VINS-Mono/blob/master/config/extrinsic_parameter_example.pdf)

5.4 **Temporal calibration**:
Most self-made visual-inertial sensor sets are unsynchronized. You can set **estimate_td** to 1 to online estimate the time offset between your camera and IMU.  

5.5 **Rolling shutter**:
For rolling shutter camera (carefully calibrated, reprojection error under 0.5 pixel), set **rolling_shutter** to 1. Also, you should set rolling shutter readout time **rolling_shutter_tr**, which is from sensor datasheet(usually 0-0.05s, not exposure time). Don't try web camera, the web camera is so awful.

5.6 Other parameter settings: Details are included in the config file.

5.7 Performance on different devices: 

(global shutter camera + synchronized high-end IMU, e.g. VI-Sensor) > (global shutter camera + synchronized low-end IMU) > (global camera + unsync high frequency IMU) > (global camera + unsync low frequency IMU) > (rolling camera + unsync low frequency IMU). 

## 6. Docker Support

## 7. Core Modules

### 7.1 NetVLAD-based Initialization (`vins_estimator/src/netvlad_initial`)

This module provides robust initialization for the visual-inertial system using pre-computed NetVLAD descriptors:

- **Functionality**: Loads NetVLAD descriptors and corresponding poses from HDF5 files, uses FAISS for efficient similarity search to find top-K similar images, and provides initial pose estimates for system startup.

- **Key Components**:
  - `NetVladInitializer`: Main class that manages descriptor loading and query operations
  - HDF5 data loading with multi-threaded processing for efficiency
  - FAISS-based L2 distance search for fast similarity matching
  - Pose validation through consistency checking of top-K results

- **Usage**: The module is integrated into the estimator initialization pipeline and automatically provides initial pose estimates when starting the system in a known environment.

### 7.2 NetVLAD-based Loop Closure Detection (`pose_graph/src/netvald_loop`)

This module implements real-time loop closure detection using NetVLAD descriptors:

- **Functionality**: Computes NetVLAD descriptors for incoming keyframes, performs parallel similarity search against historical keyframes, and detects loop closures for pose graph optimization.

- **Key Components**:
  - `NetvladDetector`: Main detector class that manages descriptor storage and loop detection
  - `ImageInferenceWrapper`: Python inference interface for NetVLAD descriptor extraction
  - OpenMP-based parallel similarity computation for efficient matching
  - Spatial filtering to avoid false positives from nearby keyframes

- **Usage**: The module is integrated into the pose graph optimization pipeline and automatically detects loop closures during system operation, enabling global map consistency.

## 8. Acknowledgements

RSC-VINS is based on [VINS-Mono](https://github.com/HKUST-Aerial-Robotics/VINS-Mono) by the HKUST Aerial Robotics Group. We use [ceres solver](http://ceres-solver.org/) for non-linear optimization, [DBoW2](https://github.com/dorian3d/DBoW2) for loop detection, [FAISS](https://github.com/facebookresearch/faiss) for efficient similarity search, and a generic [camera model](https://github.com/hengli/camodocal).

## 9. License

The source code is released under [GPLv3](http://www.gnu.org/licenses/) license.

## 10. Citation

If you use RSC-VINS for your academic research, please cite the original VINS-Mono papers and our work:

* **VINS-Mono: A Robust and Versatile Monocular Visual-Inertial State Estimator**, Tong Qin, Peiliang Li, Zhenfei Yang, Shaojie Shen, IEEE Transactions on Robotics [pdf](https://ieeexplore.ieee.org/document/8421746/?arnumber=8421746&source=authoralert)

* **Online Temporal Calibration for Monocular Visual-Inertial Systems**, Tong Qin, Shaojie Shen, IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS, 2018), **best student paper award** [pdf](https://ieeexplore.ieee.org/abstract/document/8593603)

* @INPROCEEDINGS{11277738,
  author={Lin, Shuyue and Sun, Yuxiang},
  booktitle={2025 International Conference on Information and Automation (ICIA)}, 
  title={Region-based Initialization and Spatial-constrained Loop Closure Detection for Efficient Visual-inertial SLAM}, 
  year={2025},
  volume={},
  number={},
  pages={353-358},
  keywords={Location awareness;Simultaneous localization and mapping;Accuracy;Automation;Liquid crystal displays;Robustness;Real-time systems},
  doi={10.1109/ICIA64617.2025.11277738}}
