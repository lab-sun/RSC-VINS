#pragma once

#include "netvlad_inference.h"
#include <Eigen/Dense>
#include <atomic>
#include <map>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <unordered_map>

class NetvladDetector {
 public:
  NetvladDetector(const float loop_score);

  ~NetvladDetector();

  void Process(const int keyframe_inx, const cv::Mat& input_image,
               const std::vector<float>& input_descriptor, const Eigen::Vector3d& key_pos,
               int* const loop_index);
  void NetvladDetectLoop(const int keyframe_inx, const std::vector<float>& input_descriptor,
                         const Eigen::Vector3d& key_pos, int* const loop_index);

 private:
  std::unordered_map<int, std::vector<float>> netvlad_descriptors_;
  std::unordered_map<int, cv::Mat> image_pool_;
  std::unordered_map<int, Eigen::Vector3d> keypos_;
  std::shared_ptr<ImageInferenceWrapper> inference_wrapper_ = nullptr;
  float loop_score_ = 0.0;
  float max_loop_dis_ = 30.0;  // meters
};