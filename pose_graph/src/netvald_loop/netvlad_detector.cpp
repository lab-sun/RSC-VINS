#include "netvlad_detector.h"

#include <omp.h>
#include <ros/ros.h>

#include <cmath>
#include <future>
#include <iostream>
#include <limits>
#include <thread>

NetvladDetector::NetvladDetector(const float loop_score) : loop_score_(loop_score) {}

NetvladDetector::~NetvladDetector() {
  std::cout << "Finalizing Python..." << std::endl;
  // Py_Finalize();  // 在一切 Python 对象销毁之后
}

void NetvladDetector::Process(const int keyframe_inx, const cv::Mat& input_image,
                              const std::vector<float>& input_descriptor,
                              const Eigen::Vector3d& key_pos, int* const loop_index) {
  image_pool_.insert({keyframe_inx, input_image});
  NetvladDetectLoop(keyframe_inx, input_descriptor, key_pos, loop_index);
}

void NetvladDetector::NetvladDetectLoop(const int keyframe_inx,
                                        const std::vector<float>& net_descriptor,
                                        const Eigen::Vector3d& key_pos, int* const loop_index) {
  if (loop_index == nullptr) {
    throw std::runtime_error("loop_index is null");
    return;
  }
  if (net_descriptor.size() != 4096) {
    std::cout << "Netvlad descriptor size is not valid" << std::endl;
    return;
  }
  netvlad_descriptors_.insert({keyframe_inx, net_descriptor});
  keypos_.insert({keyframe_inx, key_pos});

  float best_score = -1.0f;
  int best_index = -1;
  std::vector<std::pair<int, std::vector<float>>> candidates;
  for (const auto& kv : netvlad_descriptors_) {
    if (kv.first == keyframe_inx || kv.first >= keyframe_inx - 15) {
      continue;
    }
    if (keypos_.find(kv.first) == keypos_.end()) {
      std::cout << "Key position not found for keyframe index: " << kv.first << std::endl;
      continue;
    }
    const Eigen::Vector3d& pos = keypos_.at(kv.first);
    if ((pos - key_pos).norm() > max_loop_dis_) {
      continue;
    }
    candidates.emplace_back(kv);
  }

  const size_t N = candidates.size();
  std::vector<float> scores(N, -1.0f);
  std::vector<int> indices(N, -1);
  omp_set_num_threads(4);
#pragma omp parallel for
  for (int i = 0; i < static_cast<int>(N); ++i) {
    const auto& [idx, desc] = candidates[i];

    float dot = 0.0f;
    for (size_t j = 0; j < 4096; ++j) {
      dot += net_descriptor[j] * desc[j];  // dot product
    }

    scores[i] = dot;
    indices[i] = idx;
  }

  for (int i = 0; i < static_cast<int>(N); ++i) {
    if (scores[i] > best_score) {
      best_score = scores[i];
      best_index = indices[i];
    }
  }

  if (best_score > loop_score_) {
    // // std::cout << "Loop detected with score: " << best_score << " at index: " << best_index
    // << std::endl;
  } else {
    std::cout << "No loop detected, best score: " << best_score << std::endl;
  }

  *loop_index = best_index;
}
