#include "netvlad_initializer.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>  // for std::setprecision
#include <iostream>
#include <numeric>

typedef int64_t idx_t;

NetVladInitializer::NetVladInitializer(const std::string& hdf5_file)
    : hdf5_file_(hdf5_file), index_(std::make_unique<faiss::IndexFlatL2>(4096)) {}

void NetVladInitializer::LoadDescriptors(const std::string& area_id) {
  image_ids_.clear();
  image_pose_map_.clear();
  index_->reset();

  std::vector<std::vector<float>> descriptors;
  std::vector<std::string> ids;
  std::vector<Pose> poses;
  std::mutex mtx;
  auto t0 = std::chrono::steady_clock::now();
  try {
    H5::H5File file(hdf5_file_, H5F_ACC_RDONLY);
    H5::Group area_group = file.openGroup("/" + area_id);
    hsize_t num_objs = area_group.getNumObjs();

    std::vector<std::future<void>> futures;

    for (hsize_t i = 0; i < num_objs; ++i) {
      std::string image_id = area_group.getObjnameByIdx(i);
      futures.emplace_back(std::async(std::launch::async, [&, image_id]() {
        try {
          H5::Group img_group = area_group.openGroup(image_id);

          // --- 读取 netvlad 描述子 ---
          H5::DataSet desc_set = img_group.openDataSet("netvlad");
          H5::DataSpace desc_space = desc_set.getSpace();
          hsize_t dims[1];
          desc_space.getSimpleExtentDims(dims);
          std::vector<float> desc(dims[0]);
          desc_set.read(desc.data(), H5::PredType::NATIVE_FLOAT);

          // 归一化
          float norm = std::sqrt(std::inner_product(desc.begin(), desc.end(), desc.begin(), 0.0f));

          if (norm <= 0 || !std::isfinite(norm)) {
            std::cerr << "Warning: descriptor norm invalid for image: " << image_id << std::endl;
            return;
          }

          for (auto& val : desc) val /= norm;

          // --- 读取 pose ---
          H5::DataSet pose_set = img_group.openDataSet("pose");
          double pose_arr[7];
          pose_set.read(pose_arr, H5::PredType::NATIVE_DOUBLE);
          Pose pose{pose_arr[0], pose_arr[1], pose_arr[2], pose_arr[3],
                    pose_arr[4], pose_arr[5], pose_arr[6]};

          std::lock_guard<std::mutex> lock(mtx);
          descriptors.push_back(desc);
          ids.push_back(image_id);
          poses.push_back(pose);
        } catch (const H5::Exception& e) {
          std::cerr << "Failed to load image: " << image_id << ", error: " << e.getDetailMsg()
                    << std::endl;
        }
      }));
    }

    for (auto& fut : futures) fut.get();

    auto t1 = std::chrono::steady_clock::now();  // 加载并归一化所有数据后
    std::chrono::duration<double> load_time = t1 - t0;
    std::cout << "[Timer] Data loading and normalization took: " << load_time.count() << " seconds."
              << std::endl;

    if (descriptors.empty()) {
      std::cerr << "Error: No descriptors were loaded! Check your HDF5 content." << std::endl;
      return;
    }

    std::cout << "Adding " << descriptors.size() << " descriptors to FAISS index..." << std::endl;
    auto t2 = std::chrono::steady_clock::now();

    for (size_t i = 0; i < descriptors.size(); ++i) {
      index_->add(1, descriptors[i].data());
      image_ids_.push_back(ids[i]);
      image_pose_map_[ids[i]] = poses[i];
    }

    std::cout << "Finished loading " << image_ids_.size() << " descriptors from area: " << area_id
              << std::endl;

    auto t3 = std::chrono::steady_clock::now();
    std::chrono::duration<double> index_add_time = t3 - t2;
    std::cout << "[Timer] Adding descriptors to FAISS took: " << index_add_time.count()
              << " seconds." << std::endl;

    auto t4 = std::chrono::steady_clock::now();
    std::chrono::duration<double> total_load_time = t4 - t0;
    std::cout << "[Timer] Total LoadDescriptors time: " << total_load_time.count() << " seconds."
              << std::endl;
  } catch (const H5::Exception& e) {
    std::cerr << "HDF5 Load Error: " << e.getDetailMsg() << std::endl;
  }
}

bool NetVladInitializer::QueryTopK(const std::vector<float>& query_desc, int k, Pose& result_pose,
                                   float distance_thresh) {
  auto t0 = std::chrono::steady_clock::now();
  if (index_->ntotal == 0) {
    std::cerr << "FAISS index is empty. Cannot perform search." << std::endl;
    return false;
  }

  float query_norm =
      std::sqrt(std::inner_product(query_desc.begin(), query_desc.end(), query_desc.begin(), 0.0f));
  std::cout << "Query norm: " << query_norm << std::endl;

  if (query_norm <= 0 || !std::isfinite(query_norm)) {
    std::cerr << "Invalid query descriptor: norm = " << query_norm << std::endl;
    return false;
  }

  std::vector<float> normalized_query_desc = query_desc;
  for (auto& val : normalized_query_desc) val /= query_norm;

  std::vector<float> distances(k);
  std::vector<idx_t> indices(k);
  index_->search(1, normalized_query_desc.data(), k, distances.data(), indices.data());

  std::vector<Pose> poses_for_comparison;
  std::vector<float> dist_for_comparison;

  for (int i = 0; i < k; ++i) {
    if (indices[i] < 0 || indices[i] >= image_ids_.size()) {
      std::cerr << "Invalid index returned by FAISS: " << indices[i] << std::endl;
      continue;
    }

    std::string img_id = image_ids_[indices[i]];
    Pose pose = image_pose_map_[img_id];
    poses_for_comparison.push_back(pose);
    dist_for_comparison.push_back(distances[i]);

    std::cout << "Top-" << i + 1 << " image_id: " << img_id << ", Distance: " << distances[i]
              << std::endl;
  }

  float pose_diff_sum = 0.0;
  for (int i = 1; i < poses_for_comparison.size(); ++i) {
    const Pose& p1 = poses_for_comparison[i - 1];
    const Pose& p2 = poses_for_comparison[i];
    float d = std::sqrt(std::pow(p1.tx - p2.tx, 2) + std::pow(p1.ty - p2.ty, 2) +
                        std::pow(p1.tz - p2.tz, 2));
    pose_diff_sum += d;
  }

  if (pose_diff_sum > distance_thresh) {
    std::cout << "Pose mismatch detected, returning failure." << std::endl;
    return false;
  }

  int best_index = std::min_element(dist_for_comparison.begin(), dist_for_comparison.end()) -
                   dist_for_comparison.begin();
  result_pose = poses_for_comparison[best_index];
  auto t1 = std::chrono::steady_clock::now();
  std::chrono::duration<double> total_load_time = t1 - t0;
  std::cout << "[Timer] QueryTopK time: " << total_load_time.count() << " seconds." << std::endl;
  printf("[timer] query time %f seconds\n", total_load_time.count());
  return true;
}
