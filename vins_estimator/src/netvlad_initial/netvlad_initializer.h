#pragma once

#include <H5Cpp.h>
#include <faiss/IndexFlat.h>

#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

struct Pose {
  double tx, ty, tz;
  double qx, qy, qz, qw;
};

class NetVladInitializer {
 public:
  explicit NetVladInitializer(const std::string& hdf5_file);
  void LoadDescriptors(const std::string& area_id);
  bool QueryTopK(const std::vector<float>& query_desc, int k, Pose& result_pose,
                 float distance_thresh = 3.0);

 private:
  std::string hdf5_file_;
  std::unique_ptr<faiss::IndexFlatL2> index_;
  std::vector<std::string> image_ids_;
  std::unordered_map<std::string, Pose> image_pose_map_;
};
