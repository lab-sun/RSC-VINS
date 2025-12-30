#pragma once

#include "H5Cpp.h"
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <string>
#include <vector>

struct Pose {
  double timestamp;
  double tx, ty, tz;
  double qx, qy, qz, qw;
};

class MapBuilder {
public:
  MapBuilder(const std::string& image_folder,
      const std::string& pose_file,
      const std::string& hdf5_file,
      const std::string& area_id);
  ~MapBuilder() = default;
  void Process();

private:
  std::string image_folder_;
  std::string pose_file_;
  std::string hdf5_file_;
  std::string area_id_;
  std::vector<Pose> poses_;
  std::vector<std::string> image_files_;

  std::vector<Pose> LoadPoseFile(const std::string& filename);
  std::vector<std::string> GetImageFiles(const std::string& folder);
  double ExtractTimestampFromFilename(const std::string& name);
  Pose InterpolatePose(double ts, const std::vector<Pose>& poses);
  std::string GetAreaFromTimestamp(double ts);
  std::vector<float> ExtractNetVLADDescriptor(const std::string& image_path);
  void SaveToHDF5(const std::string& area_id,
      const std::string& image_id,
      const Pose& pose,
      const std::vector<float>& descriptor);
};
