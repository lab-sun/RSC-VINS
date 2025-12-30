#include "map_builder.h"

#include "netvlad_inference.h"
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <sstream>

MapBuilder::MapBuilder(const std::string& image_folder, const std::string& pose_file,
                       const std::string& hdf5_file, const std::string& area_id)
    : image_folder_(image_folder)
    , pose_file_(pose_file)
    , hdf5_file_(hdf5_file)
    , area_id_(area_id) {}

std::vector<Pose> MapBuilder::LoadPoseFile(const std::string& filename) {
  std::vector<Pose> poses;
  std::ifstream in(filename);
  std::string line;
  while (std::getline(in, line)) {
    std::istringstream ss(line);
    Pose p;
    ss >> p.timestamp >> p.tx >> p.ty >> p.tz >> p.qx >> p.qy >> p.qz >> p.qw;
    poses.push_back(p);
  }
  return poses;
}

std::vector<std::string> MapBuilder::GetImageFiles(const std::string& folder) {
  std::vector<std::string> files;
  for (const auto& entry : std::filesystem::directory_iterator(folder)) {
    if (entry.path().extension() == ".png") {
      files.push_back(entry.path().string());
    }
  }
  std::sort(files.begin(), files.end());
  return files;
}

double MapBuilder::ExtractTimestampFromFilename(const std::string& name) {
  return std::stod(name) * 1e-9;
}

Pose MapBuilder::InterpolatePose(double ts, const std::vector<Pose>& poses) {
  for (size_t i = 0; i < poses.size() - 1; ++i) {
    if (poses[i].timestamp <= ts && ts <= poses[i + 1].timestamp) {
      double alpha = (ts - poses[i].timestamp) / (poses[i + 1].timestamp - poses[i].timestamp);
      Pose interp;
      interp.timestamp = ts;
      interp.tx = (1 - alpha) * poses[i].tx + alpha * poses[i + 1].tx;
      interp.ty = (1 - alpha) * poses[i].ty + alpha * poses[i + 1].ty;
      interp.tz = (1 - alpha) * poses[i].tz + alpha * poses[i + 1].tz;
      interp.qx = (1 - alpha) * poses[i].qx + alpha * poses[i + 1].qx;
      interp.qy = (1 - alpha) * poses[i].qy + alpha * poses[i + 1].qy;
      interp.qz = (1 - alpha) * poses[i].qz + alpha * poses[i + 1].qz;
      interp.qw = (1 - alpha) * poses[i].qw + alpha * poses[i + 1].qw;
      return interp;
    }
  }
  return poses.front();
}

std::string MapBuilder::GetAreaFromTimestamp(double ts) {
  if (ts < 1403636580.0)
    return "area_1";
  else if (ts < 1403636590.0)
    return "area_2";
  else
    return "area_3";
}

std::vector<float> MapBuilder::ExtractNetVLADDescriptor(const std::string& image_path) {
  std::vector<float> desc(4096, 0.0f);  // Replace with actual inference
  ImageInferenceWrapper inference_wrapper(
      "/home/zty/Desktop/LSY/catkin_ws_v1/src/VINS-Mono/pose_graph/src/netvald_loop/"
      "mobilenetvlad_depth-0.35");
  std::vector<float> descriptor = inference_wrapper.infer_image(image_path);
  return desc;
}

void MapBuilder::SaveToHDF5(const std::string& area_id, const std::string& image_id,
                            const Pose& pose, const std::vector<float>& descriptor) {
  try {
    std::cout << "Opening HDF5 file: " << hdf5_file_ << std::endl;
    H5::H5File file(hdf5_file_, H5F_ACC_RDWR);
    std::cout << "File opened successfully!" << std::endl;
    H5::Group area_group;
    try {
      area_group = file.openGroup("/" + area_id);
      std::cout << "Opened existing group: " << area_id << std::endl;
    } catch (...) {
      area_group = file.createGroup("/" + area_id);
      std::cout << "Created new group: " << area_id << std::endl;
    }
    H5::Group img_group = area_group.createGroup(image_id);
    std::cout << "Created image group: " << image_id << std::endl;
    hsize_t pose_dims[1] = {7};
    H5::DataSpace pose_space(1, pose_dims);
    H5::DataSet pose_set = img_group.createDataSet("pose", H5::PredType::NATIVE_DOUBLE, pose_space);
    double pose_data[7] = {pose.tx, pose.ty, pose.tz, pose.qx, pose.qy, pose.qz, pose.qw};
    pose_set.write(pose_data, H5::PredType::NATIVE_DOUBLE);
    std::cout << "Pose data written successfully!" << std::endl;
    hsize_t desc_dims[1] = {descriptor.size()};
    H5::DataSpace desc_space(1, desc_dims);
    H5::DataSet desc_set =
        img_group.createDataSet("netvlad", H5::PredType::NATIVE_FLOAT, desc_space);
    desc_set.write(descriptor.data(), H5::PredType::NATIVE_FLOAT);
    std::cout << "Descriptor data written successfully!" << std::endl;
  } catch (const H5::Exception& e) {
    std::cerr << "HDF5 Error: " << e.getDetailMsg() << std::endl;
  }
}

void MapBuilder::Process() {
  poses_ = LoadPoseFile(pose_file_);
  image_files_ = GetImageFiles(image_folder_);

  H5::H5File file(hdf5_file_, H5F_ACC_RDWR);
  file.close();

  ImageInferenceWrapper inference_wrapper(
      "/home/zty/Desktop/LSY/catkin_ws_v1/src/VINS-Mono/pose_graph/src/netvald_loop/"
      "mobilenetvlad_depth-0.35");

  for (const auto& image_path : image_files_) {
    cv::Mat img = cv::imread(image_path);
    if (img.empty()) continue;

    std::string name = std::filesystem::path(image_path).stem().string();
    double ts = ExtractTimestampFromFilename(name);
    Pose pose = InterpolatePose(ts, poses_);
    std::vector<float> desc = inference_wrapper.infer_image(image_path);
    // std::string area_id = GetAreaFromTimestamp(ts);

    SaveToHDF5(area_id_, name, pose, desc);
  }
}
