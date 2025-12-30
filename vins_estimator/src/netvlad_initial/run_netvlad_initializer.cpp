#include "netvlad_inference.h"
#include "netvlad_initializer.h"
#include <chrono>
#include <iostream>
#include <vector>

int main() {
  auto t0 = std::chrono::high_resolution_clock::now();
  std::string hdf5_file =
      "/home/zty/Desktop/LSY/catkin_ws_v1/src/VINS-Mono/vins_estimator/src/netvlad_initial/EuRoC/"
      "mh_01.h5";
  NetVladInitializer netvlad_initializer(hdf5_file);

  std::string area_id = "mh_01";
  netvlad_initializer.LoadDescriptors(area_id);

  ImageInferenceWrapper inference_wrapper(
      "/home/zty/Desktop/LSY/catkin_ws_v1/src/VINS-Mono/pose_graph/src/netvald_loop/"
      "mobilenetvlad_depth-0.35");
  std::vector<float> query_desc = inference_wrapper.infer_image(
      "/home/zty/Desktop/LSY/catkin_ws_v1/src/VINS-Mono/vins_estimator/src/netvlad_initial/EuRoC/"
      "mh_01/data/1403636580263555584.png");
  // std::vector<float> query_desc(4096, 0.0f);
  Pose result_pose;

  bool success = netvlad_initializer.QueryTopK(query_desc, 3, result_pose);
  auto t1 = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed_time = t1 - t0;
  printf("Time to netvlad initlizer for %f seconds\n", elapsed_time.count());
  if (success) {
    std::cout << "Best matching pose: "
              << "tx: " << result_pose.tx << ", ty: " << result_pose.ty
              << ", tz: " << result_pose.tz << ", qx: " << result_pose.qx
              << ", qy: " << result_pose.qy << ", qz: " << result_pose.qz
              << ", qw: " << result_pose.qw << std::endl;
  } else {
    std::cout << "Pose matching failed." << std::endl;
  }

  return 0;
}
