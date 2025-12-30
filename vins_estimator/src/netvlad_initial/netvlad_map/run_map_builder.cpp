#include "map_builder.h"

int main() {
  MapBuilder builder(
      "/home/zty/Desktop/LSY/catkin_ws_v1/src/VINS-Mono/vins_estimator/src/netvlad_initial/EuRoC/"
      "v2_03/data",
      "/home/zty/Desktop/LSY/catkin_ws_v1/src/VINS-Mono/vins_estimator/src/netvlad_initial/EuRoC/"
      "v2_03/truth.txt",
      "/home/zty/Desktop/LSY/catkin_ws_v1/src/VINS-Mono/vins_estimator/src/netvlad_initial/EuRoC/"
      "euroc_global_netmap.h5",
      "v2_03");
  builder.Process();

  return 0;
}