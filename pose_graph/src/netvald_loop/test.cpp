#include "netvlad_detector.h"

int main() {
  try {
    ImageInferenceWrapper inference_wrapper(
        "/home/linlin/NetVLAD-based-VINS/pybind_cpp/mobilenetvlad_depth-0.35");

    std::vector<std::string> image_paths = {
        "/home/linlin/NetVLAD-based-VINS/pybind_cpp/data/image1.png",
        "/home/linlin/NetVLAD-based-VINS/pybind_cpp/data/image2.png",
        "/home/linlin/NetVLAD-based-VINS/pybind_cpp/data/image3.png"};

    for (const auto &image_path : image_paths) {
      std::vector<float> descriptor = inference_wrapper.infer_image(image_path);
      std::cout << "Descriptor size: " << descriptor.size() << std::endl;

      if (image_path.find("image1.png") != std::string::npos) {
        std::ofstream outfile(
            "/home/linlin/NetVLAD-based-VINS/"
            "pybind_multi_cpp/descriptor_image1.txt");
        if (!outfile) {
          std::cerr << "Failed to open file for writing descriptor." << std::endl;
          continue;
        }

        for (const auto &val : descriptor) {
          outfile << val << "\n";
        }
        outfile.close();
        std::cout << "Descriptor for image1.png saved to descriptor_image1.txt" << std::endl;
      }
    }

  } catch (const std::exception &e) {
    std::cerr << "[Main Error] " << e.what() << std::endl;
    return 1;
  }

  std::cout << "Finalizing Python..." << std::endl;
  Py_Finalize();  // 在一切 Python 对象销毁之后

  return 0;
}