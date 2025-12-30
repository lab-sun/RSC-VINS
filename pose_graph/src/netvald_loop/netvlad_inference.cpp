#include "netvlad_inference.h"

ImageInferenceWrapper::ImageInferenceWrapper(const std::string& model_path) {
  try {
    pybind11::initialize_interpreter();

    pybind11::module sys = pybind11::module::import("sys");
    sys.attr("path").cast<pybind11::list>().append(
        "/home/zty/Desktop/LSY/catkin_ws_v1/src/VINS-Mono/pose_graph/src/netvald_loop");

    pybind11::print("sys.path =", sys.attr("path"));

    pybind11::module netvlad = pybind11::module::import("inference");
    image_inference_ = netvlad.attr("ImageInference")(model_path);

    std::cout << "Model loaded successfully." << std::endl;
  } catch (const pybind11::error_already_set& e) {
    std::cerr << "[Init Error] Python exception: " << e.what() << std::endl;
    throw std::runtime_error("Failed to initialize Python or load inference module.");
  }
}

ImageInferenceWrapper::~ImageInferenceWrapper() {
  std::cout << "Destroying ImageInferenceWrapper..." << std::endl;
  try {
    if (pybind11::hasattr(image_inference_, "__del__")) {
      image_inference_.attr("__del__")();
    }
    // pybind11::finalize_interpreter();
  } catch (const pybind11::error_already_set& e) {
    std::cerr << "[Finalize Error] Python exception during finalization: " << e.what() << std::endl;
  }
}

std::vector<float> ImageInferenceWrapper::infer_image(const std::string& image_path) {
  try {
    if (image_inference_.is_none()) {
      throw std::runtime_error("ImageInference object is None.");
    }
    auto start_time = std::chrono::high_resolution_clock::now();
    pybind11::object result = image_inference_.attr("infer_image")(image_path);
    if (result.is_none()) {
      throw std::runtime_error("Inference returned None.");
    }

    if (pybind11::isinstance<pybind11::list>(result)) {
      pybind11::list descriptor = result.cast<pybind11::list>();
      std::vector<float> result_vec;
      for (size_t i = 0; i < descriptor.size(); ++i) {
        result_vec.push_back(descriptor[i].cast<float>());
      }
      auto end_time = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> elapsed_time = end_time - start_time;
      // std::cout << "Time to compute descriptor for " << elapsed_time.count() << " seconds"
      //           << std::endl;

      return result_vec;
    } else {
      throw std::runtime_error("Unsupported Python return type.");
    }
  } catch (const pybind11::error_already_set& e) {
    std::cerr << "[Inference Error] Python exception: " << e.what() << std::endl;
    throw std::runtime_error("Inference failed.");
  }
}