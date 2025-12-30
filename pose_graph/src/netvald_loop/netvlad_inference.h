#pragma once

#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <chrono>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

class ImageInferenceWrapper {
public:
  // ImageInferenceWrapper();
  ImageInferenceWrapper(const std::string& model_path);
  ~ImageInferenceWrapper();

  std::vector<float> infer_image(const std::string& image_path);

private:
  pybind11::object image_inference_;
};