#pragma once

#include <vector>

namespace learn {

enum class ActivationType {
  ReLU,
  Sigmoid,
};

enum class LayerType {
  Dense,
};

struct LayerShape {
  int input_size;
  int output_size;
  LayerType type;
  ActivationType activation;
};

struct NeuralNetworkShape {
  std::vector<LayerShape> layers;

  bool is_valid() const;
};

} // namespace learn