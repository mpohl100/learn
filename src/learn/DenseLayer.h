#pragma once

#include "Layer.h"

#include <random>
#include <vector>

namespace learn {

class DenseLayer : public Layer {
public:
  DenseLayer(int input_size, int output_size);

  std::vector<double> forward(const std::vector<double> &input) override;
  std::vector<double> backward(const std::vector<double> &grad_output) override;
  void updateWeights(double learning_rate) override;

private:
  std::vector<std::vector<double>> weights;
  std::vector<double> biases;
  std::vector<double> input_cache;
};

} // namespace learn
