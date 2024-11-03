#pragma once

#include "Layer.h"
#include "matrix/Matrix.h"

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
  matrix::Matrix<double> weights;
  std::vector<double> biases;
  std::vector<double> input_cache;
  matrix::Matrix<double> weight_grads; // Gradients for weights
  std::vector<double> bias_grads;      // Gradients for biases
};

} // namespace learn
