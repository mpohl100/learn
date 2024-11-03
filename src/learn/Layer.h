#pragma once

#include <vector>

namespace learn {

class Layer {
public:
  virtual std::vector<double> forward(const std::vector<double> &input) = 0;
  virtual std::vector<double>
  backward(const std::vector<double> &grad_output) = 0;
  virtual void updateWeights(double learning_rate) = 0;
  virtual ~Layer() = default;
};

} // namespace learn