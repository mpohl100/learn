#pragma once

#include "Activation.h"

#include <algorithm>
#include <vector>

namespace learn {

class ReLU : public Activation {
public:
  std::vector<double> forward(const std::vector<double> &input) override {
    std::vector<double> output(input.size());
    std::transform(input.begin(), input.end(), output.begin(),
                   [](double x) { return std::max(0.0, x); });
    return output;
  }

  std::vector<double>
  backward(const std::vector<double> &grad_output) override {
    std::vector<double> grad_input(grad_output.size());

    for (size_t i = 0; i < grad_output.size(); ++i) {
      // Calculate sigmoid(x) for the current input element
      double sigmoid_x = 1.0 / (1.0 + std::exp(-grad_output[i]));
      // Sigmoid derivative: sigmoid(x) * (1 - sigmoid(x))
      double sigmoid_derivative = sigmoid_x * (1.0 - sigmoid_x);
      grad_input[i] = grad_output[i] * sigmoid_derivative;
    }

    return grad_input;
  }
};

} // namespace learn
