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
    std::vector<double> grad_input(input_cache.size());

    for (size_t i = 0; i < input_cache.size(); ++i) {
      // Gradient is passed through if input_cache[i] > 0; otherwise, it is
      // zeroed out
      grad_input[i] = input_cache[i] > 0 ? grad_output[i] : 0.0;
    }

    return grad_input;
  }
};

} // namespace learn
