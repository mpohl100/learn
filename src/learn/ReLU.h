#pragma once

#include "Activation.h"

#include <vector>
#include <algorithm>

namespace learn {

class ReLU : public Activation {
public:
    std::vector<double> forward(const std::vector<double>& input) override {
        std::vector<double> output(input.size());
        std::transform(input.begin(), input.end(), output.begin(), [](double x) {
            return std::max(0.0, x);
        });
        return output;
    }

    std::vector<double> backward(const std::vector<double>& grad_output) override {
        // Implement the derivative of ReLU
    }
};

} // namespace learn
