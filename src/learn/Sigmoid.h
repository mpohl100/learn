#pragma once

#include "Activation.h"

#include <vector>
#include <algorithm>
#include <cmath>

namespace learn {

class Sigmoid : public Activation {
public:
    std::vector<double> forward(const std::vector<double>& input) override {
        std::vector<double> output(input.size());
        std::transform(input.begin(), input.end(), output.begin(), [](double x) {
            return 1.0 / (1.0 + std::exp(-x));
        });
        return output;
    }

    std::vector<double> backward(const std::vector<double>& grad_output) override {
        // Implement the derivative of Sigmoid
    }
};

} // namespace learn
