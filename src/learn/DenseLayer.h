#pragma once

#include "Layer.h"

#include <random>
#include <vector>

namespace learn {

#ifndef DENSE_LAYER_H
#define DENSE_LAYER_H

#include "Layer.h"
#include <vector>

class DenseLayer : public Layer {
public:
    DenseLayer(int input_size, int output_size);

    std::vector<double> forward(const std::vector<double>& input) override;
    std::vector<double> backward(const std::vector<double>& grad_output) override;
    void updateWeights(double learning_rate) override;

private:
    std::vector<std::vector<double>> weights;
    std::vector<double> biases;
    std::vector<double> input_cache;
    std::vector<std::vector<double>> weight_grads;  // Gradients for weights
    std::vector<double> bias_grads;                 // Gradients for biases
};

#endif // DENSE_LAYER_H


} // namespace learn
