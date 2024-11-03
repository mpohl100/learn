#pragma once

#include "Activation.h"
#include "Layer.h"

#include <vector>
#include <memory>

namespace learn {

class NeuralNetwork {
private:
    std::vector<std::unique_ptr<Layer>> layers;
    std::vector<std::unique_ptr<Activation>> activations;

public:
    void addLayer(std::unique_ptr<Layer> layer);
    void addActivation(std::unique_ptr<Activation> activation);

    std::vector<double> forward(const std::vector<double>& input);
    void backward(const std::vector<double>& grad_output);
    void train(const std::vector<std::vector<double>>& inputs,
               const std::vector<std::vector<double>>& targets,
               double learning_rate, int epochs);
};

} // namespace learn
