#pragma once

#include "Activation.h"
#include "Layer.h"

#include <memory>
#include <vector>

namespace learn {

class NeuralNetwork {
public:
  void addActivationAndLayer(std::unique_ptr<Activation> activation,
                             std::unique_ptr<Layer> layer);

  std::vector<double> forward(const std::vector<double> &input);
  void backward(const std::vector<double> &grad_output);
  void train(const std::vector<std::vector<double>> &inputs,
             const std::vector<std::vector<double>> &targets,
             double learning_rate, int epochs);
  int predict(const std::vector<double> &input);
private:
  std::vector<std::unique_ptr<Layer>> layers;
  std::vector<std::unique_ptr<Activation>> activations;
};

} // namespace learn
