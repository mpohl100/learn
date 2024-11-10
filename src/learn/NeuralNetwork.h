#pragma once

#include "Activation.h"
#include "Layer.h"
#include "NeuralNetworkParams.h"

#include <memory>
#include <vector>

namespace learn {

class NeuralNetwork {
public:
  NeuralNetwork() = default;
  NeuralNetwork(const NeuralNetwork &) = default;
  NeuralNetwork(NeuralNetwork &&) = default;
  NeuralNetwork &operator=(const NeuralNetwork &) = default;
  NeuralNetwork &operator=(NeuralNetwork &&) = default;

  NeuralNetwork(const NeuralNetworkShape& shape);

  std::vector<double> forward(const std::vector<double> &input);
  void backward(const std::vector<double> &grad_output);
  void train(const std::vector<std::vector<double>> &inputs,
             const std::vector<std::vector<double>> &targets,
             double learning_rate, int epochs);
  std::vector<double> predict(const std::vector<double> &input);

  size_t input_size() const;
  size_t output_size() const;
private:
  void addActivationAndLayer(std::unique_ptr<Activation> activation,
                             std::unique_ptr<Layer> layer);
  std::vector<std::unique_ptr<Layer>> layers;
  std::vector<std::unique_ptr<Activation>> activations;
  NeuralNetworkShape _shape;
};

} // namespace learn
