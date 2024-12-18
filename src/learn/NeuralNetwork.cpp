#include "NeuralNetwork.h"
#include "NeuralNetworkParams.h"
#include "Activation.h"
#include "DenseLayer.h"
#include "ReLU.h"
#include "Sigmoid.h"

#include <algorithm>
#include <iostream>
#include <stdexcept>

namespace learn {

NeuralNetwork::NeuralNetwork(const NeuralNetworkShape& shape)
: _shape{shape} {
  if (!_shape.is_valid()) {
    throw std::invalid_argument("Invalid neural network shape");
  }

  for (const auto& layer_shape : _shape.layers) {
    std::unique_ptr<Activation> activation;
    std::unique_ptr<Layer> layer;

    switch (layer_shape.activation) {
      case ActivationType::ReLU:
        activation = std::make_unique<ReLU>();
        break;
      case ActivationType::Sigmoid:
        activation = std::make_unique<Sigmoid>();
        break;
    }

    switch (layer_shape.type) {
      case LayerType::Dense:
        layer = std::make_unique<DenseLayer>(layer_shape.input_size,
                                             layer_shape.output_size);
        break;
    }

    addActivationAndLayer(std::move(activation), std::move(layer));
  }
}



// Adds a layer and its corresponding activation function to the network
void NeuralNetwork::addActivationAndLayer(
    std::unique_ptr<Activation> activation, std::unique_ptr<Layer> layer) {
  activations.push_back(std::move(activation));
  layers.push_back(std::move(layer));
}

// Forward pass: processes the input through all layers and activations
std::vector<double> NeuralNetwork::forward(const std::vector<double> &input) {
  std::vector<double> output = input;

  for (size_t i = 0; i < layers.size(); ++i) {
    output = layers[i]->forward(output);      // Pass through layer
    output = activations[i]->forward(output); // Pass through activation
  }

  return output;
}

// Backward pass: propagates gradients back through all layers and activations
void NeuralNetwork::backward(const std::vector<double> &grad_output) {
  std::vector<double> grad = grad_output;

  for (int i = layers.size() - 1; i >= 0; --i) {
    grad = activations[i]->backward(grad); // Backward through activation
    grad = layers[i]->backward(grad);      // Backward through layer
  }
}

// Training loop: performs forward and backward passes and updates weights
void NeuralNetwork::train(const std::vector<std::vector<double>> &inputs,
                          const std::vector<std::vector<double>> &targets,
                          double learning_rate, int epochs) {
  for (int epoch = 0; epoch < epochs; ++epoch) {
    double loss = 0.0;

    for (size_t i = 0; i < inputs.size(); ++i) {
      std::vector<double> output = forward(inputs[i]); // Forward pass

      // Compute loss (e.g., Mean Squared Error)
      std::vector<double> grad_output(output.size());
      for (size_t j = 0; j < output.size(); ++j) {
        double error = output[j] - targets[i][j];
        grad_output[j] = 2.0 * error; // Derivative for MSE
        loss += error * error;
      }

      backward(grad_output); // Backward pass

      // Update weights for each layer
      for (auto &layer : layers) {
        layer->updateWeights(learning_rate);
      }
    }

    // Average loss per epoch
    loss /= inputs.size();
    std::cout << "Epoch " << epoch + 1 << " - Loss: " << loss << std::endl;
  }
}

// New predict method to classify an input and return the predicted class
std::vector<double> NeuralNetwork::predict(const std::vector<double> &input) {
  return forward(input);
}

size_t NeuralNetwork::input_size() const { return layers[0]->input_size(); }

size_t NeuralNetwork::output_size() const {
  if (layers.empty()) {
    throw std::runtime_error("No layers in the network");
  }
  return layers.back()->output_size();
}

} // namespace learn
