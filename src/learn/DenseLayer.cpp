#include "DenseLayer.h"
#include <algorithm>
#include <numeric>
#include <random>

namespace learn {

DenseLayer::DenseLayer(int input_size, int output_size)
    : weights(output_size, input_size), biases(output_size, 0.0),
      weight_grads(output_size, input_size), bias_grads(output_size, 0.0) {
  // Initialize weights and biases with small random values
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(-0.1, 0.1);

  for (size_t i = 0; i < weights.rows(); ++i) {
    for (size_t j = 0; j < weights.cols(); ++j) {
      weights.get(i, j) = dis(gen);
    }
  }
  std::fill(biases.begin(), biases.end(), 0.0);
}

std::vector<double> DenseLayer::forward(const std::vector<double> &input) {
  input_cache = input; // Store input for backward pass
  std::vector<double> output(biases.size());

  for (size_t i = 0; i < biases.size(); ++i) {
    output[i] = biases[i];
    for (size_t j = 0; j < input.size(); ++j) {
      output[i] += weights.get(i, j) * input[j];
    }
  }

  return output;
}

std::vector<double>
DenseLayer::backward(const std::vector<double> &grad_output) {
  std::vector<double> grad_input(input_cache.size(), 0.0);

  // Calculate gradients for weights and biases
  for (size_t i = 0; i < weights.rows(); ++i) {
    for (size_t j = 0; j < input_cache.size(); ++j) {
      weight_grads.get(i, j) += grad_output[i] * input_cache[j];
    }
    bias_grads[i] += grad_output[i];
  }

  // Calculate gradient with respect to the input for backpropagation
  for (size_t i = 0; i < weights.rows(); ++i) {
    for (size_t j = 0; j < input_cache.size(); ++j) {
      grad_input[j] += weights.get(i, j) * grad_output[i];
    }
  }

  return grad_input;
}

void DenseLayer::updateWeights(double learning_rate) {
  // Update weights and biases using the accumulated gradients
  for (size_t i = 0; i < weights.rows(); ++i) {
    for (size_t j = 0; j < weights.cols(); ++j) {
      weights.get(i, j) -= learning_rate * weight_grads.get(i, j);
      weight_grads.get(i, j) = 0.0; // Reset gradient after update
    }
    biases[i] -= learning_rate * bias_grads[i];
    bias_grads[i] = 0.0; // Reset gradient after update
  }
}

} // namespace learn