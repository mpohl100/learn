#include "TrainingSession.h"

#include "DenseLayer.h"
#include "NeuralNetwork.h"
#include "ReLU.h"
#include "Sigmoid.h"

#include <iostream>
#include <memory>

namespace learn {

TrainingSession::TrainingSession(const TrainingParams &params)
    : _params{params} {
  if (params.num_training_samples <= 0) {
    throw std::invalid_argument("Number of training samples must be positive");
  }
  if (params.num_verification_samples <= 0) {
    throw std::invalid_argument(
        "Number of verification samples must be positive");
  }
  if (params.learning_rate <= 0) {
    throw std::invalid_argument("Learning rate must be positive");
  }
  if (params.learning_rate >= 1) {
    throw std::invalid_argument("Learning rate must be less than 1");
  }
  if (params.epochs <= 0) {
    throw std::invalid_argument("Number of epochs must be positive");
  }
}

learn::NeuralNetwork TrainingSession::prepare_network() const {
  learn::NeuralNetwork nn;

  // Add layers and activations using the new addActivationAndLayer method
  nn.addActivationAndLayer(std::make_unique<learn::ReLU>(),
                           std::make_unique<learn::DenseLayer>(784, 128));
  nn.addActivationAndLayer(std::make_unique<learn::Sigmoid>(),
                           std::make_unique<learn::DenseLayer>(128, 10));

  return nn;
}

void TrainingSession::train() const {
  const auto [inputs, targets] = prepare_data();
  std::vector<std::vector<double>> training_inputs;
  std::vector<std::vector<double>> training_targets;
  for (size_t i = 0; i < _params.num_training_samples; ++i) {
    if (i >= inputs.size()) {
      throw std::runtime_error("Not enough training samples");
    }
    training_inputs.push_back(inputs[i]);
    training_targets.push_back(targets[i]);
  }
  const auto input_size = training_inputs[0].size();
  // Display dimensions to confirm
  std::cout << "Inputs: " << inputs.size() << " x " << inputs[0].size()
            << std::endl;
  std::cout << "Targets: " << targets.size() << " x " << targets[0].size()
            << std::endl;

  auto nn = prepare_network();
  if (nn.input_size() != input_size) {
    throw std::runtime_error("Input size mismatch with neural network");
  }
  if (nn.output_size() != targets[0].size()) {
    throw std::runtime_error("Output size mismatch with neural network");
  }

  // Train the neural network
  nn.train(training_inputs, training_targets, _params.learning_rate,
           _params.epochs); // Train for 10 epochs with learning rate 0.01

  // Classify a sample input (all zeros in this case)
  std::vector<double> sample_input(input_size, 0.0); // Use an example input
  int predicted_class = nn.predict(sample_input);

  std::cout << "Predicted class for the sample input: " << predicted_class
            << std::endl;
}

} // namespace learn