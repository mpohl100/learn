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
                           std::make_unique<learn::DenseLayer>(784, 784));
  nn.addActivationAndLayer(std::make_unique<learn::ReLU>(),
                           std::make_unique<learn::DenseLayer>(784, 128));
  nn.addActivationAndLayer(std::make_unique<learn::Sigmoid>(),
                           std::make_unique<learn::DenseLayer>(128, 10));

  return nn;
}

double TrainingSession::train() const {
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

  // Use the trained neural network to predict the output for an example input
  size_t success_count = 0;
  for(size_t i = 0; i < _params.num_verification_samples; ++i) {
    if(_params.num_training_samples + i >= inputs.size()) {
      throw std::runtime_error("Not enough verification samples");
    }
    auto output = nn.predict(inputs[_params.num_training_samples + i]);
    auto target = targets[_params.num_training_samples + i];
    // Check if the output matches the target
    if(std::equal(output.begin(), output.end(), target.begin(), [this](double out, double t) { return std::abs(out - t) < _params.tolerance;  })) {
      success_count++;
    } 
  }
  return success_count / static_cast<double>(_params.num_verification_samples);
}

} // namespace learn