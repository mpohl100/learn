#include "TrainingSession.h"

#include "DenseLayer.h"
#include "NeuralNetwork.h"
#include "ReLU.h"
#include "Sigmoid.h"

#include <iostream>
#include <memory>

namespace learn {

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
  const auto input_size = inputs[0].size();
  // Display dimensions to confirm
  std::cout << "Inputs: " << inputs.size() << " x " << inputs[0].size()
            << std::endl;
  std::cout << "Targets: " << targets.size() << " x " << targets[0].size()
            << std::endl;

  auto nn = prepare_network();

  // Train the neural network
  nn.train(inputs, targets, 0.01,
           10); // Train for 10 epochs with learning rate 0.01

  // Classify a sample input (all zeros in this case)
  std::vector<double> sample_input(input_size, 0.0); // Use an example input
  int predicted_class = nn.predict(sample_input);

  std::cout << "Predicted class for the sample input: " << predicted_class
            << std::endl;
}

} // namespace learn