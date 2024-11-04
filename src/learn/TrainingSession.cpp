#include "TrainingSession.h"

#include "DenseLayer.h"
#include "NeuralNetwork.h"
#include "ReLU.h"
#include "Sigmoid.h"

#include <iostream>
#include <memory>

namespace learn {

void TrainingSession::train() const {
  // Initialize inputs and targets with zeros
  int num_samples = 1000; // Number of training samples
  int input_size = 784;   // Input dimension (e.g., 28x28 image flattened)
  int num_classes =
      10; // Number of output classes (e.g., for digit classification)

  std::vector<std::vector<double>> inputs(num_samples,
                                          std::vector<double>(input_size, 0.0));
  std::vector<std::vector<double>> targets(
      num_samples, std::vector<double>(num_classes, 0.0));

  // Display dimensions to confirm
  std::cout << "Inputs: " << inputs.size() << " x " << inputs[0].size()
            << std::endl;
  std::cout << "Targets: " << targets.size() << " x " << targets[0].size()
            << std::endl;

  learn::NeuralNetwork nn;

  // Add layers and activations using the new addActivationAndLayer method
  nn.addActivationAndLayer(std::make_unique<learn::ReLU>(),
                           std::make_unique<learn::DenseLayer>(input_size, 128));
  nn.addActivationAndLayer(std::make_unique<learn::Sigmoid>(),
                           std::make_unique<learn::DenseLayer>(128, num_classes));

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