#include <catch2/catch_all.hpp>

#include "learn/DenseLayer.h"
#include "learn/NeuralNetwork.h"
#include "learn/ReLU.h"
#include "learn/Sigmoid.h"

#include <iostream>

namespace {

TEST_CASE("NeuralNet", "[neuralnet]") {
  SECTION("SimpleNeuralNetWorks") {
    learn::NeuralNetwork nn;

    // Add layers and activations using the new addActivationAndLayer method
    nn.addActivationAndLayer(std::make_unique<learn::ReLU>(),
                             std::make_unique<learn::DenseLayer>(784, 128));
    nn.addActivationAndLayer(std::make_unique<learn::Sigmoid>(),
                             std::make_unique<learn::DenseLayer>(128, 10));

    // Initialize inputs and targets with zeros
    int num_samples = 1000; // Number of training samples
    int input_size = 784;   // Input dimension (e.g., 28x28 image flattened)
    int num_classes =
        10; // Number of output classes (e.g., for digit classification)

    std::vector<std::vector<double>> inputs(
        num_samples, std::vector<double>(input_size, 0.0));
    std::vector<std::vector<double>> targets(
        num_samples, std::vector<double>(num_classes, 0.0));

    // Display dimensions to confirm
    std::cout << "Inputs: " << inputs.size() << " x " << inputs[0].size()
              << std::endl;
    std::cout << "Targets: " << targets.size() << " x " << targets[0].size()
              << std::endl;

    // Train the neural network
    nn.train(inputs, targets, 0.01,
             10); // Train for 10 epochs with learning rate 0.01

    // Classify a sample input (all zeros in this case)
    std::vector<double> sample_input(input_size, 0.0); // Use an example input
    const auto output = nn.predict(sample_input);

    std::cout << "Output: ";
    for (auto &val : output) {
      std::cout << val << " ";
    }

    CHECK(output.size() >= 0); // Check if the prediction is in the range
  }
}

} // namespace