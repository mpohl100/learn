#include <catch2/catch_all.hpp>

#include "learn/DenseLayer.h"
#include "learn/NeuralNetwork.h"
#include "learn/ReLU.h"
#include "learn/Sigmoid.h"
#include "learn/TrainingSession.h"

#include <iostream>

namespace {

struct DumbTrainingSession : learn::TrainingSession {
  virtual ~DumbTrainingSession() = default;

  DumbTrainingSession(const learn::TrainingParams &params)
      : learn::TrainingSession{params}, _params{params} {}

  learn::TrainingSession::SessionData prepare_data() const override {
    // Initialize inputs and targets with zeros
    int num_samples = 1000; // Number of training samples
    int input_size = 784;   // Input dimension (e.g., 28x28 image flattened)
    int num_classes =
        10; // Number of output classes (e.g., for digit classification)

    std::vector<std::vector<double>> inputs(
        num_samples, std::vector<double>(input_size, 0.0));
    std::vector<std::vector<double>> targets(
        num_samples, std::vector<double>(num_classes, 0.0));

    return {inputs, targets};
  }

private:
  learn::TrainingParams _params;
};

TEST_CASE("NeuralNet", "[neuralnet]") {
  SECTION("SimpleNeuralNetWorks") {

    const auto nn_shape = learn::NeuralNetworkShape{{
        {784, 784, learn::LayerType::Dense, learn::ActivationType::ReLU},
        {784, 128, learn::LayerType::Dense, learn::ActivationType::ReLU},
        {128, 10, learn::LayerType::Dense, learn::ActivationType::Sigmoid},
    }};

    const auto training_params =
        learn::TrainingParams{nn_shape, 700, 300, 0.01, 10, 0.1};

    const auto training_session = DumbTrainingSession{training_params};
    auto success_rate = training_session.train();
    CHECK(success_rate >= 0.9);
  }
}

} // namespace