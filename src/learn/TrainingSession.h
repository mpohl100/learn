#pragma once

#include "NeuralNetwork.h"

#include <stdexcept>
#include <vector>

namespace learn {

struct TrainingParams {
  int num_training_samples = 700;
  int num_verification_samples = 300;
  double learning_rate = 0.01;
  int epochs = 10;
};

class TrainingSession {
public:
  TrainingSession(const TrainingParams& params);

  TrainingSession() = default;
  virtual ~TrainingSession() = default;
  TrainingSession(const TrainingSession &) = default;
  TrainingSession(TrainingSession &&) = default;
  TrainingSession &operator=(const TrainingSession &) = default;
  TrainingSession &operator=(TrainingSession &&) = default;

  struct SessionData {
    std::vector<std::vector<double>> inputs;
    std::vector<std::vector<double>> targets;
  };

  virtual SessionData prepare_data() const = 0;
  learn::NeuralNetwork prepare_network() const;
  void train() const;

private:
  TrainingParams _params;
};

} // namespace learn