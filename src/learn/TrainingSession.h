#pragma once

#include "NeuralNetwork.h"

#include <vector>

namespace learn {

class TrainingSession {
public:
  TrainingSession() = default;
  virtual ~TrainingSession() = default;
  TrainingSession(const TrainingSession &) = default;
  TrainingSession(TrainingSession &&) = default;
  TrainingSession &operator=(const TrainingSession &) = default;
  TrainingSession &operator=(TrainingSession &&) = default;

  struct SessionData{
    std::vector<std::vector<double>> inputs;
    std::vector<std::vector<double>> targets;
  };

  virtual SessionData prepare_data() const = 0;
  learn::NeuralNetwork prepare_network() const;
  void train() const;
};

} // namespace learn