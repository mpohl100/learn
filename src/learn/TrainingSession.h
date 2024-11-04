#pragma once

namespace learn {

class TrainingSession {
public:
  TrainingSession() = default;
  ~TrainingSession() = default;
  TrainingSession(const TrainingSession &) = default;
  TrainingSession(TrainingSession &&) = default;
  TrainingSession &operator=(const TrainingSession &) = default;
  TrainingSession &operator=(TrainingSession &&) = default;

  void train() const;
};

} // namespace learn