#include "NeuralNetworkParams.h"

#include <stddef.h>

namespace learn {

bool NeuralNetworkShape::is_valid() const {
  if (layers.empty()) {
    return false;
  }
  for (size_t i = 0; i < layers.size() - 1; ++i) {
    if (layers[i].output_size != layers[i + 1].input_size) {
      return false;
    }
  }
  return true;
}

} // namespace learn