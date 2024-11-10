include_directories( ${CMAKE_SOURCE_DIR}/src )

add_library(neuralnet SHARED NeuralNetwork.cpp DenseLayer.cpp TrainingSession.cpp NeuralNetworkParams.cpp)
target_include_directories(neuralnet PUBLIC ${CMAKE_SOURCE_DIR}/src)
