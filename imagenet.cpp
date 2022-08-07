#include "imagenet.h"
#include <iostream>
#include <fstream>
#include <format>
#include <algorithm>
#include <cmath>
#include <string>
#include <array>
#include <chrono>
#include "DataLoaders/ImageNetDataLoader.h"
#include "NeuralNetworks/ClassificationNeuralNetwork.h"
#include "ActivationFunctions/LeakyRelu.h"
#include "CostFunctions/QuadraticCost.h"

using namespace std::chrono;

void imagenet() {
    // Configuration

    const auto LEARNING_RATE_ETA = 0.1f; // default: 0.1
    const auto REGULARIZATION_LAMBDA = 0.001f; // no regularization: 0
    const auto MOMENTUM_COEFFICIENT_MU = 0.2f; // no momentum: 0
    const auto RELU_LEAK = 0.1f; // no leak: 0

    const auto INPUT_NEURONS = 784;
    const auto OUTPUT_NEURONS = 10;
    const auto HIDDEN_LAYERS = std::vector<int32_t>{ 1000 };

    const auto BATCH_SIZE = 32;

    auto costFunction = QuadraticCost(REGULARIZATION_LAMBDA, MOMENTUM_COEFFICIENT_MU);
    auto activationFunction = LeakyRelu(RELU_LEAK);

    // Initialize neural network

    auto net = ClassificationNeuralNetwork(activationFunction, costFunction,
        INPUT_NEURONS, OUTPUT_NEURONS, HIDDEN_LAYERS);

    // Load datasets

    auto trainingDataSet = ImageNetDataLoader("val", ".ubyte", 11);
    // std::cout << std::format("Finished reading {} training data items into memory\n", trainingDataSet.size());

    // Print status

    std::stringstream hlStringStream;
    std::copy(HIDDEN_LAYERS.begin(), HIDDEN_LAYERS.end(), std::ostream_iterator<int32_t>(hlStringStream, "x"));
    std::string hlString = hlStringStream.str();
    hlString.pop_back();

    std::cout << std::format("eta = {} | HL = {} | lambda = {} | mu = {} | leak = {}\n",
        LEARNING_RATE_ETA, hlString, REGULARIZATION_LAMBDA, MOMENTUM_COEFFICIENT_MU, RELU_LEAK);
}
