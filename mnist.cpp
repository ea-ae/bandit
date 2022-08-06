#include "mnist.h"
#include <iostream>
#include <fstream>
#include <format>
#include <algorithm>
#include <cmath>
#include <string>
#include <array>
#include <chrono>
#include "DataLoaders/DataLoader.h"
#include "DataLoaders/MnistDataLoader.h"
#include "NeuralNetworks/ClassificationNeuralNetwork.h"
#include "ActivationFunctions/LeakyRelu.h"
#include "CostFunctions/QuadraticCost.h"

using namespace std::chrono;

void mnist() {
    // Configuration

    const auto LEARNING_RATE_ETA = 0.1f; // default: 0.01
    const auto REGULARIZATION_LAMBDA = 0.001f; // no regularization: 0
    const auto MOMENTUM_COEFFICIENT_MU = 0.2f; // no momentum: 0
    const auto RELU_LEAK = 0.1f; // no leak: 0

    const auto INPUT_NEURONS = 784;
    const auto OUTPUT_NEURONS = 10;
    const auto HIDDEN_LAYERS = std::vector<int32_t>{ 400 };

    const auto BATCH_SIZE = 32;

    auto costFunction = QuadraticCost(REGULARIZATION_LAMBDA, MOMENTUM_COEFFICIENT_MU);
    auto activationFunction = LeakyRelu(RELU_LEAK);

    // Initialize neural network

    auto net = ClassificationNeuralNetwork(activationFunction, costFunction,
        INPUT_NEURONS, OUTPUT_NEURONS, HIDDEN_LAYERS);

    // Load datasets

    auto trainingDataSet = MnistDataLoader("./training-images.idx3-ubyte", "./training-labels.idx1-ubyte");
    std::cout << std::format("Finished reading {} training data items into memory\n", trainingDataSet.size());
    auto testingDataSet = MnistDataLoader("./test-images.idx3-ubyte", "./test-labels.idx1-ubyte");
    std::cout << std::format("Finished reading {} testing data items into memory\n", testingDataSet.size());
    
    // Print status

    std::stringstream hlStringStream;
    std::copy(HIDDEN_LAYERS.begin(), HIDDEN_LAYERS.end(), std::ostream_iterator<int32_t>(hlStringStream, "-"));
    std::string hlString = hlStringStream.str();
    hlString.pop_back();

    std::cout << std::format("eta = {} | HL = {} | lambda = {} | mu = {}\n",
        LEARNING_RATE_ETA, hlString, REGULARIZATION_LAMBDA, MOMENTUM_COEFFICIENT_MU);

    // Begin learning

    int32_t epoch = 1, done = 0;
    auto trainingStart = steady_clock::now();

    while (true) {
        auto epochStart = steady_clock::now();
        int32_t trainingCorrect = 0, testingCorrect = 0;
        std::optional<int8_t> label;

        // Training

        while ((label = trainingDataSet.loadDataItem(net)).has_value()) {
            net.calculateOutput();
            net.backpropagate(label.value());

            if (net.getHighestOutputNode() == label.value()) trainingCorrect++;

            if (++done == BATCH_SIZE) {
                done = 0;
                net.update(BATCH_SIZE, LEARNING_RATE_ETA);
            }
        }
        trainingDataSet.resetDataIterator();

        // Testing

        while ((label = testingDataSet.loadDataItem(net)).has_value()) {
            net.calculateOutput();
            if (net.getHighestOutputNode() == label) testingCorrect++;
        }
        testingDataSet.resetDataIterator();

        // Calculate and print stats

        float trainPassRate = (100 * trainingCorrect) / (float)trainingDataSet.size();
        float testPassRate = (100 * testingCorrect) / (float)testingDataSet.size();

        auto epochEnd = steady_clock::now();
        auto epochDuration = duration_cast<seconds>(epochEnd - epochStart).count();
        auto totalDuration = duration_cast<seconds>(epochEnd - trainingStart).count();

        std::cout << std::format("Epoch {:03} | training: {:.2f}%, testing: {:.2f}% | took: {}s, total: {}s\n",
            epoch, trainPassRate, testPassRate, epochDuration, totalDuration);

        epoch++;
    }
}
