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

    const auto LEARNING_RATE_ETA = 0.1f; // default: 0.1
    const auto REGULARIZATION_LAMBDA = 0.001f; // no regularization: 0
    const auto MOMENTUM_COEFFICIENT_MU = 0.2f; // no momentum: 0
    const auto RELU_LEAK = 0.1f; // no leak: 0

    const auto INPUT_NEURONS = 784;
    const auto OUTPUT_NEURONS = 10;
    const auto HIDDEN_LAYERS = std::vector<int32_t>{ 30 };

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

    std::cout << std::format("eta = {} | HL = {} | lambda = {} | mu = {} | leak = {}\n",
        LEARNING_RATE_ETA, hlString, REGULARIZATION_LAMBDA, MOMENTUM_COEFFICIENT_MU, RELU_LEAK);

    // Begin learning

    int32_t epoch = 1;
    auto trainingStart = steady_clock::now();

    while (true) {
        auto epochStart = steady_clock::now();
        int32_t trainingCorrect = 0, testingCorrect = 0;
        auto dataSetRatio = static_cast<int32_t>(
            std::ceilf(static_cast<float>(trainingDataSet.size()) / testingDataSet.size()));
        
        // Training and testing

        std::optional<int8_t> trainLabel, testLabel;
        while (true) {
            int32_t batchItemsDone = 0, batchesDone = 0;

            while ((trainLabel = trainingDataSet.loadDataItem(net)).has_value()) {
                net.calculateOutput();
                net.backpropagate(trainLabel.value());

                if (net.getHighestOutputNode() == trainLabel.value()) trainingCorrect++;

                if (++batchItemsDone == BATCH_SIZE) {
                    batchItemsDone = 0;
                    net.update(BATCH_SIZE, LEARNING_RATE_ETA);
                    if (++batchesDone == dataSetRatio) break; // occasionally switch between training/testing
                }
            }

            while ((testLabel = testingDataSet.loadDataItem(net)).has_value()) {
                net.calculateOutput();
                if (net.getHighestOutputNode() == testLabel) testingCorrect++;

                if (++batchItemsDone == BATCH_SIZE) break; // do 1 testing batch for every n training batches
            }

            if (!trainLabel.has_value() && !testLabel.has_value()) break; // both data sets depleted
        }

        trainingDataSet.resetDataIterator();
        testingDataSet.resetDataIterator();

        // Calculate and print stats

        float trainPassRate = static_cast<float>(100 * trainingCorrect) / trainingDataSet.size();
        float testPassRate = static_cast<float>(100 * testingCorrect) / testingDataSet.size();

        auto epochEnd = steady_clock::now();
        auto epochDuration = duration_cast<seconds>(epochEnd - epochStart).count();
        auto totalDuration = duration_cast<seconds>(epochEnd - trainingStart).count();

        std::cout << std::format("Epoch {:03} | training: {:.2f}%, testing: {:.2f}% | took: {}s, total: {}s\n",
            epoch, trainPassRate, testPassRate, epochDuration, totalDuration);

        epoch++;
    }
}
