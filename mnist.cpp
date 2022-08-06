#include "mnist.h"
#include <iostream>
#include <fstream>
#include <format>
#include <algorithm>
#include <cmath>
#include <array>
#include <chrono>
#include "NeuralNetworks/SupervisedNeuralNetwork.h"
#include "ActivationFunctions/LeakyRelu.h"
#include "CostFunctions/QuadraticCost.h"

using namespace std::chrono;

template <class T> // ty, stackoverflow
void endswap(T* objp) {
    unsigned char* memp = reinterpret_cast<unsigned char*>(objp);
    std::reverse(memp, memp + sizeof(T));
}

template <class T>
void read(T* buffer, std::ifstream& stream) {
    stream.read(reinterpret_cast<char*>(buffer), sizeof(T));
    endswap(buffer);
}

struct DataItem {
    std::array<uint8_t, 784> pixels;
    uint8_t label;
};

std::vector<DataItem> createDataItems(std::ifstream& data, std::ifstream& labels) {
    int32_t dataMagic, dataCount, dataRows, dataColumns;
    read<int32_t>(&dataMagic, data);
    read<int32_t>(&dataCount, data);
    read<int32_t>(&dataRows, data);
    read<int32_t>(&dataColumns, data);

    int32_t labelMagic, labelCount;
    read<int32_t>(&labelMagic, labels);
    read<int32_t>(&labelCount, labels);

    std::vector<DataItem> dataItems(dataCount);
    for (auto& dataItem : dataItems) {
        for (int i = 0; i < (dataRows * dataColumns); i++) { // for each pixel
            read<uint8_t>(&dataItem.pixels[i], data);
            /*if (nodeValue > 0.8) std::cout << "X";
            else if (nodeValue > 0.2) std::cout << "x";
            else std::cout << "_";
            if (i % dataRows == 0) std::cout << "\n";*/
        }
        //std::cout << "\n\n";
        read<uint8_t>(&dataItem.label, labels);
    }
    return dataItems;
}

void mnist() {
    const auto LEARNING_RATE_ETA = 0.1f; // default: 0.01
    const auto REGULARIZATION_LAMBDA = 0.001f; // no regularization: 0
    const auto MOMENTUM_COEFFICIENT_MU = 0.2f; // no momentum: 0
    const auto RELU_LEAK = 0.1f; // no leak: 0

    const auto INPUT_NEURONS = 784;
    const auto OUTPUT_NEURONS = 10;
    const auto HIDDEN_LAYERS = 1;
    const auto HIDDEN_LAYER_NEURONS = 300; // default: 30, 400

    const auto BATCH_SIZE = 32;

    auto costFunction = QuadraticCost(REGULARIZATION_LAMBDA, MOMENTUM_COEFFICIENT_MU);
    auto activationFunction = LeakyRelu(RELU_LEAK);

    auto perceptron = SupervisedNeuralNetwork(activationFunction, costFunction,
        INPUT_NEURONS, OUTPUT_NEURONS, HIDDEN_LAYERS, HIDDEN_LAYER_NEURONS);

    auto trainingData = std::ifstream("./training-images.idx3-ubyte", std::ios::binary);
    auto trainingLabels = std::ifstream("./training-labels.idx1-ubyte", std::ios::binary);
    auto testingData = std::ifstream("./test-images.idx3-ubyte", std::ios::binary);
    auto testingLabels = std::ifstream("./test-labels.idx1-ubyte", std::ios::binary);

    auto trainingDataSet = createDataItems(trainingData, trainingLabels);
    std::cout << std::format("Finished reading {} training data items into memory\n", trainingDataSet.size());

    auto testingDataSet = createDataItems(testingData, testingLabels);
    std::cout << std::format("Finished reading {} testing data items into memory\n", testingDataSet.size());

    std::cout << std::format("eta = {} | L = {} | Ln = {} | lambda = {} | mu = {}\n",
        LEARNING_RATE_ETA, HIDDEN_LAYERS, HIDDEN_LAYER_NEURONS, REGULARIZATION_LAMBDA, MOMENTUM_COEFFICIENT_MU);
    int32_t epoch = 1, done = 0, correctGuesses = 0;
    auto trainingStart = steady_clock::now();

    while (true) {
        auto epochStart = steady_clock::now();

        for (auto& dataItem : trainingDataSet) {
            for (int i = 0; i < dataItem.pixels.size(); i++) {
                float value = dataItem.pixels[i] / 255.0f; // [0, 1]
                perceptron.setInputNode(i, value);
            }
            perceptron.calculateOutput();
            perceptron.backpropagate(dataItem.label);

            if (perceptron.getHighestOutputNode() == dataItem.label) correctGuesses++;

            if (++done == BATCH_SIZE) {
                done = 0;
                perceptron.update(BATCH_SIZE, LEARNING_RATE_ETA);
            }
        }

        int32_t testsPassed = 0;
        for (auto& dataItem : testingDataSet) { // DRY is for weak people
            for (int i = 0; i < dataItem.pixels.size(); i++) {
                float value = dataItem.pixels[i] / 255.0f;
                perceptron.setInputNode(i, value);
            }
            perceptron.calculateOutput();
            if (perceptron.getHighestOutputNode() == dataItem.label) testsPassed++;
        }

        float trainPassRate = (100 * correctGuesses) / (float)trainingDataSet.size();
        float testPassRate = (100 * testsPassed) / (float)testingDataSet.size();

        auto epochEnd = steady_clock::now();
        auto epochDuration = duration_cast<seconds>(epochEnd - epochStart).count();
        auto totalDuration = duration_cast<seconds>(epochEnd - trainingStart).count();

        std::cout << std::format("Epoch {:03} | training: {:.2f}%, testing: {:.2f}% | took: {}s, total: {}s\n",
            epoch, trainPassRate, testPassRate, epochDuration, totalDuration);
        correctGuesses = 0;
        epoch++;
    }
}
