#include "perceptron.h"
#include <iostream>
#include <fstream>
#include <format>
#include <algorithm>
#include <cmath>
#include <array>
#include "SupervisedNeuralNetwork.h"
#include "ActivationFunctions/Relu.h"

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

void perceptron() {
    std::cout << "Creating (kind of a) perceptron\n";

    const auto activationFunction = Relu();
    const auto inputNodes = 784;
    const auto outputNodes = 10;
    const auto hiddenLayers = 2;
    const auto hiddenLayerNeurons = 30; // default: 30, 400
    const auto batchSize = 32;
    const auto learningRate = 0.05; // default: 0.01

    auto perceptron = SupervisedNeuralNetwork(activationFunction, inputNodes, outputNodes, hiddenLayers, hiddenLayerNeurons);

    auto trainingData = std::ifstream("./training-images.idx3-ubyte", std::ios::binary);
    auto trainingLabels = std::ifstream("./training-labels.idx1-ubyte", std::ios::binary);
    auto testData = std::ifstream("./test-images.idx3-ubyte", std::ios::binary);
    auto testLabels = std::ifstream("./test-labels.idx1-ubyte", std::ios::binary);

    int32_t dataMagic, dataCount, dataRows, dataColumns;
    read<int32_t>(&dataMagic, trainingData);
    read<int32_t>(&dataCount, trainingData);
    read<int32_t>(&dataRows, trainingData);
    read<int32_t>(&dataColumns, trainingData);
    std::cout << std::format("Training data magic number: {}\n", dataMagic);
    std::cout << std::format("Training with {} images of {}x{} resolution\n", dataCount, dataRows, dataColumns);

    int32_t labelMagic, labelCount;
    read<int32_t>(&labelMagic, trainingLabels);
    read<int32_t>(&labelCount, trainingLabels);
    std::cout << std::format("Training labels magic number: {}\n", labelMagic);
    std::cout << std::format("Label count: {}\n", labelCount);

    std::vector<DataItem> dataItems(dataCount);
    for (auto& dataItem : dataItems) {
        for (int i = 0; i < (dataRows * dataColumns); i++) { // for each pixel
            read<uint8_t>(&dataItem.pixels[i], trainingData);

            /*if (nodeValue > 0.8) std::cout << "X";
            else if (nodeValue > 0.2) std::cout << "x";
            else std::cout << "_";
            if (i % dataRows == 0) std::cout << "\n";*/
        }
        //std::cout << "\n\n";

        read<uint8_t>(&dataItem.label, trainingLabels);
    }

    std::cout << std::format("Finished reading {} training data items into memory\n", dataItems.size());

    // Only weak people stay DRY
    read<int32_t>(&dataMagic, testData);
    read<int32_t>(&dataCount, testData);
    read<int32_t>(&dataRows, testData);
    read<int32_t>(&dataColumns, testData);
    std::cout << std::format("Test data magic number: {}\n", dataMagic);
    std::cout << std::format("Testing with {} images of {}x{} resolution\n", dataCount, dataRows, dataColumns);

    read<int32_t>(&labelMagic, testLabels);
    read<int32_t>(&labelCount, testLabels);
    std::cout << std::format("Test labels magic number: {}\n", labelMagic);
    std::cout << std::format("Test label count: {}\n", labelCount);

    std::vector<DataItem> testItems(dataCount);
    for (auto& dataItem : testItems) {
        for (int i = 0; i < (dataRows * dataColumns); i++) {
            read<uint8_t>(&dataItem.pixels[i], testData);
        }
        read<uint8_t>(&dataItem.label, testLabels);
    }

    std::cout << "Begin training\n";
    int32_t epoch = 1, done = 0, correctGuesses = 0;
    while (true) {
        for (auto& dataItem : dataItems) {
            for (int i = 0; i < dataItem.pixels.size(); i++) {
                double value = dataItem.pixels[i] / 255.0; // [0, 1]
                perceptron.setInputNode(i, value);
            }
            perceptron.calculateOutput();
            perceptron.backpropagate(dataItem.label);

            if (perceptron.getHighestOutputNode() == dataItem.label) correctGuesses++;

            if (++done == batchSize) {
                done = 0;
                perceptron.update(batchSize, learningRate);
            }
        }

        int32_t testsPassed = 0;
        for (auto& dataItem : testItems) { // DRY is a weakness, vol. 2
            for (int i = 0; i < dataItem.pixels.size(); i++) {
                double value = dataItem.pixels[i] / 255.0;
                perceptron.setInputNode(i, value);
            }
            perceptron.calculateOutput();
            if (perceptron.getHighestOutputNode() == dataItem.label) testsPassed++;
        }

        double dataPassRate = std::ceil((correctGuesses / (double)dataItems.size()) * 10000.0) / 100.0;
        double testPassRate = std::ceil((testsPassed / (double)testItems.size()) * 10000.0) / 100.0;

        std::cout << std::format("Epoch {} ({}% correct, {}% tests passed)\n", epoch, dataPassRate, testPassRate);
        correctGuesses = 0;
        epoch++;
    }
}
