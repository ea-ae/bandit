#include "perceptron.h"
#include <iostream>
#include <fstream>
#include <format>
#include <algorithm>
#include "NeuralNetwork.h"
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

void perceptron() {
    std::cout << "Creating perceptron\n";

    const auto activationFunction = Relu();
    const auto inputNodes = 3;
    const auto outputNodes = 3;

    auto perceptron = NeuralNetwork(activationFunction, inputNodes, outputNodes, 0, 0);

    auto trainingData = std::ifstream("./training-images.idx3-ubyte", std::ios::binary);
    auto trainingLabels = std::ifstream("./training-labels.idx1-ubyte", std::ios::binary);

    int32_t magic;
    read<int32_t>(&magic, trainingData);
    std::cout << std::format("Training data magic number: {}\n", magic);

    read<int32_t>(&magic, trainingLabels);
    std::cout << std::format("Training labels magic number: {}\n", magic);

    int32_t labelCount;
    read<int32_t>(&labelCount, trainingLabels);
    std::cout << std::format("Label count: {}\n", labelCount);
}