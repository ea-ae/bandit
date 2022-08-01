#include "perceptron.h"
#include <iostream>
#include <fstream>
#include <format>
#include <algorithm>
#include <array>
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

struct DataItem {
    std::array<uint8_t, 784> pixels;
    uint8_t label;
};

void perceptron() {
    std::cout << "Creating (kind of a) perceptron\n";

    const auto activationFunction = Relu();
    const auto inputNodes = 784;
    const auto outputNodes = 10;

    auto perceptron = NeuralNetwork(activationFunction, inputNodes, outputNodes, 0, 0);

    auto trainingData = std::ifstream("./training-images.idx3-ubyte", std::ios::binary);
    auto trainingLabels = std::ifstream("./training-labels.idx1-ubyte", std::ios::binary);

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

    std::vector<DataItem> dataItems(dataCount); // temp
    int barrier = 10;
    for (auto dataItem : dataItems) {
        for (int i = 0; i < (dataRows * dataColumns); i++) { // for each pixel
            read<uint8_t>(&dataItem.pixels[i], trainingData);
            perceptron.setInputNode(i, dataItem.pixels[i]);
        }
        read<uint8_t>(&dataItem.label, trainingLabels);

        perceptron.calculateOutput();
        std::cout << std::format("Answer {} for label {}\n", perceptron.getHighestOutputNode(), dataItem.label);

        if (--barrier == 0) break; // exit early, temp
    }

    std::cout << std::format("Read {} data items into memory\n", dataItems.size());


}
