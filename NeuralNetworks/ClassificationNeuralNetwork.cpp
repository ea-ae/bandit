#include "ClassificationNeuralNetwork.h"
#include <algorithm>
#include <cmath>
#include <vector>

int32_t ClassificationNeuralNetwork::getHighestOutputNode() {
    int32_t highestNodeId = 0;
    float highestNodeValue = outputLayer->neurons[0]->value;
    for (int i = 0; i < outputLayer->layerSize; i++) {
        auto value = outputLayer->neurons[i]->value;
        if (value > highestNodeValue) {
            highestNodeId = i;
            highestNodeValue = value;
        }
    }
    return highestNodeId;
}

float ClassificationNeuralNetwork::calculateCost(int32_t label) {
    std::vector<int32_t> expected(outputLayer->layerSize);
    std::generate(expected.begin(), expected.end(), [label, i = 0]() mutable {
        return label == i++ ? 1 : 0;
    });
    return costFunction.getCost(*outputLayer.get(), expected);
}

float ClassificationNeuralNetwork::getExpectedValue(int32_t label, int32_t neuronIndex) {
    return neuronIndex == label ? 1.0f : 0.0f;
}
