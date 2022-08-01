#include "SupervisedNeuralNetwork.h"
#include <cmath>

int32_t SupervisedNeuralNetwork::getHighestOutputNode() {
    int32_t highestNodeId = 0;
    double highestNodeValue = outputLayer->neurons[0]->value;
    for (int i = 0; i < outputLayer->layerSize; i++) {
        auto value = outputLayer->neurons[i]->value;
        if (value > highestNodeValue) {
            highestNodeId = i;
            highestNodeValue = value;
        }
    }
    return highestNodeId;
}

double SupervisedNeuralNetwork::calculateCost(int32_t label) {
    double totalCost = 0.0;
    for (int i = 0; i < outputLayer->layerSize; i++) {
        auto value = outputLayer->neurons[i]->value;
        auto expected = i == label ? 1 : 0;
        auto cost = std::pow(value - expected, 2);
        totalCost += cost;
    }
    return totalCost;
}

void SupervisedNeuralNetwork::backpropagate(int32_t label) {
    for (auto& neuron : outputLayer->neurons) {
        neuron->backpropagate(*outputLayer->previousLayer, label); // todo: move this stuff into Layer??
    }
}
