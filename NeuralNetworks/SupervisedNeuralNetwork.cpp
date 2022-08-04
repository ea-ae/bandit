#include "SupervisedNeuralNetwork.h"
#include <algorithm>
#include <cmath>
#include <vector>

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
    std::vector<int32_t> expected(outputLayer->layerSize);
    std::generate(expected.begin(), expected.end(), [label, i = 0]() mutable {
        return label == i++ ? 1 : 0;
    });
    return costFunction.getCost(*outputLayer.get(), expected);
}

void SupervisedNeuralNetwork::backpropagate(int32_t label) {
    for (int i = 0; i < outputLayer->layerSize; i++) {
        outputLayer->neurons[i]->backpropagate(*outputLayer->previousLayer, i == label ? 1 : 0);
    }

    for (auto& layer : hiddenLayers) {
        for (int i = 0; i < layer->layerSize; i++) {
            layer->neurons[i]->backpropagate(*layer->previousLayer);
        }
    }
}

//void SupervisedNeuralNetwork::backpropagate(int32_t label) {
//    for (int i = 0; i < outputLayer->layerSize; i++) {
//        outputLayer->neurons[i]->backpropagate(*outputLayer->previousLayer, i == label ? 1 : 0);
//    }
//}

void SupervisedNeuralNetwork::update(int32_t batchSize, double learningRate) {
    outputLayer->update(batchSize, learningRate);
}
