#include "ClassificationNeuralNetwork.h"

#include <algorithm>
#include <cmath>
#include <vector>

int32_t ClassificationNeuralNetwork::getHighestOutputNode(int32_t nthBatchItem) {
    int32_t highestNodeId = 0;
    float highestNodeValue = outputLayer->getNeurons()[0]->values[nthBatchItem];
    for (int i = 0; i < outputLayer->getNeurons().size(); i++) {
        auto value = outputLayer->getNeurons()[i]->values[nthBatchItem];
        if (value > highestNodeValue) {
            highestNodeId = i;
            highestNodeValue = value;
        }
    }
    return highestNodeId;
}

float ClassificationNeuralNetwork::calculateCost(int32_t label) {  // add cost param here
    std::vector<int32_t> expected(outputLayer->getNeurons().size());
    std::generate(expected.begin(), expected.end(), [label, i = 0]() mutable {
        return label == i++ ? 1 : 0;
    });
    // return costFunction.getCost(*outputLayer.get(), expected);
    return -1;  // todo temp until we fix costs with vectorization
}

float ClassificationNeuralNetwork::getExpectedValue(int32_t label, int32_t neuronIndex) {
    return neuronIndex == label ? 1.0f : 0.0f;
}
