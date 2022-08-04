#include "QuadraticCost.h"

double QuadraticCost::calculateCost(Layer& outputLayer, std::vector<int32_t> expected) const {
    double totalCost = 0.0;
    for (int i = 0; i < outputLayer.layerSize; i++) {
        auto value = outputLayer.neurons[i]->value;
        auto cost = std::pow(value - expected[i], 2);
        totalCost += cost;
    }
    return totalCost;
}
