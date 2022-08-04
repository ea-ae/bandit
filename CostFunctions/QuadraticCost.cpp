#include "QuadraticCost.h"
#include <cmath>

double QuadraticCost::getCost(Layer& outputLayer, std::vector<int32_t> expected) const { // todo int->double
    double totalCost = 0.0;
    for (int i = 0; i < outputLayer.layerSize; i++) {
        auto a = outputLayer.neurons[i]->value;
        auto y = expected[i];
        auto cost = std::pow(a - y, 2) / 2;
        totalCost += cost;
    }
    return totalCost;
}
