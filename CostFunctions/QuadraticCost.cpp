#include "QuadraticCost.h"
#include <cmath>

double QuadraticCost::getCost(Layer& outputLayer, std::vector<int32_t> expected) const { // todo int->double
    double totalCost = 0.0;
    for (int i = 0; i < outputLayer.layerSize; i++) {
        auto value = outputLayer.neurons[i]->value;
        auto cost = std::pow(value - expected[i], 2);
        totalCost += cost;
    }
    return totalCost;
}

double QuadraticCost::getActivationDerivative(double activation, double expected) const {
    return 2 * (activation - expected);
}
