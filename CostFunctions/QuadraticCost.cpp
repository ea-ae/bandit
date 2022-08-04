#include "QuadraticCost.h"
#include <cmath>

double QuadraticCost::getCost(Layer& outputLayer, std::vector<int32_t> expected) const { // todo int->double
    double totalCost = 0.0;
    for (int i = 0; i < outputLayer.layerSize; i++) {
        auto a = outputLayer.neurons[i]->value;
        auto y = expected[i];
        auto cost = std::pow(a - y, 2); // todo divide by 2 for cleaner derivation
        totalCost += cost;
    }
    return totalCost;
}

double QuadraticCost::getActivationDerivative(double activation, double expected) const {
    return 2 * (activation - expected);
}
