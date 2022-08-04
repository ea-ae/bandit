#include "CrossEntropyCost.h"
#include <cmath>

double CrossEntropyCost::getCost(Layer& outputLayer, std::vector<int32_t> expected) const {
    double totalCost = 0.0;
    for (int i = 0; i < outputLayer.layerSize; i++) {
        auto a = outputLayer.neurons[i]->value;
        auto y = expected[i];
        auto cost = y * std::log(a) + (1 - y) * std::log(1 - a);
        totalCost -= cost; // inverse
    }
    return totalCost;
}

double CrossEntropyCost::getActivationDerivative(double activation, double expected) const {
    return activation - expected; // assumes sigmoid/softmax!
}
