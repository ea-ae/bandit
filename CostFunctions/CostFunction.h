#pragma once
#include <vector>
#include "../NeuralNetworks/Layer.h"

class Layer;

class CostFunction {
public:
    virtual double getCost(Layer& outputLayer, std::vector<int32_t> expected) const = 0;
    virtual double getActivationDerivative(double activation, double expected) const = 0;
};
