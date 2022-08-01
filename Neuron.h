#pragma once
#include <vector>
#include "ActivationFunctions/ActivationFunction.h"
#include "Layer.h"

class Layer;

class Neuron {
private:
    std::vector<double> weights;
    double bias = 0.0;
    const ActivationFunction& activationFunction;
public:
    Neuron(const ActivationFunction& activationFunction, const Layer* previousLayer, const Layer* nextLayer);
};
