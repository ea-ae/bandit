#pragma once
#include "stdint.h"
#include <vector>
#include "Layer.h"

class Neuron {
private:
    std::vector<double> weights;
    const ActivationFunction& activationFunction;
public:
    Neuron(const ActivationFunction& activationFunction, Layer* previousLayer, Layer* nextLayer);
};
