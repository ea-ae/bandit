#pragma once
#include <vector>
#include <memory>
#include "Neuron.h"
#include "ActivationFunctions/ActivationFunction.h"

class Neuron;

class Layer {
public:
    const int32_t layerSize;
    std::vector<std::unique_ptr<Neuron>> neurons;
    Layer& previousLayer;
    Layer& nextLayer;
public:
    Layer(const ActivationFunction& activationFunction, int32_t layerSize, Layer& previousLayer, Layer& nextLayer);
};
