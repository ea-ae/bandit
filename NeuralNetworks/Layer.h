#pragma once
#include <vector>
#include <memory>
#include "Neuron.h"
#include "../ActivationFunctions/ActivationFunction.h"
#include "../CostFunctions/CostFunction.h"

class Neuron;

class Layer {
public:
    const int32_t layerSize;
    std::vector<std::unique_ptr<Neuron>> neurons;

    Layer* previousLayer = nullptr;
    Layer* nextLayer = nullptr;
public:
    Layer(int32_t layerSize, Layer* previousLayer);
    void initializeNodeValues(const ActivationFunction& activationFunction, const CostFunction& costFunction);
    void calculateNodeValues();
    void update(int32_t batchSize, double learningRate);
};
