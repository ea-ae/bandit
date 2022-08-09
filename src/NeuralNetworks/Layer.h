#pragma once
#include <vector>
#include <memory>
#include "Neuron.h"
#include "../ActivationFunctions/ActivationFunction.h"
#include "../CostFunctions/CostFunction.h"

class Neuron;
class CostFunction;

class Layer {
public:
    const int32_t layerSize;
    std::vector<std::unique_ptr<Neuron>> neurons;

    Layer* previousLayer = nullptr;
    Layer* nextLayer = nullptr;
public:
    Layer(int32_t layerSize);
    void connectNextLayer(const ActivationFunction& activation, const CostFunction& cost);
    void calculateNodeValues();
    void update(int32_t batchSize, float learningRate);
    size_t getWeightCount();
};
