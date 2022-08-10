#pragma once
#include <vector>
#include <memory>
#include "../Neuron.h"
#include "../../ActivationFunctions/ActivationFunction.h"
#include "../../CostFunctions/CostFunction.h"

class Neuron;
class CostFunction;

class Layer {
public:
    const int32_t layerSize;
    std::vector<std::unique_ptr<Neuron>> neurons;

    Layer* previousLayer = nullptr;
    Layer* nextLayer = nullptr;
public:
    Layer(int32_t neuronCount);
    virtual void connectPreviousLayer(const ActivationFunction& activation, const CostFunction& cost) = 0;
    void calculateNodeValues();
    void update(float learningRate);
    size_t getWeightCount();
};
