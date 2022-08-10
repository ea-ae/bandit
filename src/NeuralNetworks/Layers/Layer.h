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
    Layer* previousLayer = nullptr;
    Layer* nextLayer = nullptr;
public:
    virtual std::vector<std::shared_ptr<Neuron>>& getNeurons() = 0;
    virtual void connectPreviousLayer(const ActivationFunction& activation, const CostFunction& cost) = 0;
    void calculateNodeValues();
    void update(float learningRate);
    size_t getWeightCount();
};
