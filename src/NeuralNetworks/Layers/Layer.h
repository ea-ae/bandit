#pragma once
#include <vector>
#include <memory>
#include "../Neuron.h"
#include "../../ActivationFunctions/ActivationFunction.h"
#include "../../CostFunctions/CostFunction.h"

class Neuron;
class CostFunction;

struct Size {
    int32_t x;
    int32_t y;
    Size(int32_t x, int32_t y);
};

struct Size3 {
    int32_t x;
    int32_t y;
    int32_t z;
    Size3(int32_t x, int32_t y, int32_t z);
};

class Layer {
public:
    Layer* previousLayer = nullptr;
    Layer* nextLayer = nullptr;
public:
    virtual std::vector<std::unique_ptr<Neuron>>& getNeurons() = 0;
    virtual void connectPreviousLayer(const ActivationFunction& activation, const CostFunction& cost) = 0;
    virtual const Size3 outputSize() const = 0;
    void calculateNodeValues();
    void update(float learningRate);
    size_t getWeightCount();
};
