#pragma once
#include <vector>
#include "../NeuralNetworks/Layer.h"

class CostFunction {
public:
    virtual double calculateCost(Layer& outputLayer, std::vector<int32_t> expected) const = 0;
};
