#pragma once
#include "CostFunction.h"
#include <cmath>

class QuadraticCost : public CostFunction {
public:
    double calculateCost(Layer& outputLayer, std::vector<int32_t> expected) const;
};
