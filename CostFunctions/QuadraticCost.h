#pragma once
#include "CostFunction.h"

class QuadraticCost : public CostFunction {
public:
    double getCost(Layer& outputLayer, std::vector<int32_t> expected) const;
};
