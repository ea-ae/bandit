#pragma once
#include "CostFunction.h"

class QuadraticCost : public CostFunction {
public:
    using CostFunction::CostFunction;
    double getCost(Layer& outputLayer, std::vector<int32_t> expected) const;
};
