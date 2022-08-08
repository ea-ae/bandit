#pragma once
#include "CostFunction.h"

class QuadraticCost : public CostFunction {
public:
    using CostFunction::CostFunction;
    // float getCost(Layer& outputLayer, std::vector<int32_t> expected) const;
};
