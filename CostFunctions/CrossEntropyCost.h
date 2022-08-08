#pragma once
#include "CostFunction.h"

class CrossEntropyCost : public CostFunction { // assumes softmax/sigmoid combination
public:
    using CostFunction::CostFunction;
    // float getCost(Layer& outputLayer, std::vector<int32_t> expected) const;
};
