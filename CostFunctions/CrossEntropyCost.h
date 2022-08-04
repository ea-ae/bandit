#pragma once
#include "CostFunction.h"

class CrossEntropyCost : public CostFunction { // assumes softmax/sigmoid combination
    double getCost(Layer& outputLayer, std::vector<int32_t> expected) const;
};
