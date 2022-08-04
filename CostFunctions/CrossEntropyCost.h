#pragma once
#include "CostFunction.h"

class CrossEntropyCost : public CostFunction {
    double getCost(Layer& outputLayer, std::vector<int32_t> expected) const;
    double getActivationDerivative(double activation, double expected) const;
};
