#pragma once
#include "stdint.h"
#include "ActivationFunction.h"

class Relu : public ActivationFunction {
    double map(double input) const;
    double generateRandomWeight(int32_t connectionsIn, int32_t connectionsOut) const;
};

