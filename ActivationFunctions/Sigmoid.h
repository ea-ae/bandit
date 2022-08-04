#pragma once
#include "ActivationFunction.h"
#include <stdint.h>

class Sigmoid : public ActivationFunction {
    double map(double input) const;
    double getPreValueDerivative(double input) const;
    double generateRandomWeight(int32_t connectionsIn, int32_t connectionsOut) const;
};
