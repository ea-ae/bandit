#pragma once
#include <stdint.h>

class ActivationFunction {
public:
    virtual double map(double input) const = 0;
    virtual double getPreValueDerivative(double input) const = 0;
    virtual double generateRandomWeight(int32_t connectionsIn, int32_t connectionsOut) const = 0;
};
