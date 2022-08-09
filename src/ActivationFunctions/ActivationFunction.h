#pragma once
#include <stdint.h>

class ActivationFunction {
public:
    virtual float map(float input) const = 0;
    virtual float getPreValueDerivative(float input) const = 0;
    virtual float generateRandomWeight(int32_t connectionsIn) const = 0;
};
