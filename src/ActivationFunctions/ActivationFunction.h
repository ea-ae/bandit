#pragma once
#include <stdint.h>

class ActivationFunction {
   public:
    virtual ~ActivationFunction() = default;
    virtual float map(float input) const = 0;
    virtual float getPreValueDerivative(float input) const = 0;
    virtual float generateRandomWeight(size_t connectionsIn) const = 0;
};
