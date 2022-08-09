#pragma once
#include "ActivationFunction.h"
#include <stdint.h>

class Sigmoid : public ActivationFunction {
    float map(float input) const;
    float getPreValueDerivative(float input) const;
    float generateRandomWeight(size_t connectionsIn) const;
};
