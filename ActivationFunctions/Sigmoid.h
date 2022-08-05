#pragma once
#include "ActivationFunction.h"
#include <stdint.h>

class Sigmoid : public ActivationFunction {
    float map(float input) const;
    float getPreValueDerivative(float input) const;
    float generateRandomWeight(int32_t connectionsIn, int32_t) const;
};
