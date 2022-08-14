#pragma once
#include <stdint.h>

#include "ActivationFunction.h"

class Sigmoid : public ActivationFunction {
    float map(float input) const;
    float getPreValueDerivative(float input) const;
    float generateRandomWeight(size_t connectionsIn) const;
};
