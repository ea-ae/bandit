#pragma once
#include "stdint.h"
#include "ActivationFunction.h"

class Relu : public ActivationFunction {
    float map(float input) const;
    float getPreValueDerivative(float input) const;
    float generateRandomWeight(int32_t connectionsIn) const;
};
