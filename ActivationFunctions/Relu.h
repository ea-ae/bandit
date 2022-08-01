#pragma once
#include <utility>
#include "ActivationFunction.h"

class Relu : public ActivationFunction {
    double map(double input) {
        return std::max(input, 0.0);
    }
};

