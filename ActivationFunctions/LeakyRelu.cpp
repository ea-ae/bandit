#include "LeakyRelu.h"
#include <cmath>

LeakyRelu::LeakyRelu() : LeakyRelu(0.1) {}

LeakyRelu::LeakyRelu(float leakRate) : leakRate(leakRate) {}

float LeakyRelu::map(float input) const {
    return input > 0 ? input : input * leakRate;
}

float LeakyRelu::getPreValueDerivative(float input) const {
    return input > 0.0f ? 1.0f : leakRate;
}
