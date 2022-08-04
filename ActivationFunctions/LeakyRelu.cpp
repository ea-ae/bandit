#include "LeakyRelu.h"
#include <cmath>

LeakyRelu::LeakyRelu() : LeakyRelu(0.1) {}

LeakyRelu::LeakyRelu(double leakRate) : leakRate(leakRate) {}

double LeakyRelu::map(double input) const {
    return input > 0 ? input : input * leakRate;
}

double LeakyRelu::getPreValueDerivative(double input) const {
    return input > 0 ? 1 : leakRate;
}
