#include "Sigmoid.h"

#include <cmath>
#include <random>

float Sigmoid::map(float input) const {
    return 1 / (1 + std::exp(-input));
}

float Sigmoid::getPreValueDerivative(float input) const {
    return input * (1 - input);
}

float Sigmoid::generateRandomWeight(size_t connectionsIn) const {  // Xavier initialization
    const float r = 1 / std::sqrtf(static_cast<float>(connectionsIn));

    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<float> generate(-r, r);

    return generate(mt);  // returns a random real number in [-r, r] range
}
