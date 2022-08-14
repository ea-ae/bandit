#include "Relu.h"

#include <algorithm>
#include <cmath>
#include <random>
#include <utility>

float Relu::map(float input) const {
    return std::max(input, 0.0f);
}

float Relu::getPreValueDerivative(float input) const {
    return input > 0 ? 1.0f : 0.0f;
}

float Relu::generateRandomWeight(size_t connectionsIn) const {  // He initialization
    const float r = std::sqrt(2.0f / connectionsIn);

    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<float> generate(-r, r);

    return generate(mt);  // returns a random real number in [-r, r] range
}
