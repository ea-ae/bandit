#include "Sigmoid.h"
#include <cmath>
#include <random>

double Sigmoid::map(double input) const {
    return 1 / (1 + std::exp(-input));
}

double Sigmoid::getPreValueDerivative(double input) const {
    return input * (1 - input);
}

double Sigmoid::generateRandomWeight(int32_t connectionsIn, int32_t connectionsOut) const { // Xavier initialization
    const double r = 1 / std::sqrt(connectionsIn);

    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<double> generate(-r, r);

    return generate(mt); // returns a random real number in [-r, r] range
}
