#include "Relu.h"
#include <utility>
#include <cmath>
#include <random>

double Relu::map(double input) const
{
    return std::max(input, 0.0);
}

double Relu::getPreValueDerivative(double input) const {
    return input > 0 ? 1 : 0;
}

double Relu::generateRandomWeight(int32_t connectionsIn, int32_t connectionsOut) const {
    const double r = std::sqrt(2.0 / (connectionsIn + connectionsOut));

    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<double> generate(-r, r); 

    return generate(mt); // returns a random real number in [-r, r] range
}
