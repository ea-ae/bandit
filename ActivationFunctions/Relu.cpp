#include "Relu.h"
#include <utility>
#include <cmath>
#include <random>

double Relu::map(double input) const
{
    return std::max(input, 0.0);
}

double Relu::generateRandomWeight(int32_t connectionsIn, int32_t connectionsOut) const {
    const double r = std::sqrt(12.0 / (connectionsIn + connectionsOut));

    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<double> generate(-r, r); 

    return generate(mt); // eturns random real number in [-r, r] range
}
