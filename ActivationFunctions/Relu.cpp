#include "Relu.h"
#include <utility>
#include <cmath>

double Relu::map(double input) const
{
    return std::max(input, 0.0);
}

double Relu::generateRandomWeight(int32_t connectionsIn, int32_t connectionsOut) const {
    return std::sqrt(12 / (connectionsIn + connectionsOut));
}
