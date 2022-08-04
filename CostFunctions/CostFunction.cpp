#include "CostFunction.h"

double CostFunction::getActivationDerivative(double activation, double expected) const {
    return activation - expected;
}
