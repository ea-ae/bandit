#include "CostFunction.h"
#include <cmath>
#include <numeric>

CostFunction::CostFunction(double regularizationLambda) : regularizationLambda(regularizationLambda) {}

double CostFunction::getActivationDerivative(double activation, double expected) const {
    return activation - expected;
}

double CostFunction::getRegularizationCost(std::vector<double> weights) const {
    double sum = std::accumulate(weights.begin(), weights.end(), 0.0, [](double s, double w) {
        return s + std::pow(w, 2);
    });
    return (regularizationLambda * sum) / (2 * totalWeightCount);
}

double CostFunction::getRegularizationDerivative(double weight) const {
    return (regularizationLambda * weight) / totalWeightCount;
}
