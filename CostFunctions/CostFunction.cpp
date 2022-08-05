#include "CostFunction.h"
#include <cmath>
#include <numeric>

CostFunction::CostFunction(float regularizationLambda) : regularizationLambda(regularizationLambda) {}

float CostFunction::getActivationDerivative(float activation, float expected) const {
    return activation - expected;
}

float CostFunction::getRegularizationCost(std::vector<float> weights) const {
    float sum = std::accumulate(weights.begin(), weights.end(), 0.0f, [](float s, float w) {
        return s + std::pow(w, 2);
    });
    return (regularizationLambda * sum) / (2 * totalWeightCount);
}

float CostFunction::getRegularizationDerivative(float weight) const {
    return (regularizationLambda * weight) / totalWeightCount;
}
