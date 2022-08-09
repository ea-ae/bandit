#include "CostFunction.h"
#include <cmath>
#include <numeric>

CostFunction::CostFunction(float regularizationLambda, float momentumCoefficientMu) 
    : regularizationLambda(regularizationLambda), momentumCoefficientMu(momentumCoefficientMu) { }

//BatchArray CostFunction::getActivationDerivatives(BatchArray& activations, BatchArray& expected) const {
//    return activations - expected;
//}

FixThis CostFunction::getActivationDerivatives(const FixThis& activations, const FixThis& expected) const {
    return activations - expected;
}

float CostFunction::getRegularizationCost(std::vector<float> weights, int32_t batchSize) const {
    float sum = std::accumulate(weights.begin(), weights.end(), 0.0f, [](float s, float w) {
        return s + std::powf(w, 2);
    });
    return (regularizationLambda * sum) / (2 * batchSize);
}

float CostFunction::getRegularizationDerivative(float weight, int32_t batchSize) const {
    return (regularizationLambda * weight) / batchSize;
}

float CostFunction::getMomentum(float previousMomentum, float weightGradient) const {
    return momentumCoefficientMu * previousMomentum + weightGradient;
}
