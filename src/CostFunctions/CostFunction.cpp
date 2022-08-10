#include "CostFunction.h"
#include <cmath>
#include <numeric>

CostFunction::CostFunction(float regularizationLambda, float momentumCoefficientMu) 
    : regularizationLambda(regularizationLambda), momentumCoefficientMu(momentumCoefficientMu) { }

BatchArray CostFunction::getActivationDerivatives(BatchArray& activations, BatchArray& expected) const {
    return activations - expected;
}

float CostFunction::getRegularizationCost(std::vector<float> weights) const {
    float sum = std::accumulate(weights.begin(), weights.end(), 0.0f, [](float s, float w) {
        return s + std::powf(w, 2);
    });
    return (regularizationLambda * sum) / (2 * BATCH_SIZE);
}

float CostFunction::getRegularizationDerivative(float weight) const {
    return (regularizationLambda * weight) / BATCH_SIZE;
}

float CostFunction::getMomentum(float previousMomentum, float weightGradient) const {
    return momentumCoefficientMu * previousMomentum + weightGradient;
}
