#pragma once
#include <stdint.h>

#include <vector>

#include "../bandit.h"

class CostFunction {
   public:
    CostFunction(float regularizationLambda = 0, float momentumCoefficientMu = 0);
    BatchArray getActivationDerivatives(BatchArray& activations, BatchArray& expected) const;
    float getRegularizationCost(std::vector<float> weights) const;
    float getRegularizationDerivative(float weight) const;
    float getMomentum(float previousMomentum, float weightGradient) const;

   protected:
    float regularizationLambda = 0;
    float momentumCoefficientMu = 0;
};
