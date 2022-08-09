#pragma once
#include <vector>
#include <stdint.h>
#include "../NeuralNetworks/Neuron.h"

class Layer;

class CostFunction {
public:
    CostFunction(float regularizationLambda = 0, float momentumCoefficientMu = 0);
    BatchArray getActivationDerivatives(BatchArray& activations, BatchArray& expected) const;
    float getRegularizationCost(std::vector<float> weights, int32_t batchSize) const;
    float getRegularizationDerivative(float weight, int32_t batchSize) const;
    float getMomentum(float previousMomentum, float weightGradient) const;
protected:
    float regularizationLambda = 0;
    float momentumCoefficientMu = 0;
};
