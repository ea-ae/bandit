#pragma once
#include <vector>
#include <stdint.h>
#include "../ActivationFunctions/ActivationFunction.h"
#include "../NeuralNetworks/Layer.h"

class Layer;

class CostFunction {
public:
    CostFunction(float regularizationLambda = 0, float momentumCoefficientMu = 0);
    virtual float getCost(Layer& outputLayer, std::vector<int32_t> expected) const = 0;
    virtual float getActivationDerivative(float activation, float expected) const;
    float getRegularizationCost(std::vector<float> weights, int32_t batchSize) const;
    float getRegularizationDerivative(float weight, int32_t batchSize) const;
    float getMomentum(float previousMomentum, float weightGradient) const;
protected:
    float regularizationLambda = 0;
    float momentumCoefficientMu = 0;
};
