#pragma once
#include <vector>
#include <stdint.h>
#include "../ActivationFunctions/ActivationFunction.h"
#include "../NeuralNetworks/Layer.h"

class Layer;

class CostFunction {
public:
    size_t totalWeightCount = 0;
public:
    CostFunction(float regularizationLambda = 0);
    virtual float getCost(Layer& outputLayer, std::vector<int32_t> expected) const = 0;
    virtual float getActivationDerivative(float activation, float expected) const;
    float getRegularizationCost(std::vector<float> weights) const;
    float getRegularizationDerivative(float weight) const;
protected:
    float regularizationLambda = 0;
};
