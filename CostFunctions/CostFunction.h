#pragma once
#include <vector>
#include <stdint.h>
#include "../ActivationFunctions/ActivationFunction.h"
// #include "../NeuralNetworks/Layer.h"
#include <eigen/Eigen/Dense>
#include "../NeuralNetworks/Neuron.h"

// I don't even use the Layer class anymore, but removing it (+ the #include "Layer.h")
// increases the amount of spammy compiler errors by an order of magnitude
//class Layer; 
//class Neuron;

using FixThis = Eigen::Array<float, 1, 128>;

class CostFunction {
public:
    CostFunction(float regularizationLambda = 0, float momentumCoefficientMu = 0);
    // virtual float getCost(Layer& outputLayer, std::vector<int32_t> expected) const = 0;
    // BatchArray getActivationDerivatives(BatchArray& activations, BatchArray& expected) const;
    // FIX THIS v
    FixThis getActivationDerivatives(const FixThis& activations, const FixThis& expected) const;
    // void getActivationDerivatives(BatchArray& activations, BatchArray& expected) const;
    float getRegularizationCost(std::vector<float> weights, int32_t batchSize) const;
    float getRegularizationDerivative(float weight, int32_t batchSize) const;
    float getMomentum(float previousMomentum, float weightGradient) const;
protected:
    float regularizationLambda = 0;
    float momentumCoefficientMu = 0;
};
