#pragma once
#pragma warning(push, 0)
#include <eigen/Eigen/Dense>
#pragma warning( pop )
#include <vector>
#include <stdint.h>
#include "../NeuralNetworks/Neuron.h"

using FixThis = Eigen::Array<float, 1, 128>;

class Layer;
class Neuron;
class ActivationFunction;

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
