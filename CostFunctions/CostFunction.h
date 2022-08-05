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
    CostFunction(double regularizationLambda = 0);
    virtual double getCost(Layer& outputLayer, std::vector<int32_t> expected) const = 0;
    virtual double getActivationDerivative(double activation, double expected) const;
    double getRegularizationCost(std::vector<double> weights) const;
    double getRegularizationDerivative(double weight) const;
protected:
    double regularizationLambda = 0;
};
