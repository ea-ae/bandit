#pragma once
#include <vector>
#include <utility>
#include <optional>
#include "ActivationFunctions/ActivationFunction.h"
#include "Layer.h"

class Layer;

struct Weight {
    double weight;
    double gradient = 0;
};

class Neuron {
public:
    double value = 0.0;
private:
    double bias = 0.0;
    double biasGradient = 0.0;

    std::vector<Weight> weights;
    // int32_t gradientsAdded = 0; // required to find the average later through S/n
    double preTransformedValue = 0.0;
    
    const ActivationFunction& activationFunction;
public:
    Neuron(const ActivationFunction& activationFunction, const Layer* previousLayer, const Layer* nextLayer);
    void calculate(const Layer& previousLayer);
    void backpropagate(Layer& previousLayer, std::optional<double> expectedValue = std::nullopt, double activationDerivative = 1);
    void update(double batchSize, double learningRate);
};
