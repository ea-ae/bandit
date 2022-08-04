#pragma once
#include <vector>
#include <utility>
#include <optional>
#include "Layer.h"
#include "../ActivationFunctions/ActivationFunction.h"

class Layer;
class CostFunction;

struct Weight {
    double weight = 0;
    double gradient = 0;
};

class Neuron {
public:
    double value = 0.0;
private:
    double bias = 0.0;
    double biasGradient = 0.0;

    std::vector<Weight> weights;
    double activationGradient = 0.0;
    double preTransformedValue = 0.0;
    
    const ActivationFunction& activationFunction;
    const CostFunction& costFunction;
public:
    Neuron(const ActivationFunction& activationFunction, const CostFunction& costFunction, const Layer* previousLayer, const Layer* nextLayer);
    void calculate(const Layer& previousLayer);
    void addActivationGradient(double gradient);
    void backpropagate(Layer& previousLayer, std::optional<double> expectedValue = std::nullopt);
    void update(double batchSize, double learningRate);
};
