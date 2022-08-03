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
    double activationGradient = 0.0;
    double preTransformedValue = 0.0;
    
    const ActivationFunction& activationFunction;
public:
    Neuron(const ActivationFunction& activationFunction, const Layer* previousLayer, const Layer* nextLayer);
    void calculate(const Layer& previousLayer);
    void addActivationGradient(double gradient);
    void backpropagate(Layer& previousLayer, std::optional<double> expectedValue = std::nullopt);
    void update(double batchSize, double learningRate);
};
