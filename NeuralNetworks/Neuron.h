#pragma once
#include <vector>
#include <utility>
#include <optional>
#include "Layer.h"
#include "../ActivationFunctions/ActivationFunction.h"

class Layer;
class CostFunction;

struct Weight {
    float weight = 0;
    float gradient = 0;
};

class Neuron {
public:
    float value = 0.0;
private:
    float bias = 0.0;
    float biasGradient = 0.0;

    std::vector<Weight> weights;
    float activationGradient = 0.0;
    float preTransformedValue = 0.0;
    
    const ActivationFunction& activationFunction;
    const CostFunction& costFunction;
public:
    Neuron(const ActivationFunction& activationFunction, const CostFunction& costFunction, const Layer* previousLayer, const Layer* nextLayer);
    void calculate(const Layer& previousLayer);
    void addActivationGradient(float gradient);
    void backpropagate(Layer& previousLayer, std::optional<float> expectedValue = std::nullopt);
    void update(int32_t batchSize, float learningRate);
    size_t getWeightCount();
};
