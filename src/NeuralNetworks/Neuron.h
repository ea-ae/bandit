#pragma once
#pragma warning(push, 0)
#include <eigen/Eigen/Dense>
#pragma warning( pop )
#include <vector>
#include <utility>
#include <optional>
#include "../bandit.h"
#include "Layer.h"
#include "../ActivationFunctions/ActivationFunction.h"
#include "../CostFunctions/CostFunction.h"

class Layer;

struct Weight {
    float weight = 0;
    float gradient = 0;
};

class Neuron {
public:
    BatchArray values = BatchArray(BatchArray::Zero());
private:
    float momentum = 0.0f;
    float bias = 0.0f;
    float biasGradient = 0.0f;

    std::vector<std::unique_ptr<Neuron>>* inputNeurons;
    std::vector<Weight> weights;
    BatchArray activationGradients = BatchArray(BatchArray::Zero());
    BatchArray preTransformedValues;
    
    const ActivationFunction& activationFunction;
    const CostFunction& costFunction;
public:
    Neuron(std::vector<std::unique_ptr<Neuron>>* inputNeurons, const ActivationFunction& activation, const CostFunction& cost);
    void calculate();
    void addActivationGradients(const BatchArray& gradients);
    void backpropagate(bool backpropagateGradients, BatchArray* expectedValues = nullptr);
    void update(int32_t batchSize, float learningRate);
    size_t getWeightCount();
};
