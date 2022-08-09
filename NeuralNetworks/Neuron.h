#pragma once
#pragma warning(push, 0)
#include <eigen/Eigen/Dense>
#pragma warning( pop )
#include <vector>
#include <utility>
#include <optional>
#include "Layer.h"
#include "../ActivationFunctions/ActivationFunction.h"
#include "../CostFunctions/CostFunction.h"

const auto BATCH_SIZE = 16;

using BatchArray = Eigen::Array<float, 1, BATCH_SIZE>;
using BatchLabelArray = Eigen::Array<int16_t, 1, BATCH_SIZE>;

class Layer;
class CostFunction;

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

    std::vector<Weight> weights;
    BatchArray activationGradients = BatchArray(BatchArray::Zero());
    BatchArray preTransformedValues;
    
    const ActivationFunction& activationFunction;
    const CostFunction& costFunction;
public:
    Neuron(const ActivationFunction& activationFunction, const CostFunction& costFunction, 
        const Layer* previousLayer, const Layer* nextLayer);
    void calculate(const Layer& previousLayer);
    void addActivationGradients(const BatchArray& gradients);
    void backpropagate(Layer& previousLayer, BatchArray* expectedValues = nullptr);
    void update(int32_t batchSize, float learningRate);
    size_t getWeightCount();
};
