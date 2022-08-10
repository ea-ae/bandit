#pragma once
#pragma warning(push, 0)
#include <eigen/Eigen/Dense>
#pragma warning( pop )
#include <vector>
#include <utility>
#include <optional>
#include "../bandit.h"
#include "Layers/Layer.h"
#include "../ActivationFunctions/ActivationFunction.h"
#include "../CostFunctions/CostFunction.h"

class Layer;

struct Weight {
    float weight = 0.0f;
    float gradient = 0.0f;
};

struct Bias {
    float bias = 0.0f;
    float gradient = 0.0f;
    bool done = false;
};

class Neuron {
public:
    BatchArray values = BatchArray(BatchArray::Zero());
private:
    Bias selfBias;
    Bias* bias = nullptr;
    float momentum = 0.0f;
    /*float bias = 0.0f;
    float biasGradient = 0.0f;*/

    std::vector<std::shared_ptr<Neuron>>* inputNeurons;
    std::vector<Weight> selfWeights;
    std::vector<Weight>* weights = nullptr; // useful in case of shared weights
    BatchArray activationGradients = BatchArray(BatchArray::Zero());
    BatchArray preTransformedValues;
    
    const ActivationFunction& activationFunction;
    const CostFunction& costFunction;
public:
    Neuron(std::vector<std::shared_ptr<Neuron>>* inputNeurons, const ActivationFunction& activation, const CostFunction& cost);
    Neuron(std::vector<std::shared_ptr<Neuron>>* inputNeurons, const ActivationFunction& activation, const CostFunction& cost,
        std::vector<Weight>* sharedWeights, Bias* sharedBias);
    void calculate();
    void addActivationGradients(const BatchArray& gradients);
    void backpropagate(bool backpropagateGradients, BatchArray* expectedValues = nullptr);
    void update(float learningRate);
    size_t getWeightCount();
};
