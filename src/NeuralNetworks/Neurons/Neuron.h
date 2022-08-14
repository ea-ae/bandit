#pragma once
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "../../ActivationFunctions/ActivationFunction.h"
#include "../../CostFunctions/CostFunction.h"
#include "../../bandit.h"

struct Weight {
    float weight = 0.0f;
    float gradient = 0.0f;
    float momentum = 0.0f;
};

struct Bias {
    float bias = 0.0f;
    float gradient = 0.0f;
    float momentum = 0.0f;
    bool done = false;
};

class Neuron {
   public:
    BatchArray values = BatchArray(BatchArray::Zero());

   protected:
    std::shared_ptr<Bias> bias = nullptr;

    std::vector<std::shared_ptr<Weight>> weights;  // useful in case of shared weights
    BatchArray activationGradients = BatchArray(BatchArray::Zero());

    const ActivationFunction& activationFunction;
    const CostFunction& costFunction;

   public:
    virtual ~Neuron() = default;
    void calculate();
    void addActivationGradients(const BatchArray& gradients);
    void backpropagate(bool backpropagateGradients, BatchArray* expectedValues = nullptr);
    void update(float learningRate);
    size_t getWeightCount();

   protected:
    Neuron(const ActivationFunction& activation, const CostFunction& cost);
    virtual Neuron& getInputNeuron(size_t i) = 0;
    virtual size_t getInputNeuronCount() = 0;
};
