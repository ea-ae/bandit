#pragma once
#include <vector>
#include <memory>
#include "../ActivationFunctions/ActivationFunction.h"
#include "../CostFunctions/CostFunction.h"
#include "Layers/Layer.h"
#include "Layers/DenseLayer.h"
#include "Neurons/Neuron.h"

class NeuralNetwork {
protected:
    std::unique_ptr<Layer> inputLayer;
    std::unique_ptr<Layer> outputLayer;
    std::vector<std::unique_ptr<Layer>> hiddenLayers;
public:
    NeuralNetwork(int32_t inputs, int32_t outputs);
    Layer* addLayer(Layer* layer);
    void buildLayers(const ActivationFunction& activationFunction, const CostFunction& costFunction);
    void setInputNode(int32_t inputNode, int32_t nthBatchItem, float value);
    void calculateOutput();
    void backpropagate(BatchLabelArray& labels);
    void update(float learningRate);
private:
    // todo: templates/abstraction, label might not be int32_t
    virtual float getExpectedValue(int32_t label, int32_t neuronIndex) = 0;
};
