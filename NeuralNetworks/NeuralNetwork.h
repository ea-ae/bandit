#pragma once
#include <vector>
#include <memory>
#include "../ActivationFunctions/ActivationFunction.h"
#include "../CostFunctions/CostFunction.h"
#include "Layer.h"

class NeuralNetwork {
protected:
    CostFunction& costFunction;
    std::unique_ptr<Layer> inputLayer;
    std::unique_ptr<Layer> outputLayer;
    std::vector<std::unique_ptr<Layer>> hiddenLayers;
public:
    NeuralNetwork(const ActivationFunction& activationFunction, CostFunction& costFunction,
        int32_t inputs, int32_t outputs, int32_t hiddenLayerCount, int32_t hiddenLayerNeurons);
    void calculateOutput();
    void setInputNode(int32_t inputNode, double value);
};
