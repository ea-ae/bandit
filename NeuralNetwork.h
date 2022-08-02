#pragma once
#include <vector>
#include <memory>
#include "ActivationFunctions/ActivationFunction.h"
#include "InputLayer.h"
#include "OutputLayer.h"
#include "HiddenLayer.h"

class NeuralNetwork {
protected:
    std::unique_ptr<Layer> inputLayer;
    std::unique_ptr<Layer> outputLayer;
    std::vector<std::unique_ptr<Layer>> hiddenLayers;
public:
    NeuralNetwork(const ActivationFunction& activationFunction, 
        int32_t inputs, int32_t outputs, int32_t hiddenLayerCount, int32_t hiddenLayerNeurons);
    void calculateOutput();
    void setInputNode(int32_t inputNode, double value);
};
