#pragma once
#include <vector>
#include <memory>
#include "ActivationFunctions/ActivationFunction.h"
#include "InputLayer.h"
#include "OutputLayer.h"
#include "HiddenLayer.h"

class NeuralNetwork {
protected:
    std::unique_ptr<InputLayer> inputLayer;
    std::unique_ptr<OutputLayer> outputLayer;
    std::vector<std::unique_ptr<HiddenLayer>> hiddenLayers;
public:
    NeuralNetwork(const ActivationFunction& activationFunction, 
        int32_t inputs, int32_t outputs, int32_t hiddenLayers, int32_t hiddenLayerNeurons);
    void calculateOutput();
    void setInputNode(int32_t inputNode, double value);
};
