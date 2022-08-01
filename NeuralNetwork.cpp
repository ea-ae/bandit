#include "NeuralNetwork.h"

NeuralNetwork::NeuralNetwork(const ActivationFunction& activationFunction,
    int32_t inputs, int32_t outputs, int32_t hiddenLayers, int32_t hiddenLayerNeurons)
{
    inputLayer = std::make_unique<InputLayer>(activationFunction, inputs, nullptr, nullptr);
    outputLayer = std::make_unique<OutputLayer>(activationFunction, outputs, inputLayer.get(), nullptr);
    outputLayer->previousLayer = inputLayer.get();
}
