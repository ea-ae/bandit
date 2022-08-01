#include "NeuralNetwork.h"

NeuralNetwork::NeuralNetwork(const ActivationFunction& activationFunction,
    int32_t inputs, int32_t outputs, int32_t hiddenLayers, int32_t hiddenLayerNeurons)
    :
    activationFunction(activationFunction),
    inputLayer(std::make_unique<InputLayer>()),
    outputLayer(std::make_unique<OutputLayer>()) 
{
    
}
