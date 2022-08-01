#include "NeuralNetwork.h"

NeuralNetwork::NeuralNetwork(const ActivationFunction& activationFunction,
    int32_t inputs, int32_t outputs, int32_t hiddenLayers, int32_t hiddenLayerNeurons)
{
    inputLayer = std::make_unique<InputLayer>(activationFunction, inputs, nullptr, nullptr);
    outputLayer = std::make_unique<OutputLayer>(activationFunction, outputs, inputLayer.get(), nullptr);

    inputLayer->nextLayer = outputLayer.get();
}

void NeuralNetwork::calculateOutput() {
    inputLayer->calculateNodeValues();
}

void NeuralNetwork::setInputNode(int32_t inputNode, double value) {
    inputLayer->neurons[inputNode]->value = value;
}
