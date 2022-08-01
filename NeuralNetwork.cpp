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

int32_t NeuralNetwork::getHighestOutputNode() {
    int32_t highestNodeId = 0;
    double highestNodeValue = outputLayer->neurons[0]->value;
    for (int i = 0; i < outputLayer->layerSize; i++) {
        auto value = outputLayer->neurons[i]->value;
        if (value > highestNodeValue) {
            highestNodeId = i;
            highestNodeValue = value;
        }
    }
    return highestNodeId;
}

void NeuralNetwork::setInputNode(int32_t inputNode, double value) {
    inputLayer->neurons[inputNode]->value = value;
}
