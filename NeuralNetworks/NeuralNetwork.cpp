#include "NeuralNetwork.h"

NeuralNetwork::NeuralNetwork(const ActivationFunction& activationFunction, const CostFunction& costFunction,
    int32_t inputs, int32_t outputs, int32_t hiddenLayerCount, int32_t hiddenLayerNeurons)
    : costFunction(costFunction)
{
    inputLayer = std::make_unique<Layer>(inputs, nullptr);
    auto lastLayer = inputLayer.get();

    for (int i = 0; i < hiddenLayerCount; i++) {
        hiddenLayers.push_back(std::make_unique<Layer>(hiddenLayerNeurons, lastLayer));
        lastLayer->nextLayer = hiddenLayers.back().get(); // set lastLayer's nextLayer field
        lastLayer = hiddenLayers.back().get(); // set current layer as the new lastLayer
    }

    outputLayer = std::make_unique<Layer>(outputs, lastLayer);
    lastLayer->nextLayer = outputLayer.get();

    for (auto& hiddenLayer : hiddenLayers) hiddenLayer->initializeNodeValues(activationFunction, costFunction);
    inputLayer->initializeNodeValues(activationFunction, costFunction);
    outputLayer->initializeNodeValues(activationFunction, costFunction);
}

void NeuralNetwork::calculateOutput() {
    inputLayer->calculateNodeValues();
}

void NeuralNetwork::setInputNode(int32_t inputNode, double value) {
    inputLayer->neurons[inputNode]->value = value;
}
