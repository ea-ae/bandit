#include "NeuralNetwork.h"

NeuralNetwork::NeuralNetwork(const ActivationFunction& activationFunction, CostFunction& costFunction,
    int32_t inputs, int32_t outputs, int32_t hiddenLayerCount, int32_t hiddenLayerNeurons)
    : costFunction(costFunction)
{
    inputLayer = std::make_unique<Layer>(inputs, nullptr);
    auto lastLayer = inputLayer.get();

    // create layers and assign the lastLayer/nextLayer members for each layer
    for (int i = 0; i < hiddenLayerCount; i++) {
        hiddenLayers.push_back(std::make_unique<Layer>(hiddenLayerNeurons, lastLayer));
        lastLayer->nextLayer = hiddenLayers.back().get(); // set lastLayer's nextLayer field
        lastLayer = hiddenLayers.back().get(); // set current layer as the new lastLayer
    }
    outputLayer = std::make_unique<Layer>(outputs, lastLayer);
    lastLayer->nextLayer = outputLayer.get();

    // initialize neurons inside the layers
    for (auto& hiddenLayer : hiddenLayers) hiddenLayer->initializeNodeValues(activationFunction, costFunction);
    inputLayer->initializeNodeValues(activationFunction, costFunction);
    outputLayer->initializeNodeValues(activationFunction, costFunction);

    // calculate total amount of weights in network for the regularization term
    size_t totalNetworkWeights = 0;
    for (auto& layer : hiddenLayers) {
        totalNetworkWeights += layer->getWeightCount();
    }
    totalNetworkWeights += outputLayer->getWeightCount();
    costFunction.totalWeightCount = totalNetworkWeights;
}

void NeuralNetwork::calculateOutput() {
    inputLayer->calculateNodeValues();
}

void NeuralNetwork::setInputNode(int32_t inputNode, double value) {
    inputLayer->neurons[inputNode]->value = value;
}
