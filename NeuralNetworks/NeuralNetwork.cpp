#include "NeuralNetwork.h"

NeuralNetwork::NeuralNetwork(const ActivationFunction& activationFunction, CostFunction& costFunction,
    int32_t inputs, int32_t outputs, std::vector<int32_t> hiddenLayerNeurons)
    : costFunction(costFunction)
{
    inputLayer = std::make_unique<Layer>(inputs, nullptr);
    auto lastLayer = inputLayer.get();

    // create layers and assign the lastLayer/nextLayer members for each layer
    for (auto& neuronCount : hiddenLayerNeurons) {
        hiddenLayers.push_back(std::make_unique<Layer>(neuronCount, lastLayer));
        lastLayer->nextLayer = hiddenLayers.back().get(); // set lastLayer's nextLayer field
        lastLayer = hiddenLayers.back().get(); // set current layer as the new lastLayer
    }
    outputLayer = std::make_unique<Layer>(outputs, lastLayer);
    lastLayer->nextLayer = outputLayer.get();

    // initialize neurons inside the layers
    for (auto& hiddenLayer : hiddenLayers) hiddenLayer->initializeNodeValues(activationFunction, costFunction);
    inputLayer->initializeNodeValues(activationFunction, costFunction);
    outputLayer->initializeNodeValues(activationFunction, costFunction);
}

void NeuralNetwork::setInputNode(int32_t inputNode, float value) {
    inputLayer->neurons[inputNode]->value = value;
}

void NeuralNetwork::calculateOutput() {
    inputLayer->calculateNodeValues();
}

void NeuralNetwork::backpropagate(int32_t label) {
    for (int32_t i = 0; i < outputLayer->layerSize; i++) {
        outputLayer->neurons[i]->backpropagate(*outputLayer->previousLayer, getExpectedValue(label, i));
    }

    for (auto& layer : hiddenLayers) {
        for (int32_t i = 0; i < layer->layerSize; i++) {
            layer->neurons[i]->backpropagate(*layer->previousLayer);
        }
    }
}

void NeuralNetwork::update(int32_t batchSize, float learningRate) {
    outputLayer->update(batchSize, learningRate);
}
