#include "NeuralNetwork.h"

NeuralNetwork::NeuralNetwork(int32_t inputs, int32_t outputs) {
    inputLayer = std::make_unique<Layer>(inputs);
    outputLayer = std::make_unique<Layer>(outputs);
}

void NeuralNetwork::addLayer(Layer&& layer) {
    auto previousLayer = hiddenLayers.empty() ? inputLayer.get() : hiddenLayers.back().get();
    layer.previousLayer = previousLayer;
    hiddenLayers.push_back(std::make_unique<Layer>(std::move(layer)));
    previousLayer->nextLayer = hiddenLayers.back().get();

}

void NeuralNetwork::buildLayers(const ActivationFunction& activation, const CostFunction& cost) {
    // link the last layer
    auto lastLayer = hiddenLayers.back().get();
    lastLayer->nextLayer = outputLayer.get();
    outputLayer->previousLayer = lastLayer;

    // initialize neurons inside the layers
    for (auto& hiddenLayer : hiddenLayers) hiddenLayer->initializeNodeValues(activation, cost);
    inputLayer->initializeNodeValues(activation, cost);
    outputLayer->initializeNodeValues(activation, cost);
}

void NeuralNetwork::setInputNode(int32_t inputNode, int32_t nthBatchItem, float value) {
    inputLayer->neurons[inputNode]->values[nthBatchItem] = value;
}

void NeuralNetwork::calculateOutput() {
    inputLayer->calculateNodeValues();
}

void NeuralNetwork::backpropagate(BatchLabelArray& labels) {
    auto expectedValues = BatchArray(BatchArray::Zero());
    for (int32_t i = 0; i < outputLayer->layerSize; i++) {
        expectedValues = labels.unaryExpr([this, &i](int32_t label) {
            return getExpectedValue(label, i);
        });
        outputLayer->neurons[i]->backpropagate(*outputLayer->previousLayer, &expectedValues);
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
