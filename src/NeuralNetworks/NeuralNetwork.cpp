#include "NeuralNetwork.h"
#include <algorithm>

NeuralNetwork::NeuralNetwork(int32_t inputs, int32_t outputs) {
    inputLayer = std::make_unique<DenseLayer>(inputs);
    outputLayer = std::make_unique<DenseLayer>(outputs);
}

void NeuralNetwork::addLayer(Layer* layer) {
    auto previousLayer = hiddenLayers.empty() ? inputLayer.get() : hiddenLayers.back().get();
    layer->previousLayer = previousLayer;
    hiddenLayers.push_back(std::unique_ptr<Layer>(layer));
    previousLayer->nextLayer = hiddenLayers.back().get();
}

void NeuralNetwork::buildLayers(const ActivationFunction& activation, const CostFunction& cost) {
    // link the last layer
    auto lastLayer = hiddenLayers.back().get();
    lastLayer->nextLayer = outputLayer.get();
    outputLayer->previousLayer = lastLayer;

    // initialize neurons of input layer
    std::generate(inputLayer->neurons.begin(), inputLayer->neurons.end(), [&]() {
        return std::make_unique<Neuron>(nullptr, activation, cost); // input layer has no inputNeurons
    });

    // initialize neurons inside the layers (each layer sets the inputNeurons of the next layer)
    inputLayer->connectNextLayer(activation, cost);
    for (auto& hiddenLayer : hiddenLayers) hiddenLayer->connectNextLayer(activation, cost);
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
        outputLayer->neurons[i]->backpropagate(true, &expectedValues);
    }

    for (auto& layer : hiddenLayers) {
        for (int32_t i = 0; i < layer->layerSize; i++) {
            bool backpropagateGradients = layer->previousLayer->previousLayer != nullptr; // input layer?
            layer->neurons[i]->backpropagate(backpropagateGradients);
        }
    }
}

void NeuralNetwork::update(float learningRate) {
    outputLayer->update(learningRate);
}
