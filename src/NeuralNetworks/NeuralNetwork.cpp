#include "NeuralNetwork.h"
#include <algorithm>
#include "Neurons/DenseNeuron.h"

NeuralNetwork::NeuralNetwork(int32_t inputs, int32_t outputs) {
    inputLayer = std::make_unique<DenseLayer>(inputs);
    outputLayer = std::make_unique<DenseLayer>(outputs);
}

Layer* NeuralNetwork::addLayer(Layer* layer) {
    auto previousLayer = hiddenLayers.empty() ? inputLayer.get() : hiddenLayers.back().get();
    layer->previousLayer = previousLayer;
    hiddenLayers.push_back(std::unique_ptr<Layer>(layer));
    previousLayer->nextLayer = hiddenLayers.back().get();
    
    return layer;
}

void NeuralNetwork::buildLayers(const ActivationFunction& activation, const CostFunction& cost) {
    // link the last layer
    auto lastLayer = hiddenLayers.back().get();
    lastLayer->nextLayer = outputLayer.get();
    outputLayer->previousLayer = lastLayer;

    // initialize neurons of input layer
    std::generate(inputLayer->getNeurons().begin(), inputLayer->getNeurons().end(), [&]() {
        return std::make_unique<DenseNeuron>(nullptr, activation, cost); // input layer has no inputNeurons
    });

    // initialize neurons inside the layers (each layer sets its inputNeurons from previous layer)
    for (auto& hiddenLayer : hiddenLayers) hiddenLayer->connectPreviousLayer(activation, cost);
    outputLayer->connectPreviousLayer(activation, cost);
}

void NeuralNetwork::setInputNode(int32_t inputNode, int32_t nthBatchItem, float value) {
    inputLayer->getNeurons()[inputNode]->values[nthBatchItem] = value;
}

void NeuralNetwork::calculateOutput() {
    inputLayer->calculateNodeValues();
}

void NeuralNetwork::backpropagate(BatchLabelArray& labels) {
    auto expectedValues = BatchArray(BatchArray::Zero());
    for (int32_t i = 0; i < outputLayer->getNeurons().size(); i++) {
        expectedValues = labels.unaryExpr([this, &i](int32_t label) {
            return getExpectedValue(label, i);
        });
        outputLayer->getNeurons()[i]->backpropagate(true, &expectedValues);
    }

    for (auto& layer : hiddenLayers) {
        for (int32_t i = 0; i < layer->getNeurons().size(); i++) {
            bool backpropagateGradients = layer->previousLayer->previousLayer != nullptr; // input layer?
            layer->getNeurons()[i]->backpropagate(backpropagateGradients);
        }
    }
}

void NeuralNetwork::update(float learningRate) {
    outputLayer->update(learningRate);
}
