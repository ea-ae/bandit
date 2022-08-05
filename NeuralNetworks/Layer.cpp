#include "Layer.h"
#include <algorithm>
#include <numeric>

Layer::Layer(int32_t layerSize, Layer* previousLayer)
    : neurons(layerSize), layerSize(layerSize), previousLayer(previousLayer)
{
    neurons = std::vector<std::unique_ptr<Neuron>>(layerSize);
}

void Layer::initializeNodeValues(const ActivationFunction& activationFunction, const CostFunction& costFunction) {
    std::generate(neurons.begin(), neurons.end(), [&]() {
        return std::make_unique<Neuron>(activationFunction, costFunction, previousLayer, nextLayer);
    });
}

void Layer::calculateNodeValues() {
    if (previousLayer) {
        for (auto& neuron : neurons) {
            neuron->calculate(*previousLayer);
        }
    }
    if (nextLayer) nextLayer->calculateNodeValues();
}

void Layer::update(int32_t batchSize, float learningRate) {
    if (!previousLayer) return; // don't update input nodes

    for (auto& neuron : neurons) neuron->update(batchSize, learningRate);

    previousLayer->update(batchSize, learningRate); // move onto next layer
}

size_t Layer::getWeightCount() {
    return std::accumulate(neurons.begin(), neurons.end(), (size_t)0, [](size_t sum, auto& neuron) {
        return sum + neuron->getWeightCount();
    });
}
