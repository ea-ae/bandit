#include "Layer.h"
#include <algorithm>
#include <numeric>

Layer::Layer(int32_t layerSize) : neurons(layerSize), layerSize(layerSize)
{
    neurons = std::vector<std::unique_ptr<Neuron>>(layerSize);
}

void Layer::connectNextLayer(const ActivationFunction& activation, const CostFunction& cost) {
    if (nextLayer == nullptr) return;

    std::generate(nextLayer->neurons.begin(), nextLayer->neurons.end(), [&]() {
        return std::make_unique<Neuron>(&neurons, activation, cost); // dense layer, pass full vector
    });
}

void Layer::calculateNodeValues() {
    if (previousLayer) {
        for (auto& neuron : neurons) {
            neuron->calculate();
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
