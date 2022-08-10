#include "Layer.h"
#include <algorithm>
#include <numeric>

Layer::Layer(int32_t neuronCount) : neurons(neuronCount), layerSize(neuronCount) {}

void Layer::calculateNodeValues() {
    if (previousLayer) {
        for (auto& neuron : neurons) {
            neuron->calculate();
        }
    }
    if (nextLayer) nextLayer->calculateNodeValues();
}

void Layer::update(float learningRate) {
    if (!previousLayer) return; // don't update input nodes

    for (auto& neuron : neurons) neuron->update(learningRate);

    previousLayer->update(learningRate); // move onto next layer
}

size_t Layer::getWeightCount() {
    return std::accumulate(neurons.begin(), neurons.end(), (size_t)0, [](size_t sum, auto& neuron) {
        return sum + neuron->getWeightCount();
    });
}
